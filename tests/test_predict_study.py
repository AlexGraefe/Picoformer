from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import optuna
from omegaconf import OmegaConf

from picoformer.predict_study import fit_loss, predict_study, read_curve, select_best_curve


def write_curve(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = root / "automodel.yaml"
    OmegaConf.save(OmegaConf.create({
        "step_scheduler": {"global_batch_size": 4, "max_steps": 100},
        "dataset": {"seq_len": 16}, "optimizer": {"lr": 0.001},
        "lr_scheduler": {"lr_decay_style": "WSD", "lr_warmup_steps": 10,
                         "lr_decay_steps": 100, "wsd_decay_steps": 20},
        "checkpoint": {"checkpoint_dir": "/unavailable/copied/checkpoints"},
    }), config)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()
    rows = []
    for step in [0, 9, 19, 29, 39, 49, 59, 69, 79, 80, 99]:
        loss = 2 + 3 * ((step + 1) / 20) ** -0.6
        if step >= 80:
            loss = 0.1  # Deliberately destroy a fit that includes cooldown.
        rows.append({"step": step, "val_loss": loss,
                     "lr": 0.001 if 10 <= step < 80 else 0.0001,
                     "num_label_tokens": 999999})
    (checkpoints / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return config


class PredictStudyTest(unittest.TestCase):
    def test_best_selection_requires_r_squared_threshold(self):
        curves = [
            {"trial": 0, "predicted_loss": 1.0, "fit": {"r_squared": 0.949999}},
            {"trial": 1, "predicted_loss": 2.0, "fit": {"r_squared": 0.95}},
            {"trial": 2, "predicted_loss": 3.0, "fit": {"r_squared": 0.99}},
        ]
        self.assertEqual(select_best_curve(curves)["trial"], 1)
        self.assertIsNone(select_best_curve(curves[:1]))
        self.assertIsNone(select_best_curve([]))

    def test_power_law_recovers_unseen_future(self):
        tokens = np.geomspace(1e6, 1e8, 30)
        losses = 2.1 + 3.2 * (tokens / 1e6) ** -0.45
        fit = fit_loss(tokens, losses)
        self.assertAlmostEqual(fit.floor, 2.1, places=5)
        self.assertAlmostEqual(fit.exponent, 0.45, places=5)
        self.assertAlmostEqual(fit.r_squared, 1.0, places=10)
        self.assertAlmostEqual(float(fit.predict(np.asarray(1e9))),
                               2.1 + 3.2 * 1000 ** -0.45, places=5)

    def test_r_squared_uses_loss_space_residuals(self):
        tokens = np.geomspace(1e6, 1e8, 30)
        losses = 2.1 + 3.2 * (tokens / 1e6) ** -0.45 + 0.15 * np.sin(np.arange(30))
        fit = fit_loss(tokens, losses)
        expected = 1 - np.sum((losses - fit.predict(tokens)) ** 2) / np.sum((losses - losses.mean()) ** 2)
        self.assertAlmostEqual(fit.r_squared, expected, places=12)
        self.assertGreater(fit.r_squared, 0)
        self.assertLess(fit.r_squared, 1)

    def test_declines_that_would_require_negative_floor_stay_nonnegative(self):
        tokens = np.geomspace(1, 4, 20)
        fit = fit_loss(tokens, 5 - np.log(tokens))
        self.assertGreaterEqual(fit.floor, 0)
        self.assertGreaterEqual(float(fit.predict(np.asarray(1e12))), 0)

    def test_invalid_and_uninformative_data_rejected(self):
        for tokens, loss in [([1, 2, 3], [4, 3, 2]),
                             ([1, 2, 3, 4], [1, 2, 3, 4]),
                             ([1, 2, 3, 4], [2, 2, 2, 2]),
                             ([1, 1, 2, 3], [5, 4, 3, 2]),
                             ([1, 2, 3, 4], [5, 4, 3, float("nan")])]:
            with self.subTest(tokens=tokens, loss=loss), self.assertRaises(ValueError):
                fit_loss(np.array(tokens), np.array(loss))

    def test_tokens_and_schedule_exclude_cooldown(self):
        with tempfile.TemporaryDirectory() as directory:
            config = write_curve(Path(directory))
            curve = read_curve(config)
            self.assertEqual(curve["points"][0]["tokens"], 64)
            self.assertEqual(curve["decay_start_tokens"], 80 * 64)
            reasons = {p["step"]: p["reason"] for p in curve["points"]}
            self.assertEqual(reasons[9], "warmup")
            self.assertEqual(reasons[79], "fit")
            self.assertEqual(reasons[80], "final_decay")
            selected = [p for p in curve["points"] if p["used_for_fit"]]
            fit = fit_loss(np.array([p["tokens"] for p in selected]),
                           np.array([p["loss"] for p in selected]))
            self.assertAlmostEqual(fit.floor, 2, places=5)
            self.assertAlmostEqual(fit.exponent, 0.6, places=5)
            self.assertAlmostEqual(fit.r_squared, 1.0, places=10)
            later = read_curve(config, fit_start_fraction=0.4)
            self.assertEqual([p["step"] for p in later["points"] if p["used_for_fit"]],
                             [39, 49, 59, 69, 79])

    def test_study_outputs_and_attempt_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "study.db"
            study = optuna.create_study(storage=f"sqlite:///{database}", study_name="test")
            study.add_trial(optuna.trial.create_trial(
                params={"learning_rate": 0.001, "global_batch_size": 4},
                distributions={"learning_rate": optuna.distributions.FloatDistribution(0.001, 0.001),
                               "global_batch_size": optuna.distributions.CategoricalDistribution([4])},
                value=0.1, user_attrs={"batch_size_attempts": 1}))
            write_curve(root / "trial_0000" / "attempt_00_local_batch_2")
            # An unrelated later attempt must not replace the recorded successful one.
            unrelated = root / "trial_0000" / "attempt_01_local_batch_1"
            unrelated.mkdir()
            (unrelated / "automodel.yaml").write_text("invalid: true")
            # Trial 1 has a better final objective but a worse plateau forecast.
            study.add_trial(optuna.trial.create_trial(
                params={"learning_rate": 0.002, "global_batch_size": 4},
                distributions={"learning_rate": optuna.distributions.FloatDistribution(0.001, 0.002),
                               "global_batch_size": optuna.distributions.CategoricalDistribution([4])},
                value=0.01, user_attrs={"batch_size_attempts": 1}))
            other_config = write_curve(root / "trial_0001" / "attempt_00_local_batch_2")
            cfg = OmegaConf.load(other_config)
            cfg.optimizer.lr = 0.002
            OmegaConf.save(cfg, other_config)
            metrics = other_config.parent / "checkpoints" / "validation.jsonl"
            rows = [json.loads(line) for line in metrics.read_text().splitlines()]
            for row in rows:
                row["val_loss"] += 1.0
                row["lr"] *= 2
            metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                output = predict_study(database, root / "out")
            self.assertGreater(output.stat().st_size, 0)
            data = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(data["target_tokens"], 12800)
            self.assertEqual(data["skipped"], [])
            self.assertEqual(data["min_selection_r_squared"], 0.95)
            self.assertAlmostEqual(data["curves"][0]["predicted_loss"], 2 + 3 * 10 ** -0.6, places=5)
            self.assertEqual(len(output.with_suffix(".csv").read_text().splitlines()), 401)
            self.assertEqual(study.best_trial.number, 1)
            best = data["best_at_target"]
            self.assertEqual(best["trial"], 0)
            self.assertEqual(best["target_tokens"], 12800)
            self.assertEqual(best["params"], {"learning_rate": 0.001, "global_batch_size": 4})
            self.assertEqual(best["predicted_loss"], data["curves"][0]["predicted_loss"])
            self.assertIn("at 12,800 tokens", stdout.getvalue())
            self.assertIn("Trial #0: predicted validation loss", stdout.getvalue())
            self.assertIn("learning_rate = 0.001", stdout.getvalue())
            for curve in data["curves"]:
                self.assertAlmostEqual(curve["fit"]["r_squared"], 1.0, places=10)
                self.assertIn(f"Trial #{curve['trial']}: fit R² = 1.000000", stdout.getvalue())
            with redirect_stdout(io.StringIO()):
                selected = predict_study(database, root / "selected", trial_numbers=[1])
            self.assertEqual(json.loads(selected.with_suffix(".json").read_text())["best_at_target"]["trial"], 1)
            stdout = io.StringIO()
            with patch("picoformer.predict_study.select_best_curve", return_value=None), redirect_stdout(stdout):
                no_best = predict_study(database, root / "no_best")
            self.assertIsNone(json.loads(no_best.with_suffix(".json").read_text())["best_at_target"])
            self.assertIn("no fits have R² >= 0.95", stdout.getvalue())
            self.assertTrue(no_best.is_file())
            with self.assertRaisesRegex(ValueError, "must exceed"):
                predict_study(database, root / "out", target_tokens=6400)
            with self.assertRaisesRegex(ValueError, "not completed"):
                predict_study(database, root / "out", trial_numbers=[9])


if __name__ == "__main__":
    unittest.main()
