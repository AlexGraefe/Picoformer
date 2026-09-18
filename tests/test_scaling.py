import json
import tempfile
import unittest
from pathlib import Path

import optuna
from omegaconf import OmegaConf

from picoformer.analyze_scaling import (
    compute_budget,
    fit_power_law,
    fit_power_laws_by_model_size,
    near_optimal_trials,
    representative_hparams,
    scaling_points,
    tokens_for_compute,
)
from picoformer.scaling import (
    grid_search_space,
    local_batch_size_candidates,
    read_validation_loss,
    remaining_trial_budget,
)


class ScalingTest(unittest.TestCase):
    def test_flop_targets_expand_across_model_sizes(self) -> None:
        cfg = OmegaConf.create({
            "compute_flops": [600.0, 1200.0],
            "model_sizes": [{
                "name": "tiny",
                "num_parameters": 10,
                "architecture": {"hidden_size": 8},
            }],
        })
        points = scaling_points(cfg)
        self.assertEqual([point.num_tokens for point in points], [10, 20])
        self.assertEqual(tokens_for_compute(600.0, 10), 10)

    def test_grid_search_space_uses_configured_candidates(self) -> None:
        cfg = OmegaConf.create(
            {"search_space": {
                "learning_rate": [1e-4, 1e-3],
                "global_batch_size": [32, 64],
            }}
        )
        self.assertEqual(
            grid_search_space(cfg),
            {
                "learning_rate": [1e-4, 1e-3],
                "global_batch_size": [32, 64],
            },
        )

    def test_completed_grid_is_not_run_again(self) -> None:
        study = optuna.create_study(direction="minimize")
        for value in range(30):
            study.add_trial(optuna.trial.create_trial(value=float(value)))
        search_space = {
            "learning_rate": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            "global_batch_size": [32, 64, 128, 256, 512, 1024],
        }

        self.assertEqual(remaining_trial_budget(study, 40, search_space), 0)

    def test_resume_only_runs_remaining_trials(self) -> None:
        study = optuna.create_study(direction="minimize")
        for value in range(29):
            study.add_trial(optuna.trial.create_trial(value=float(value)))
        search_space = {
            "learning_rate": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            "global_batch_size": [32, 64, 128, 256, 512, 1024],
        }

        self.assertEqual(remaining_trial_budget(study, 40, search_space), 1)

    def test_compute_budget(self) -> None:
        self.assertEqual(compute_budget(1_000_000_000, 2_000_000_000), 1.2e19)

    def test_local_batch_size_candidates_halve_and_remain_compatible(self) -> None:
        self.assertEqual(local_batch_size_candidates(32, 128, 2), [32, 16, 8, 4, 2, 1])
        self.assertEqual(local_batch_size_candidates(32, 48, 2), [24, 12, 6, 3, 1])

    def test_fit_power_law_recovers_known_law(self) -> None:
        law = fit_power_law([1.0, 10.0, 100.0], [3.0, 30.0, 300.0])
        self.assertAlmostEqual(law.coefficient, 3.0)
        self.assertAlmostEqual(law.exponent, 1.0)
        self.assertAlmostEqual(law.r_squared, 1.0)

    def test_fit_power_law_handles_constant_values(self) -> None:
        law = fit_power_law([6e15, 6e16, 6e17], [32.0, 32.0, 32.0])
        self.assertAlmostEqual(law.coefficient, 32.0)
        self.assertAlmostEqual(law.exponent, 0.0)
        self.assertEqual(law.r_squared, 1.0)

    def test_fits_separate_power_law_for_each_model_size(self) -> None:
        rows = []
        for size, coefficient in ((10, 2.0), (20, 5.0)):
            for compute in (100.0, 1_000.0, 10_000.0):
                rows.append(
                    {
                        "num_parameters": size,
                        "compute_flops": compute,
                        "near_optimal_learning_rate": coefficient * compute,
                        "near_optimal_global_batch_size": coefficient * compute,
                    }
                )

        laws = fit_power_laws_by_model_size(rows)

        self.assertEqual(set(laws), {10, 20})
        self.assertAlmostEqual(laws[10]["learning_rate"].coefficient, 2.0)
        self.assertAlmostEqual(laws[20]["learning_rate"].coefficient, 5.0)

    def test_each_model_size_requires_two_scale_points(self) -> None:
        row = {
            "num_parameters": 10,
            "compute_flops": 100.0,
            "near_optimal_learning_rate": 0.1,
            "near_optimal_global_batch_size": 32,
        }
        with self.assertRaisesRegex(ValueError, "model size 10"):
            fit_power_laws_by_model_size([row])

    def test_near_optimal_region_and_representative(self) -> None:
        study = optuna.create_study(direction="minimize")
        distributions = {
            "learning_rate": optuna.distributions.FloatDistribution(1e-4, 1e-1, log=True),
            "weight_decay": optuna.distributions.FloatDistribution(1e-3, 1.0, log=True),
            "global_batch_size": optuna.distributions.CategoricalDistribution([32, 128]),
        }
        study.add_trial(optuna.trial.create_trial(
            params={"learning_rate": 1e-3, "weight_decay": 0.01, "global_batch_size": 32},
            distributions=distributions, value=2.0,
        ))
        study.add_trial(optuna.trial.create_trial(
            params={"learning_rate": 1e-2, "weight_decay": 0.1, "global_batch_size": 128},
            distributions=distributions, value=2.004,
        ))
        study.add_trial(optuna.trial.create_trial(
            params={"learning_rate": 5e-2, "weight_decay": 0.5, "global_batch_size": 128},
            distributions=distributions, value=2.1,
        ))
        selected = near_optimal_trials(study.trials, 0.0025)
        self.assertEqual(len(selected), 2)
        representative = representative_hparams(selected)
        self.assertAlmostEqual(representative["learning_rate"], (1e-5) ** 0.5)
        self.assertAlmostEqual(representative["global_batch_size"], 64.0)
        self.assertNotIn("weight_decay", representative)

    def test_read_final_validation_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.jsonl"
            path.write_text(
                json.dumps({"step": 1, "val_loss": 2.5}) + "\n"
                + json.dumps({"step": 2, "val_loss": 2.25}) + "\n"
            )
            self.assertEqual(read_validation_loss(path), 2.25)


if __name__ == "__main__":
    unittest.main()
