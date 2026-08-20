import json
import tempfile
import unittest
from pathlib import Path

import optuna

from picoformer.analyze_scaling import (
    compute_budget,
    fit_power_law,
    fit_power_laws_by_model_size,
    near_optimal_trials,
    representative_hparams,
)
from picoformer.scaling import (
    local_batch_size_candidates,
    read_validation_loss,
)


class ScalingTest(unittest.TestCase):
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
                        "near_optimal_weight_decay": coefficient * compute,
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
            "near_optimal_weight_decay": 0.1,
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
