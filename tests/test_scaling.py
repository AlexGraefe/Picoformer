import json
import tempfile
import unittest
from pathlib import Path

import optuna

from picoformer.scaling import (
    compute_budget,
    fit_power_law,
    local_batch_size_candidates,
    near_optimal_trials,
    read_validation_loss,
    representative_hparams,
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
