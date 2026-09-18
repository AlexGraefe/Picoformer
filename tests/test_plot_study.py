import tempfile
import unittest
from pathlib import Path

import optuna

from picoformer.plot_study import _grouped_trials, load_completed_trials, plot_study


class PlotStudyTest(unittest.TestCase):
    def test_grouped_trials_are_sorted_along_x_axis(self) -> None:
        trials = [
            optuna.trial.create_trial(
                params={"learning_rate": learning_rate, "global_batch_size": batch_size},
                distributions={
                    "learning_rate": optuna.distributions.FloatDistribution(1e-4, 1e-2),
                    "global_batch_size": optuna.distributions.CategoricalDistribution(
                        [32, 64]
                    ),
                },
                value=loss,
            )
            for learning_rate, batch_size, loss in (
                (1e-3, 64, 2.1),
                (1e-4, 32, 2.4),
                (1e-3, 32, 2.3),
                (1e-4, 64, 2.2),
            )
        ]

        by_learning_rate = _grouped_trials(
            trials, "global_batch_size", "learning_rate"
        )
        self.assertEqual([group for group, _ in by_learning_rate], [1e-4, 1e-3])
        self.assertEqual(
            [trial.params["global_batch_size"] for trial in by_learning_rate[0][1]],
            [32, 64],
        )

        by_batch_size = _grouped_trials(
            trials, "learning_rate", "global_batch_size"
        )
        self.assertEqual([group for group, _ in by_batch_size], [32.0, 64.0])
        self.assertEqual(
            [trial.params["learning_rate"] for trial in by_batch_size[0][1]],
            [1e-4, 1e-3],
        )

    def test_load_and_plot_study(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            database = directory_path / "study.db"
            study = optuna.create_study(
                study_name="tiny_100p_1000t",
                storage=f"sqlite:///{database}",
                direction="minimize",
            )
            study.enqueue_trial({"learning_rate": 1e-3, "global_batch_size": 32})

            def objective(trial: optuna.Trial) -> float:
                trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
                trial.suggest_categorical("global_batch_size", [32, 64])
                return 2.5

            study.optimize(objective, n_trials=1)

            name, trials = load_completed_trials(database)
            self.assertEqual(name, "tiny_100p_1000t")
            self.assertEqual(len(trials), 1)

            output = plot_study(database, directory_path / "plot.png")
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
