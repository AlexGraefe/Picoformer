import tempfile
import unittest
from pathlib import Path

import optuna

from picoformer.plot_study import load_completed_trials, plot_study


class PlotStudyTest(unittest.TestCase):
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
