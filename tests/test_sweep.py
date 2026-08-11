import tempfile
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from picoformer.sweep import automodel_command, write_automodel_config


class SweepTest(unittest.TestCase):
    def test_generated_config_resolves_batch_size_and_removes_sweep_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_dir = Path(__file__).parents[1] / "config"
            with initialize_config_dir(
                version_base="1.3", config_dir=str(config_dir.resolve())
            ):
                cfg = compose(
                    config_name="sweep",
                    overrides=[
                        "step_scheduler.local_batch_size=4",
                        "optimizer.lr=0.0003",
                        f"checkpoint.checkpoint_dir={tmp_path}/checkpoints",
                    ],
                )

            destination = tmp_path / "automodel.yaml"
            write_automodel_config(cfg, destination)
            generated = OmegaConf.load(destination)

            self.assertNotIn("sweep", generated)
            self.assertEqual(generated.optimizer.lr, 0.0003)
            self.assertEqual(generated.step_scheduler.local_batch_size, 4)
            self.assertEqual(generated.dataloader.batch_size, 4)
            self.assertEqual(
                generated.checkpoint.checkpoint_dir, f"{tmp_path}/checkpoints"
            )

    def test_automodel_command_uses_requested_process_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "automodel.yaml"
            command = automodel_command(config_path, nproc_per_node=2)

            self.assertEqual(Path(command[0]).name, "automodel")
            self.assertEqual(
                command[1:],
                [str(config_path), "--nproc-per-node", "2"],
            )


if __name__ == "__main__":
    unittest.main()
