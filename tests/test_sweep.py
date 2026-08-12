import tempfile
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from picoformer.optim import split_param_groups_by_dtype
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
                        "sweep_optimizer=adamw",
                        "step_scheduler.local_batch_size=4",
                        "step_scheduler.global_batch_size=16",
                        "dataset.seq_len=128",
                        "sweep.num_tokens=10241",
                        "sweep.final_decay_ratio=0.4",
                        "optimizer.lr=0.0003",
                        f"checkpoint.checkpoint_dir={tmp_path}/checkpoints",
                    ],
                )

            destination = tmp_path / "automodel.yaml"
            write_automodel_config(cfg, destination)
            generated = OmegaConf.load(destination)

            self.assertNotIn("sweep", generated)
            self.assertNotIn("sweep_optimizer", generated)
            self.assertEqual(generated.optimizer._target_, "torch.optim.AdamW")
            self.assertEqual(generated.optimizer.lr, 0.0003)
            self.assertEqual(generated.lr_scheduler.lr_decay_style, "WSD")
            self.assertEqual(generated.lr_scheduler.lr_wsd_decay_style, "linear")
            self.assertAlmostEqual(generated.lr_scheduler.min_lr, 0.00003)
            self.assertEqual(generated.step_scheduler.local_batch_size, 4)
            self.assertEqual(generated.dataloader.batch_size, 4)
            self.assertEqual(generated.step_scheduler.max_steps, 6)
            self.assertEqual(generated.lr_scheduler.lr_decay_steps, 6)
            self.assertEqual(generated.lr_scheduler.wsd_decay_steps, 2)
            self.assertEqual(generated.lr_scheduler.lr_warmup_steps, 25)
            self.assertEqual(
                generated.checkpoint.checkpoint_dir, f"{tmp_path}/checkpoints"
            )

    def test_muon_optimizer_choice_is_materialized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config_dir = Path(__file__).parents[1] / "config"
            with initialize_config_dir(
                version_base="1.3", config_dir=str(config_dir.resolve())
            ):
                cfg = compose(
                    config_name="sweep",
                    overrides=[
                        "sweep_optimizer=muon",
                        "optimizer.lr=0.02",
                        f"checkpoint.checkpoint_dir={tmp_path}/checkpoints",
                    ],
                )

            destination = tmp_path / "automodel.yaml"
            write_automodel_config(cfg, destination)
            generated = OmegaConf.load(destination)

            self.assertNotIn("sweep_optimizer", generated)
            self.assertEqual(
                generated.optimizer._target_,
                "picoformer.optim.MixedPrecisionMuonConfig",
            )
            self.assertEqual(generated.optimizer.lr, 0.02)
            self.assertEqual(generated.optimizer.scalar_opt, "lion")
            self.assertIn("epsilon", generated.optimizer)
            self.assertNotIn("eps", generated.optimizer)

    def test_muon_parameter_groups_are_split_by_dtype(self) -> None:
        bf16_param = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
        fp32_param = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
        groups = split_param_groups_by_dtype(
            [{"params": [bf16_param, fp32_param], "algorithm": "lion"}]
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(
            {group["params"][0].dtype for group in groups},
            {torch.bfloat16, torch.float32},
        )
        self.assertTrue(
            all(len({param.dtype for param in group["params"]}) == 1 for group in groups)
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
