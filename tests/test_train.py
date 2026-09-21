import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from picoformer.train import main, write_automodel_config


CONFIG = Path(__file__).resolve().parents[1] / "config/optimal_training_config.yaml"


class TrainTest(unittest.TestCase):
    def make_checkpoint(self, root: Path, step: int) -> Path:
        checkpoint = root / f"epoch_0_step_{step - 1}"
        (checkpoint / "model").mkdir(parents=True)
        (checkpoint / "optim").mkdir()
        torch.save({"step": step, "epoch": 0}, checkpoint / "step_scheduler.pt")
        return checkpoint

    def test_ablation_preserves_decay_slope_and_isolates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            self.make_checkpoint(root, 10)
            latest = self.make_checkpoint(root, 100)
            (root / "epoch_0_step_999").mkdir()  # Incomplete save is ignored.
            cfg = OmegaConf.load(CONFIG)
            cfg.checkpoint.checkpoint_dir = str(root)
            cfg.checkpoint.restore_from = "stale_checkpoint"
            cfg.num_tokens_B = 0.032768  # 1,000 steps; original decay = 100.
            cfg.wandb.extra = {"id": "original", "resume": "must", "dir": str(root / "logs")}
            original = OmegaConf.to_yaml(cfg)
            destination = Path(directory) / "train_ablation/automodel.yaml"
            write_automodel_config(cfg, destination, mode="ablation", num_tokens_b=200 * 32768 / 1e9)
            generated = OmegaConf.load(destination)
            self.assertEqual(generated.checkpoint.restore_from, str(latest))
            self.assertEqual(generated.checkpoint.checkpoint_dir, str(root) + "_ablation")
            self.assertEqual(generated.step_scheduler.max_steps, 300)
            self.assertEqual(generated.lr_scheduler.lr_decay_steps, 300)
            self.assertEqual(generated.lr_scheduler.wsd_decay_steps, 100)
            self.assertEqual(generated.lr_scheduler.lr_warmup_steps, 0)
            self.assertTrue(generated.lr_scheduler.override_opt_param_scheduler)
            self.assertFalse(generated.lr_scheduler.use_checkpoint_opt_param_scheduler)
            self.assertNotEqual(generated.wandb.id, "original")
            self.assertEqual(generated.wandb.resume, "never")
            self.assertEqual(generated.wandb.dir, str(root / "logs_ablation"))
            self.assertTrue(generated.wandb.name.endswith("_ablation"))
            self.assertEqual(OmegaConf.to_yaml(cfg), original)

    def test_ablation_restored_scheduler_reaches_minimum_at_budget(self) -> None:
        from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler

        def scheduler(cfg):
            optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=cfg.optimizer.lr)
            return OptimizerParamScheduler(
                optimizer, init_lr=0, max_lr=cfg.optimizer.lr,
                start_wd=0, end_wd=0, wd_incr_steps=1, wd_incr_style="constant",
                **OmegaConf.to_container(cfg.lr_scheduler, resolve=True),
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            self.make_checkpoint(root, 100)
            cfg = OmegaConf.load(CONFIG)
            cfg.checkpoint.checkpoint_dir = str(root)
            cfg.num_tokens_B = 0.032768
            destination = Path(directory) / "automodel.yaml"
            write_automodel_config(cfg, destination)
            original_scheduler = scheduler(OmegaConf.load(destination))
            original_scheduler.step(100)
            saved = original_scheduler.state_dict()
            for steps in (1, 50, 100, 200):
                with self.subTest(steps=steps):
                    # Fractional step budgets round up.
                    write_automodel_config(cfg, destination, mode="ablation", num_tokens_b=((steps - 1) * 32768 + 1) / 1e9)
                    generated = OmegaConf.load(destination)
                    resumed = scheduler(generated)
                    resumed.load_state_dict(saved)
                    self.assertEqual(resumed.num_steps, 100)
                    self.assertEqual(resumed.lr_decay_steps, 100 + steps)
                    decay_steps = min(100, steps)
                    self.assertEqual(resumed.wsd_decay_steps, decay_steps)
                    self.assertAlmostEqual(resumed.optimizer.param_groups[0]["lr"], cfg.optimizer.lr)
                    resumed.step(steps - decay_steps)
                    self.assertAlmostEqual(resumed.optimizer.param_groups[0]["lr"], cfg.optimizer.lr)
                    resumed.step(decay_steps)
                    self.assertAlmostEqual(resumed.optimizer.param_groups[0]["lr"], generated.lr_scheduler.min_lr)

    def test_ablation_latest_pointers_and_explicit_decay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            older = self.make_checkpoint(root, 10)
            newer = self.make_checkpoint(root, 20)
            cfg = OmegaConf.load(CONFIG)
            cfg.checkpoint.checkpoint_dir = str(root)
            cfg.lr_scheduler.wsd_decay_steps = 7
            destination = Path(directory) / "automodel.yaml"
            for pointer in ("LATEST", "LATEST.txt"):
                with self.subTest(pointer=pointer):
                    if pointer == "LATEST":
                        (root / pointer).symlink_to(older.name)
                    else:
                        (root / "LATEST").unlink()
                        (root / pointer).write_text(older.name)
                    write_automodel_config(cfg, destination, mode="ablation", num_tokens_b=10 * 32768 / 1e9)
                    generated = OmegaConf.load(destination)
                    self.assertEqual(generated.checkpoint.restore_from, str(older))
                    self.assertEqual(generated.lr_scheduler.wsd_decay_steps, 7)
            write_automodel_config(cfg, destination, mode="ablation", num_tokens_b=10 * 32768 / 1e9, resume_from=newer)
            self.assertEqual(OmegaConf.load(destination).checkpoint.restore_from, str(newer))

    def test_ablation_rejects_missing_checkpoint_or_invalid_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = OmegaConf.load(CONFIG)
            cfg.checkpoint.checkpoint_dir = directory
            destination = Path(directory) / "automodel.yaml"
            for budget in (None, 0, -1, float("inf"), float("nan"), True, 0.5):
                with self.subTest(budget=budget), self.assertRaises(ValueError):
                    write_automodel_config(cfg, destination, mode="ablation", num_tokens_b=budget)
                self.assertFalse(destination.exists())

    def test_ablation_accepts_resolved_training_config_and_creates_new_run_each_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            self.make_checkpoint(root, 100)
            cfg = OmegaConf.load(CONFIG)
            cfg.checkpoint.checkpoint_dir = str(root)
            destination = Path(directory) / "automodel.yaml"
            write_automodel_config(cfg, destination)
            resolved = OmegaConf.load(destination)
            ids = []
            for _ in range(2):
                write_automodel_config(resolved, destination, mode="ablation", num_tokens_b=0.000032768)
                generated = OmegaConf.load(destination)
                self.assertEqual(generated.lr_scheduler.min_lr, resolved.lr_scheduler.min_lr)
                self.assertEqual(generated.lr_scheduler.wsd_decay_steps, 1)
                ids.append(generated.wandb.id)
            self.assertNotEqual(*ids)

    def test_ablation_cli_separates_generated_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            self.make_checkpoint(root, 100)
            cfg = OmegaConf.load(CONFIG)
            cfg.checkpoint.checkpoint_dir = str(root)
            config = Path(directory) / "input.yaml"
            OmegaConf.save(cfg, config)
            output = Path(directory) / "train/automodel.yaml"
            with (
                patch.object(sys, "argv", ["train", "--config", str(config), "--output-config", str(output),
                                          "--mode", "ablation", "--num-tokens-b", "0.5", "--dry-run"]),
                patch("picoformer.train.subprocess.run") as run,
            ):
                main()
            run.assert_not_called()
            self.assertFalse(output.exists())
            generated = OmegaConf.load(Path(directory) / "train_ablation/automodel.yaml")
            self.assertEqual(generated.step_scheduler.max_steps, 100 + 15259)
            self.assertEqual(generated.wandb.dir, str(Path(directory) / "train_ablation"))

    def test_explicit_resume_overrides_config_without_mutating_it(self) -> None:
        cfg = OmegaConf.load(CONFIG)
        cfg.checkpoint.restore_from = "previous_checkpoint"
        cfg.checkpoint.enabled = False
        original = OmegaConf.to_yaml(cfg)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            checkpoint = Path(directory) / "epoch_0_step_499"
            checkpoint.mkdir()
            destination = Path(directory) / "automodel.yaml"
            write_automodel_config(
                cfg, destination, resume_from=checkpoint.relative_to(Path.cwd())
            )
            generated = OmegaConf.load(destination)
        self.assertEqual(generated.checkpoint.restore_from, str(checkpoint.resolve()))
        self.assertTrue(generated.checkpoint.enabled)
        self.assertEqual(generated.checkpoint.checkpoint_dir, cfg.checkpoint.checkpoint_dir)
        self.assertEqual(OmegaConf.to_yaml(cfg), original)

    def test_configured_resume_is_preserved_without_cli_override(self) -> None:
        cfg = OmegaConf.load(CONFIG)
        cfg.checkpoint.restore_from = "epoch_0_step_499"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "automodel.yaml"
            write_automodel_config(cfg, destination)
            generated = OmegaConf.load(destination)
        self.assertEqual(generated.checkpoint.restore_from, "epoch_0_step_499")

    def test_invalid_resume_fails_before_writing_or_launching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "automodel.yaml"
            regular_file = Path(directory) / "weights.pt"
            regular_file.touch()
            for checkpoint in (Path(directory) / "missing", regular_file):
                with (
                    self.subTest(checkpoint=checkpoint),
                    patch.object(sys, "argv", [
                        "main.py", "--config", str(CONFIG),
                        "--output-config", str(destination),
                        "--resume-from", str(checkpoint),
                    ]),
                    patch("picoformer.train.subprocess.run") as run,
                    self.assertRaises(SystemExit) as error,
                ):
                    main()
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(destination.exists())
                run.assert_not_called()

    def test_main_py_forwards_resume_to_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "epoch_0_step_499"
            checkpoint.mkdir()
            destination = Path(directory) / "automodel.yaml"
            with (
                patch.object(sys, "argv", [
                    "main.py", "--config", str(CONFIG),
                    "--output-config", str(destination),
                    "--resume-from", str(checkpoint),
                ]),
                patch("picoformer.train.automodel_command", return_value=["automodel"]),
                patch("picoformer.train.subprocess.run") as run,
            ):
                runpy.run_path(str(CONFIG.parents[1] / "main.py"), run_name="__main__")
                generated = OmegaConf.load(destination)
                run.assert_called_once_with(["automodel"], check=True)
            self.assertEqual(generated.checkpoint.restore_from, str(checkpoint.resolve()))

    def test_fractional_budget_rounds_up_and_synchronizes_batches(self) -> None:
        cfg = OmegaConf.load(CONFIG)
        cfg.num_tokens_B = 0.000010241  # 10,241 tokens: just over five full steps.
        cfg.dataset.seq_len = 128
        cfg.step_scheduler.local_batch_size = 4
        cfg.dataloader.batch_size = 99
        cfg.validation_dataloader.batch_size = 99
        cfg.final_decay_ratio = 0.4
        cfg.optimizer.lr = 0.003
        original = OmegaConf.to_yaml(cfg)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested/automodel.yaml"
            write_automodel_config(cfg, destination)
            generated = OmegaConf.load(destination)

        self.assertEqual(OmegaConf.to_yaml(cfg), original)
        self.assertEqual(generated.step_scheduler.max_steps, 6)
        self.assertEqual(generated.lr_scheduler.lr_decay_steps, 6)
        self.assertEqual(generated.lr_scheduler.lr_warmup_steps, 2)
        self.assertEqual(generated.lr_scheduler.wsd_decay_steps, 2)
        self.assertAlmostEqual(generated.lr_scheduler.min_lr, 0.00003)
        self.assertEqual(generated.dataloader.batch_size, 4)
        self.assertEqual(generated.validation_dataloader.batch_size, 4)
        self.assertEqual(generated.step_scheduler.val_every_steps, 500)
        self.assertEqual(generated.step_scheduler.ckpt_every_steps, 500)
        for key in ("num_tokens_B", "final_decay_ratio", "final_lr_percentage"):
            self.assertNotIn(key, generated)

    def test_budget_counts_global_batches_independently_of_local_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "automodel.yaml"
            for local_batch in (1, 4, 8):
                with self.subTest(local_batch=local_batch):
                    cfg = OmegaConf.load(CONFIG)
                    cfg.num_tokens_B = 1
                    cfg.step_scheduler.local_batch_size = local_batch
                    cfg.lr_scheduler.lr_warmup_steps = 10
                    write_automodel_config(cfg, destination)
                    generated = OmegaConf.load(destination)
                    self.assertEqual(generated.step_scheduler.max_steps, 30518)
                    self.assertEqual(generated.lr_scheduler.wsd_decay_steps, 3052)
                    self.assertEqual(generated.lr_scheduler.lr_warmup_steps, 10)
                    self.assertEqual(generated.dataloader.batch_size, local_batch)

    def test_one_step_budget_without_validation(self) -> None:
        cfg = OmegaConf.load(CONFIG)
        cfg.num_tokens_B = 1e-9
        del cfg["validation_dataset"]
        del cfg["validation_dataloader"]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "automodel.yaml"
            write_automodel_config(cfg, destination)
            generated = OmegaConf.load(destination)
        self.assertEqual(generated.step_scheduler.max_steps, 1)
        self.assertEqual(generated.lr_scheduler.lr_warmup_steps, 1)
        self.assertEqual(generated.lr_scheduler.wsd_decay_steps, 1)

    def test_invalid_settings_fail_before_writing(self) -> None:
        invalid = [
            ("num_tokens_B", 0),
            ("num_tokens_B", -1),
            ("num_tokens_B", float("inf")),
            ("num_tokens_B", float("nan")),
            ("step_scheduler.local_batch_size", 0),
            ("step_scheduler.local_batch_size", 3),
            ("step_scheduler.global_batch_size", 0),
            ("dataset.seq_len", 0),
            ("lr_scheduler.lr_warmup_steps", -1),
            ("final_decay_ratio", 1.1),
            ("final_lr_percentage", -1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "automodel.yaml"
            for key, value in invalid:
                with self.subTest(key=key, value=value):
                    cfg = OmegaConf.load(CONFIG)
                    OmegaConf.update(cfg, key, value)
                    with self.assertRaises(ValueError):
                        write_automodel_config(cfg, destination)
                    self.assertFalse(destination.exists())

    def test_cli_dry_run_and_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "automodel.yaml"
            argv = [
                "picoformer-train", "--config", str(CONFIG),
                "--output-config", str(destination), "--nproc-per-node", "2",
            ]
            with (
                patch.object(sys, "argv", argv + ["--dry-run"]),
                patch("picoformer.train.automodel_command") as command,
                patch("picoformer.train.subprocess.run") as run,
            ):
                main()
                self.assertTrue(destination.is_file())
                command.assert_not_called()
                run.assert_not_called()
            with (
                patch.object(sys, "argv", argv),
                patch("picoformer.train.automodel_command", return_value=["automodel"]) as command,
                patch("picoformer.train.subprocess.run") as run,
            ):
                main()
                command.assert_called_once_with(destination, 2)
                run.assert_called_once_with(["automodel"], check=True)


if __name__ == "__main__":
    unittest.main()
