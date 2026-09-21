"""Hydra launcher for NeMo AutoModel hyperparameter sweeps."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def _patch_argparse_for_hydra() -> None:
    """Allow Hydra 1.3's lazy help value on Python 3.14."""
    if sys.version_info < (3, 14):
        return

    original = argparse.ArgumentParser._check_help

    def check_help(parser: argparse.ArgumentParser, action: argparse.Action) -> None:
        if action.help is not None and not isinstance(action.help, str):
            return
        original(parser, action)

    argparse.ArgumentParser._check_help = check_help


_patch_argparse_for_hydra()

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "config")


def automodel_command(config_path: Path, nproc_per_node: int) -> list[str]:
    """Build the AutoModel command using the executable from this environment."""
    executable = Path(sys.executable).with_name("automodel")
    if not executable.is_file():
        discovered = shutil.which("automodel")
        if discovered is None:
            raise FileNotFoundError(
                "Could not find the 'automodel' executable in the active environment"
            )
        executable = Path(discovered)

    return [
        str(executable),
        str(config_path),
        "--nproc-per-node",
        str(nproc_per_node),
    ]


def configure_token_schedule(
    training_cfg: DictConfig,
    num_tokens: int,
    final_decay_ratio: float,
    final_lr_percentage: float,
) -> None:
    """Derive optimizer-step and WSD schedules from a training token budget."""
    if not 0.0 <= final_lr_percentage <= 100.0:
        raise ValueError("final_lr_percentage must be between 0 and 100")
    training_cfg.lr_scheduler.min_lr = (
        float(training_cfg.optimizer.lr) * final_lr_percentage / 100.0
    )

    global_batch_size = int(training_cfg.step_scheduler.global_batch_size)
    sequence_length = int(training_cfg.dataset.seq_len)
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if global_batch_size <= 0:
        raise ValueError("step_scheduler.global_batch_size must be positive")
    if sequence_length <= 0:
        raise ValueError("dataset.seq_len must be positive")
    if not 0.0 <= final_decay_ratio <= 1.0:
        raise ValueError("final_decay_ratio must be between 0 and 1")

    tokens_per_step = global_batch_size * sequence_length
    max_steps = (num_tokens + tokens_per_step - 1) // tokens_per_step
    warmup_cap = int(training_cfg.lr_scheduler.lr_warmup_steps)
    if warmup_cap < 0:
        raise ValueError("lr_scheduler.lr_warmup_steps must be non-negative")
    warmup_steps = min(math.ceil(max_steps * 0.2), warmup_cap)
    final_decay_steps = round(max_steps * final_decay_ratio)
    if final_decay_ratio > 0.0:
        final_decay_steps = max(1, final_decay_steps)

    training_cfg.step_scheduler.max_steps = max_steps
    training_cfg.lr_scheduler.lr_decay_steps = max_steps
    training_cfg.lr_scheduler.lr_warmup_steps = warmup_steps
    training_cfg.lr_scheduler.wsd_decay_steps = final_decay_steps


def write_automodel_config(cfg: DictConfig, destination: Path) -> None:
    """Write a resolved training config without Hydra launcher-only settings."""
    training_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    # Optimizer choices live outside the inherited AutoModel config so Hydra can
    # switch schemas cleanly (AdamW's ``eps`` vs Muon's ``epsilon``, for example).
    training_cfg["optimizer"] = OmegaConf.to_container(
        training_cfg["sweep_optimizer"], resolve=True
    )
    configure_token_schedule(
        training_cfg,
        num_tokens=int(training_cfg.sweep.num_tokens),
        final_decay_ratio=float(training_cfg.sweep.final_decay_ratio),
        final_lr_percentage=float(training_cfg.sweep.final_lr_percentage),
    )
    # Validation is deliberately run at the last optimizer step.  Optuna uses
    # this independent-dataset loss as its objective.
    if "validation_dataset" in training_cfg:
        training_cfg.step_scheduler.val_every_steps = training_cfg.step_scheduler.max_steps
    del training_cfg["sweep_optimizer"]
    del training_cfg["sweep"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(OmegaConf.to_yaml(training_cfg, resolve=True))


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="sweep")
def main(cfg: DictConfig) -> None:
    """Materialize this Hydra job as YAML and launch it with AutoModel."""
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    config_path = output_dir / str(cfg.sweep.generated_config_name)
    write_automodel_config(cfg, config_path)

    command = automodel_command(config_path, int(cfg.sweep.nproc_per_node))
    print(f"Generated AutoModel config: {config_path}", flush=True)
    print(f"Launching: {' '.join(command)}", flush=True)

    if not bool(cfg.sweep.dry_run):
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
