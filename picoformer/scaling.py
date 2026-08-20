"""Optuna-based hyperparameter scaling-law experiments.

Each architecture/FLOP-target pair gets an independent Optuna study.  This is
important: a single study over compute budgets would merely learn that larger
budgets tend to have lower loss rather than finding good hyperparameters at
each scale.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import optuna
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from picoformer.analyze_scaling import (
    analyze,
    compute_budget,
    fit_power_law,
    fit_power_laws_by_model_size as _fit_power_laws_by_model_size,
    near_optimal_trials,
    representative_hparams,
    scaling_points,
    study_name as _study_name,
)
from picoformer.sweep import automodel_command, write_automodel_config


def read_validation_loss(path: Path) -> float:
    """Read the final finite validation loss emitted by NeMo AutoModel."""
    if not path.is_file():
        raise FileNotFoundError(f"validation metrics were not written: {path}")
    losses: list[float] = []
    for line in path.read_text().splitlines():
        if line.strip():
            value = float(json.loads(line)["val_loss"])
            if math.isfinite(value):
                losses.append(value)
    if not losses:
        raise ValueError(f"no finite val_loss found in {path}")
    return losses[-1]


def _suggest_global_batch_size(trial: optuna.Trial, choices: list[int]) -> int:
    value = int(trial.suggest_categorical("global_batch_size", choices))
    return value


def grid_search_space(
    cfg: DictConfig, scale: DictConfig | None = None
) -> dict[str, list[str | float | int]]:
    """Return the discrete search space used by Optuna's grid sampler."""
    search = cfg.search_space
    result: dict[str, list[str | float | int]] = {
        "learning_rate": [float(value) for value in search.learning_rate],
        "global_batch_size": [int(value) for value in search.global_batch_size],
    }
    if scale is not None:
        result["model_size"] = [str(scale.model_size)]
        result["compute_flops"] = [float(scale.compute_flops)]
    return result


def local_batch_size_candidates(
    configured_local_batch_size: int, global_batch_size: int, world_size: int
) -> list[int]:
    """Return compatible local batch sizes, repeatedly halved down to one."""
    if configured_local_batch_size <= 0 or global_batch_size <= 0 or world_size <= 0:
        raise ValueError("batch sizes and world_size must be positive")
    local_batch_size = min(configured_local_batch_size, global_batch_size // world_size)
    candidates: list[int] = []
    while local_batch_size >= 1:
        if global_batch_size % (local_batch_size * world_size) == 0:
            candidates.append(local_batch_size)
        if local_batch_size == 1:
            break
        local_batch_size //= 2
    if not candidates:
        raise ValueError("global batch size must be divisible by world size")
    return candidates


class TrainingObjective:
    def __init__(self, cfg: DictConfig, scale: DictConfig, output_dir: Path):
        self.cfg = cfg
        self.scale = scale
        self.output_dir = output_dir

    def __call__(self, trial: optuna.Trial) -> float:
        search = grid_search_space(self.cfg)
        trial.suggest_categorical("model_size", [str(self.scale.model_size)])
        trial.suggest_categorical("compute_flops", [float(self.scale.compute_flops)])
        lr = trial.suggest_float(
            "learning_rate",
            min(search["learning_rate"]),
            max(search["learning_rate"]),
            log=True,
        )
        weight_decay = float(self.cfg.weight_decay)
        global_batch_size = _suggest_global_batch_size(
            trial, [int(value) for value in search["global_batch_size"]]
        )
        world_size = int(self.cfg.nproc_per_node)
        if global_batch_size % world_size:
            raise optuna.TrialPruned(
                f"global batch size {global_batch_size} is not divisible by world size {world_size}"
            )
        trial_dir = self.output_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        architecture = OmegaConf.to_container(self.scale.architecture, resolve=True)
        candidates = local_batch_size_candidates(
            int(self.cfg.local_batch_size), global_batch_size, world_size
        )
        last_error: subprocess.CalledProcessError | None = None
        for attempt, local_batch_size in enumerate(candidates):
            attempt_dir = trial_dir / f"attempt_{attempt:02d}_local_batch_{local_batch_size}"
            checkpoint_dir = attempt_dir / "checkpoints"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            overrides = [
                f"sweep.num_tokens={int(self.scale.num_tokens)}",
                f"step_scheduler.global_batch_size={global_batch_size}",
                f"step_scheduler.local_batch_size={local_batch_size}",
                f"optimizer.lr={lr}",
                f"sweep_optimizer.weight_decay={weight_decay}",
                f"checkpoint.checkpoint_dir={checkpoint_dir}",
            ]
            for key, value in architecture.items():
                overrides.append(f"model.config.{key}={value}")

            config_dir = Path(__file__).resolve().parents[1] / "config"
            with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
                train_cfg = compose(config_name=str(self.cfg.training_config), overrides=overrides)
            generated_config = attempt_dir / "automodel.yaml"
            write_automodel_config(train_cfg, generated_config)
            command = automodel_command(generated_config, world_size)
            try:
                with (attempt_dir / "training.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as error:
                last_error = error
                trial.set_user_attr(f"failed_local_batch_size_{attempt}", local_batch_size)
                continue

            trial.set_user_attr("effective_local_batch_size", local_batch_size)
            trial.set_user_attr("batch_size_attempts", attempt + 1)
            return read_validation_loss(checkpoint_dir / "validation.jsonl")

        trial.set_user_attr("batch_size_attempts", len(candidates))
        assert last_error is not None
        raise last_error


def run(config_path: Path) -> Path:
    cfg = OmegaConf.load(config_path)
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for scale in scaling_points(cfg):
        name = _study_name(scale)
        scale_dir = output_dir / name
        scale_dir.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{(scale_dir / 'study.db').resolve()}"
        study = optuna.create_study(
            study_name=name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.GridSampler(
                grid_search_space(cfg, scale), seed=int(cfg.seed)
            ),
        )
        study.optimize(
            TrainingObjective(cfg, scale, scale_dir),
            n_trials=int(cfg.n_trials),
            catch=(subprocess.CalledProcessError, FileNotFoundError, ValueError),
        )
    return analyze(config_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/scaling.yaml"))
    args = parser.parse_args()
    output_dir = run(args.config)
    print(f"Scaling-law results written to {output_dir}")


if __name__ == "__main__":
    main()
