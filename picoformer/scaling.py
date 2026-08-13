"""Optuna-based hyperparameter scaling-law experiments.

Each architecture/token-budget pair gets an independent Optuna study.  This is
important: a single study over compute budgets would merely learn that larger
budgets tend to have lower loss rather than finding good hyperparameters at
each scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import optuna
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from picoformer.sweep import automodel_command, write_automodel_config


@dataclass(frozen=True)
class PowerLaw:
    """A fitted relationship ``y = coefficient * compute_flops ** exponent``."""

    coefficient: float
    exponent: float
    r_squared: float

    def predict(self, compute_flops: float) -> float:
        return self.coefficient * compute_flops**self.exponent

    def equation(self, variable: str = "y") -> str:
        return f"{variable} = {self.coefficient:.6g} * C^{self.exponent:.6g}"


def compute_budget(num_parameters: int, num_tokens: int) -> float:
    """Return the dense-transformer training estimate C = 6 N D FLOPs."""
    if num_parameters <= 0 or num_tokens <= 0:
        raise ValueError("num_parameters and num_tokens must be positive")
    return 6.0 * num_parameters * num_tokens


def fit_power_law(compute_flops: Iterable[float], values: Iterable[float]) -> PowerLaw:
    """Fit a power law in log space and report log-space R squared."""
    x = np.asarray(list(compute_flops), dtype=float)
    y = np.asarray(list(values), dtype=float)
    if len(x) < 2 or len(x) != len(y):
        raise ValueError("at least two paired observations are required")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("power-law observations must be finite and positive")
    log_x, log_y = np.log(x), np.log(y)
    exponent, intercept = np.polyfit(log_x, log_y, 1)
    predicted = exponent * log_x + intercept
    residual = float(np.sum((log_y - predicted) ** 2))
    total = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r_squared = 1.0 if total == 0.0 and residual == 0.0 else 1.0 - residual / total
    return PowerLaw(float(np.exp(intercept)), float(exponent), r_squared)


def near_optimal_trials(
    trials: Iterable[optuna.trial.FrozenTrial], relative_margin: float = 0.0025
) -> list[optuna.trial.FrozenTrial]:
    """Select completed trials within ``relative_margin`` of minimum loss."""
    if relative_margin < 0:
        raise ValueError("relative_margin must be non-negative")
    completed = [
        trial
        for trial in trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(trial.value)
    ]
    if not completed:
        return []
    best = min(float(trial.value) for trial in completed)
    return [trial for trial in completed if float(trial.value) <= best * (1.0 + relative_margin)]


def representative_hparams(trials: Iterable[optuna.trial.FrozenTrial]) -> dict[str, float]:
    """Return geometric-median hyperparameters from a broad optimum region."""
    selected = list(trials)
    if not selected:
        raise ValueError("no near-optimal trials")
    result: dict[str, float] = {}
    for name in ("learning_rate", "weight_decay", "global_batch_size"):
        values = np.asarray([float(trial.params[name]) for trial in selected], dtype=float)
        result[name] = float(np.exp(np.median(np.log(values))))
    return result


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
        search = self.cfg.search_space
        lr = trial.suggest_float(
            "learning_rate", float(search.learning_rate.low), float(search.learning_rate.high), log=True
        )
        weight_decay = trial.suggest_float(
            "weight_decay", float(search.weight_decay.low), float(search.weight_decay.high), log=True
        )
        global_batch_size = _suggest_global_batch_size(
            trial, [int(value) for value in search.global_batch_size]
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


def _study_name(scale: DictConfig) -> str:
    return f"{scale.name}_{int(scale.num_parameters)}p_{int(scale.num_tokens)}t"


def _write_results(rows: list[dict[str, Any]], output_dir: Path, margin: float) -> None:
    fieldnames = list(rows[0])
    with (output_dir / "best_settings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    laws = {
        name: fit_power_law(
            [float(row["compute_flops"]) for row in rows],
            [float(row[f"near_optimal_{name}"]) for row in rows],
        )
        for name in ("learning_rate", "weight_decay", "global_batch_size")
    }
    best_overall = min(rows, key=lambda row: float(row["best_validation_loss"]))
    payload = {
        "compute_definition": "C = 6 * num_parameters * num_tokens",
        "near_optimal_relative_margin": margin,
        "lowest_loss_setting": best_overall,
        "power_laws": {
            name: {
                "coefficient": law.coefficient,
                "exponent": law.exponent,
                "r_squared_log_space": law.r_squared,
                "equation": law.equation(name),
            }
            for name, law in laws.items()
        },
    }
    (output_dir / "power_laws.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / "best_overall.json").write_text(json.dumps(best_overall, indent=2) + "\n")
    _plot_power_laws(rows, laws, output_dir / "power_laws.png")


def _plot_power_laws(rows: list[dict[str, Any]], laws: dict[str, PowerLaw], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compute = np.asarray([float(row["compute_flops"]) for row in rows])
    line_x = np.geomspace(compute.min(), compute.max(), 200)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = {
        "learning_rate": "Learning rate",
        "weight_decay": "Weight decay",
        "global_batch_size": "Global batch size",
    }
    for axis, (name, label) in zip(axes, labels.items(), strict=True):
        law = laws[name]
        axis.loglog(
            compute,
            [row[f"near_optimal_{name}"] for row in rows],
            "o",
            label="near-optimal geometric median",
        )
        axis.loglog(line_x, [law.predict(x) for x in line_x], "-", label=law.equation(label))
        axis.set(xlabel="Compute C (FLOPs)", ylabel=label)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(config_path: Path) -> Path:
    cfg = OmegaConf.load(config_path)
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    margin = float(cfg.near_optimal_relative_margin)
    for scale in cfg.scales:
        name = _study_name(scale)
        scale_dir = output_dir / name
        scale_dir.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{(scale_dir / 'study.db').resolve()}"
        study = optuna.create_study(
            study_name=name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=int(cfg.seed)),
        )
        study.optimize(
            TrainingObjective(cfg, scale, scale_dir),
            n_trials=int(cfg.n_trials),
            catch=(subprocess.CalledProcessError, FileNotFoundError, ValueError),
        )
        selected = near_optimal_trials(study.trials, margin)
        representative = representative_hparams(selected)
        best_params = study.best_trial.params
        rows.append(
            {
                "name": str(scale.name),
                "num_parameters": int(scale.num_parameters),
                "num_tokens": int(scale.num_tokens),
                "compute_flops": compute_budget(int(scale.num_parameters), int(scale.num_tokens)),
                "best_validation_loss": float(study.best_value),
                "near_optimal_trials": len(selected),
                "best_learning_rate": float(best_params["learning_rate"]),
                "best_weight_decay": float(best_params["weight_decay"]),
                "best_global_batch_size": int(best_params["global_batch_size"]),
                **{f"near_optimal_{key}": value for key, value in representative.items()},
            }
        )
    if len(rows) < 2:
        raise ValueError("at least two successful scale points are required to fit power laws")
    _write_results(rows, output_dir, margin)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/scaling.yaml"))
    args = parser.parse_args()
    output_dir = run(args.config)
    print(f"Scaling-law results written to {output_dir}")


if __name__ == "__main__":
    main()
