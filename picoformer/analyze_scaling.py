"""Analyze completed Optuna scaling studies without running new training trials."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import optuna
from omegaconf import DictConfig, OmegaConf


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


def tokens_for_compute(compute_flops: float, num_parameters: int) -> int:
    """Convert a FLOP target to the nearest token count using C = 6 N D."""
    if not math.isfinite(compute_flops) or compute_flops <= 0 or num_parameters <= 0:
        raise ValueError("compute_flops and num_parameters must be positive")
    return max(1, round(compute_flops / (6.0 * num_parameters)))


def scaling_points(cfg: DictConfig) -> list[DictConfig]:
    """Expand configured model sizes and FLOP targets into study points."""
    points: list[DictConfig] = []
    for model_size in cfg.model_sizes:
        for compute_flops in cfg.compute_flops:
            points.append(OmegaConf.create({
                "name": f"{model_size.name}_{float(compute_flops):.6g}flops",
                "model_size": str(model_size.name),
                "num_parameters": int(model_size.num_parameters),
                "compute_flops": float(compute_flops),
                "num_tokens": tokens_for_compute(
                    float(compute_flops), int(model_size.num_parameters)
                ),
                "architecture": OmegaConf.to_container(model_size.architecture, resolve=True),
            }))
    return points


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
    # R squared is undefined for a constant target. Treat an exact constant-law
    # fit as perfect; polyfit can leave a tiny residual from roundoff when the
    # compute values are large.
    if total == 0.0:
        r_squared = 1.0 if np.allclose(log_y, predicted) else 0.0
    else:
        r_squared = 1.0 - residual / total
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
    for name in ("learning_rate", "global_batch_size"):
        values = np.asarray([float(trial.params[name]) for trial in selected], dtype=float)
        result[name] = float(np.exp(np.median(np.log(values))))
    return result


def study_name(scale: DictConfig) -> str:
    return f"{scale.model_size}_{int(scale.num_parameters)}p_{float(scale.compute_flops):.6g}flops"


def fit_power_laws_by_model_size(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, PowerLaw]]:
    """Fit hyperparameter scaling laws independently for each model size."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["num_parameters"]), []).append(row)

    laws_by_model_size: dict[int, dict[str, PowerLaw]] = {}
    for num_parameters, model_rows in grouped.items():
        if len(model_rows) < 2:
            raise ValueError(
                "at least two successful scale points are required for model size "
                f"{num_parameters}"
            )
        laws_by_model_size[num_parameters] = {
            name: fit_power_law(
                [float(row["compute_flops"]) for row in model_rows],
                [float(row[f"near_optimal_{name}"]) for row in model_rows],
            )
            for name in ("learning_rate", "global_batch_size")
        }
    return laws_by_model_size


def plot_power_laws(
    rows: list[dict[str, Any]],
    laws_by_model_size: dict[int, dict[str, PowerLaw]],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = {
        "learning_rate": "Learning rate",
        "global_batch_size": "Global batch size",
    }
    for axis, (name, label) in zip(axes[:2], labels.items(), strict=True):
        for num_parameters, laws in laws_by_model_size.items():
            model_rows = [row for row in rows if int(row["num_parameters"]) == num_parameters]
            compute = np.asarray([float(row["compute_flops"]) for row in model_rows])
            line_x = np.geomspace(compute.min(), compute.max(), 200)
            law = laws[name]
            points = axis.loglog(
                compute,
                [row[f"near_optimal_{name}"] for row in model_rows],
                "o",
                label=f"{num_parameters:,} parameters",
            )[0]
            axis.loglog(line_x, [law.predict(x) for x in line_x], "-", color=points.get_color())
        axis.set(xlabel="Compute C (FLOPs)", ylabel=label)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)

    loss_axis = axes[2]
    for num_parameters in laws_by_model_size:
        model_rows = [row for row in rows if int(row["num_parameters"]) == num_parameters]
        compute = np.asarray([float(row["compute_flops"]) for row in model_rows])
        losses = np.asarray([float(row["best_validation_loss"]) for row in model_rows])
        law = fit_power_law(compute, losses)
        line_x = np.geomspace(compute.min(), compute.max(), 200)
        points = loss_axis.loglog(
            compute,
            losses,
            "o",
            label=f"{num_parameters:,} parameters",
        )[0]
        loss_axis.loglog(
            line_x,
            [law.predict(x) for x in line_x],
            "-",
            color=points.get_color(),
        )
    loss_axis.set(xlabel="Compute C (FLOPs)", ylabel="Best validation loss")
    loss_axis.grid(True, which="both", alpha=0.25)
    loss_axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_results(rows: list[dict[str, Any]], output_dir: Path, margin: float) -> None:
    if not rows:
        raise ValueError("no completed scaling studies were found")
    with (output_dir / "best_settings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    laws_by_model_size = fit_power_laws_by_model_size(rows)
    best_overall = min(rows, key=lambda row: float(row["best_validation_loss"]))
    payload = {
        "compute_definition": "C = 6 * num_parameters * num_tokens",
        "near_optimal_relative_margin": margin,
        "lowest_loss_setting": best_overall,
        "power_laws_by_model_size": {
            str(num_parameters): {
                name: {
                    "coefficient": law.coefficient,
                    "exponent": law.exponent,
                    "r_squared_log_space": law.r_squared,
                    "equation": law.equation(name),
                }
                for name, law in laws.items()
            }
            for num_parameters, laws in laws_by_model_size.items()
        },
    }
    (output_dir / "power_laws.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / "best_overall.json").write_text(json.dumps(best_overall, indent=2) + "\n")
    plot_power_laws(rows, laws_by_model_size, output_dir / "power_laws.png")


def analyze(config_path: Path) -> Path:
    """Reload configured studies and regenerate all summary artifacts."""
    cfg = OmegaConf.load(config_path)
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    margin = float(cfg.near_optimal_relative_margin)
    rows: list[dict[str, Any]] = []
    for scale in scaling_points(cfg):
        name = study_name(scale)
        database = (output_dir / name / "study.db").resolve()
        if not database.is_file():
            raise FileNotFoundError(f"scaling study database does not exist: {database}")
        study = optuna.load_study(study_name=name, storage=f"sqlite:///{database}")
        selected = near_optimal_trials(study.trials, margin)
        representative = representative_hparams(selected)
        best_params = study.best_trial.params
        rows.append(
            {
                "name": str(scale.name),
                "num_parameters": int(scale.num_parameters),
                "num_tokens": int(scale.num_tokens),
                "compute_flops": float(scale.compute_flops),
                "actual_compute_flops": compute_budget(
                    int(scale.num_parameters), int(scale.num_tokens)
                ),
                "best_validation_loss": float(study.best_value),
                "near_optimal_trials": len(selected),
                "best_learning_rate": float(best_params["learning_rate"]),
                "fixed_weight_decay": float(cfg.weight_decay),
                "best_global_batch_size": int(best_params["global_batch_size"]),
                **{f"near_optimal_{key}": value for key, value in representative.items()},
            }
        )
    write_results(rows, output_dir, margin)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/scaling.yaml"))
    args = parser.parse_args()
    output_dir = analyze(args.config)
    print(f"Scaling-law analysis written to {output_dir}")


if __name__ == "__main__":
    main()
