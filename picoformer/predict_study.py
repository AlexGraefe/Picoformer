"""Extrapolate one scaling study's validation losses against training tokens."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from picoformer.plot_study import load_completed_trials


MIN_SELECTION_R_SQUARED = 0.95


def select_best_curve(curves: list[dict]) -> dict | None:
    """Select the lowest target loss among fits meeting the R² threshold."""
    eligible = [curve for curve in curves
                if curve["fit"]["r_squared"] >= MIN_SELECTION_R_SQUARED]
    return min(eligible, key=lambda curve: (curve["predicted_loss"], curve["trial"]), default=None)


@dataclass(frozen=True)
class LossFit:
    floor: float
    amplitude: float
    exponent: float
    reference_tokens: float
    rmse: float
    r_squared: float
    exponent_at_bound: bool

    def predict(self, tokens: np.ndarray) -> np.ndarray:
        return self.floor + self.amplitude * (tokens / self.reference_tokens) ** -self.exponent


def fit_loss(tokens: np.ndarray, losses: np.ndarray) -> LossFit:
    """Fit L(D) = floor + amplitude * (D / reference_tokens)^(-exponent).

    Minimize squared error in loss space, with nonnegative floor/amplitude and
    exponent in [0.01, 3]. Profile the two linear coefficients over the exponent
    using a grid followed by local refinement; this needs only NumPy.
    """
    tokens, losses = np.asarray(tokens, dtype=float), np.asarray(losses, dtype=float)
    if tokens.ndim != 1 or losses.shape != tokens.shape or len(tokens) < 4:
        raise ValueError("need at least four distinct pre-decay validation points")
    if (not np.isfinite(tokens).all() or not np.isfinite(losses).all()
            or (tokens <= 0).any() or (losses <= 0).any()
            or len(np.unique(tokens)) != len(tokens)):
        raise ValueError("tokens must be distinct and positive; losses must be finite and positive")
    if np.dot(np.log(tokens) - np.log(tokens).mean(), losses - losses.mean()) >= 0:
        raise ValueError("pre-decay validation loss has no decreasing trend")
    reference = float(tokens.min())

    def solve(exponent: float) -> tuple[float, float, float]:
        x = (tokens / reference) ** -exponent
        amplitude = float(np.dot(x - x.mean(), losses - losses.mean()) / np.sum((x - x.mean()) ** 2))
        floor = float(losses.mean() - amplitude * x.mean())
        candidates = [(float(losses.mean()), 0.0), (0.0, float(np.dot(x, losses) / np.dot(x, x)))]
        if floor >= 0 and amplitude >= 0:
            candidates.append((floor, amplitude))
        return min((float(np.sum((losses - f - a * x) ** 2)), f, a) for f, a in candidates)

    lower, upper = 0.01, 3.0
    for _ in range(5):
        grid = np.linspace(lower, upper, 161)
        index = min(range(len(grid)), key=lambda i: solve(float(grid[i]))[0])
        exponent = float(grid[index])
        lower, upper = float(grid[max(0, index - 1)]), float(grid[min(len(grid) - 1, index + 1)])
    error, floor, amplitude = solve(exponent)
    # In-sample R² in loss space, using only the points passed to the fit.
    total_variation = float(np.sum((losses - losses.mean()) ** 2))
    r_squared = 1.0 - error / total_variation
    return LossFit(floor, amplitude, exponent, reference, math.sqrt(error / len(tokens)),
                   r_squared, exponent <= 0.01001 or exponent >= 2.99999)


def read_curve(config_path: Path, *, fit_start_fraction: float = 0.0) -> dict:
    """Read zero-based AutoModel validation steps and the actual WSD schedule."""
    if not 0 <= fit_start_fraction < 1:
        raise ValueError("fit_start_fraction must be in [0, 1)")
    cfg = OmegaConf.load(config_path)
    schedule = cfg.lr_scheduler
    if str(schedule.lr_decay_style).upper() != "WSD":
        raise ValueError("expected a WSD learning-rate schedule")
    batch = int(cfg.step_scheduler.global_batch_size)
    sequence_length = int(cfg.dataset.seq_len)
    max_steps = int(cfg.step_scheduler.max_steps)
    warmup = int(schedule.lr_warmup_steps)
    decay_end = int(schedule.lr_decay_steps)
    decay_steps = int(schedule.wsd_decay_steps)
    lr = float(cfg.optimizer.lr)
    if (min(batch, sequence_length, max_steps, decay_end) <= 0
            or not 0 <= decay_steps <= decay_end or not 0 <= warmup <= decay_end
            or not math.isfinite(lr) or lr <= 0):
        raise ValueError("invalid token count or learning-rate schedule in config")
    decay_start = decay_end - decay_steps
    # Prefer the sibling file so copied study directories remain usable.
    metrics = config_path.parent / "checkpoints" / "validation.jsonl"
    if not metrics.is_file():
        metrics = Path(str(cfg.checkpoint.checkpoint_dir)).expanduser() / "validation.jsonl"
    rows = {}
    for line_number, line in enumerate(metrics.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            step = int(row["step"])
            loss = float(row["val_loss"])
            logged_lr = float(row.get("lr", lr))
            if step < 0 or step != row["step"]:
                raise ValueError("step must be a nonnegative integer")
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"invalid validation record at {metrics}:{line_number}: {error}") from error
        # On resume, the last record at a repeated step supersedes older ones.
        rows[step] = (loss, logged_lr)
    points = []
    for step, (loss, logged_lr) in sorted(rows.items()):
        if not math.isfinite(loss) or loss <= 0:
            continue
        completed = step + 1
        reason = "fit"
        if completed > max_steps:
            reason = "outside_run"
        elif completed <= warmup:
            reason = "warmup"
        elif completed > decay_start:
            reason = "final_decay"
        elif completed < fit_start_fraction * max_steps:
            reason = "before_fit_start"
        elif not math.isclose(logged_lr, lr, rel_tol=1e-5, abs_tol=0):
            reason = "non_plateau_lr"
        points.append({"step": step, "tokens": completed * batch * sequence_length,
                       "loss": loss, "used_for_fit": reason == "fit", "reason": reason})
    return {"config": str(config_path), "metrics": str(metrics), "batch_size": batch,
            "learning_rate": lr, "planned_tokens": max_steps * batch * sequence_length,
            "decay_start_tokens": decay_start * batch * sequence_length, "points": points}


def predict_study(database: Path, output_dir: Path, *, target_tokens: float | None = None,
                  trial_numbers: list[int] | None = None, fit_start_fraction: float = 0.0) -> Path:
    """Write a plot, fit metadata, and forecast CSV for one model/FLOP study."""
    database = database.expanduser().resolve()
    if database.is_dir():
        database = database / "study.db"
    if not 0 <= fit_start_fraction < 1:
        raise ValueError("fit_start_fraction must be in [0, 1)")
    if target_tokens is not None and (not math.isfinite(target_tokens) or target_tokens <= 0):
        raise ValueError("target_tokens must be finite and positive")
    name, trials = load_completed_trials(database)
    budgets = {trial.params.get("compute_flops") for trial in trials}
    models = {trial.params.get("model_size") for trial in trials}
    if len(budgets) != 1 or len(models) != 1:
        raise ValueError("select a study containing one model size and one FLOP budget")
    if trial_numbers is not None:
        missing = set(trial_numbers) - {trial.number for trial in trials}
        if missing:
            raise ValueError(f"requested trials are not completed: {sorted(missing)}")
        trials = [trial for trial in trials if trial.number in trial_numbers]
    curves, skipped = [], []
    for trial in trials:
        try:
            directory = database.parent / f"trial_{trial.number:04d}"
            attempts = sorted(directory.glob("attempt_*/automodel.yaml"))
            if "batch_size_attempts" in trial.user_attrs:
                prefix = f"attempt_{int(trial.user_attrs['batch_size_attempts']) - 1:02d}_"
                attempts = [path for path in attempts if path.parent.name.startswith(prefix)]
            if not attempts:
                raise ValueError("saved automodel.yaml not found for the completed attempt")
            curve = read_curve(attempts[-1], fit_start_fraction=fit_start_fraction)
            fit_points = [point for point in curve["points"] if point["used_for_fit"]]
            fit = fit_loss(np.array([p["tokens"] for p in fit_points]),
                           np.array([p["loss"] for p in fit_points]))
            curves.append({"trial": trial.number, "params": dict(trial.params),
                           **curve, "fit": asdict(fit)})
        except (OSError, ValueError) as error:
            skipped.append({"trial": trial.number, "reason": str(error)})
    if not curves:
        reasons = "; ".join(f"#{row['trial']}: {row['reason']}" for row in skipped)
        raise ValueError(f"no usable curves; {reasons}. At least four plateau evaluations are required.")
    observed_end = max(max(p["tokens"] for p in curve["points"]) for curve in curves)
    target = target_tokens if target_tokens is not None else 2.0 * max(c["planned_tokens"] for c in curves)
    if target <= observed_end:
        raise ValueError(f"target_tokens must exceed the last observation ({observed_end:,})")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axis = plt.subplots(figsize=(12, 7))
    predictions = []
    for index, curve in enumerate(curves):
        fit = LossFit(**curve["fit"])
        points = curve["points"]
        included = [p for p in points if p["used_for_fit"]]
        excluded = [p for p in points if not p["used_for_fit"]]
        color = plt.get_cmap("tab20")(index % 20)
        label = (f"#{curve['trial']}  batch={curve['batch_size']}  "
                 f"lr={curve['learning_rate']:.3g}  R²={fit.r_squared:.4f}")
        axis.plot([p["tokens"] for p in included], [p["loss"] for p in included],
                  "o-", color=color, markersize=4, linewidth=1, label=label)
        axis.scatter([p["tokens"] for p in excluded], [p["loss"] for p in excluded],
                     marker="x", color=color, alpha=0.35)
        last_fit = included[-1]["tokens"]
        fitted_x = np.geomspace(included[0]["tokens"], last_fit, 100)
        future_x = np.geomspace(last_fit, target, 200)
        axis.plot(fitted_x, fit.predict(fitted_x), color=color, alpha=0.5)
        axis.plot(future_x, fit.predict(future_x), "--", color=color, linewidth=1.5)
        curve["predicted_loss"] = float(fit.predict(np.asarray(target)))
        for tokens, loss in zip(future_x, fit.predict(future_x), strict=True):
            predictions.append({"trial": curve["trial"], "tokens": float(tokens), "predicted_loss": float(loss)})
    axis.set(xlabel="Training tokens", ylabel="Validation loss",
             title=f"{name}\nPlateau-learning-rate extrapolation")
    axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    handles.extend([Line2D([], [], color="gray", linestyle="--"),
                    Line2D([], [], color="gray", marker="x", linestyle="None", alpha=0.35)])
    labels.extend(["Extrapolation from last fit point", "Excluded from fit"])
    axis.legend(handles, labels, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    output = output_dir / "validation_forecast.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    best_curve = select_best_curve(curves)
    best_at_target = None if best_curve is None else {
        "trial": best_curve["trial"], "target_tokens": target,
        "predicted_loss": best_curve["predicted_loss"],
        "r_squared": best_curve["fit"]["r_squared"],
        "params": best_curve["params"], "config": best_curve["config"]}
    metadata = {"study": name, "compute_flops": next(iter(budgets)), "target_tokens": target,
                "best_at_target": best_at_target,
                "min_selection_r_squared": MIN_SELECTION_R_SQUARED,
                "formula": "floor + amplitude * (tokens / reference_tokens) ** (-exponent)",
                "assumption": "Continued training at the plateau learning rate; no final decay.",
                "fit_start_fraction": fit_start_fraction, "curves": curves, "skipped": skipped}
    (output_dir / "validation_forecast.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    with (output_dir / "validation_forecast.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["trial", "tokens", "predicted_loss"])
        writer.writeheader()
        writer.writerows(predictions)
    for row in skipped:
        print(f"Skipped trial #{row['trial']}: {row['reason']}")
    for curve in curves:
        print(f"Trial #{curve['trial']}: fit R² = {curve['fit']['r_squared']:.6f}")
        if curve["fit"]["r_squared"] < MIN_SELECTION_R_SQUARED:
            print(f"  Excluded from parameter selection: R² < {MIN_SELECTION_R_SQUARED}.")
        if curve["fit"]["exponent_at_bound"]:
            print(f"Trial #{curve['trial']}: exponent reached its fit bound; inspect the curve.")
    if best_at_target is None:
        print(f"No best parameter set at {target:,.0f} tokens: no fits have R² >= {MIN_SELECTION_R_SQUARED}.")
        return output
    print(f"Best predicted parameter set at {target:,.0f} tokens (among fits with R² >= {MIN_SELECTION_R_SQUARED}):")
    print(f"  Trial #{best_at_target['trial']}: predicted validation loss = {best_at_target['predicted_loss']:.6f}")
    for parameter, value in sorted(best_at_target["params"].items()):
        print(f"  {parameter} = {value}")
    print(f"  Full training config: {best_at_target['config']}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="one study.db file or its study directory")
    parser.add_argument("--output-dir", type=Path, default=Path("results/forecasts"))
    parser.add_argument("--tokens", type=float, help="forecast horizon (default: twice the planned training tokens)")
    parser.add_argument("--trials", type=int, nargs="+", help="only these completed trial numbers")
    parser.add_argument("--fit-start-fraction", type=float, default=0.0,
                        help="additionally exclude this initial fraction of planned training (e.g. 0.2)")
    args = parser.parse_args()
    try:
        output = predict_study(args.database, args.output_dir, target_tokens=args.tokens,
                               trial_numbers=args.trials, fit_start_fraction=args.fit_start_fraction)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Forecast written to {output} (with JSON fits and CSV predictions)")


if __name__ == "__main__":
    main()
