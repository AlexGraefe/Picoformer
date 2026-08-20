"""Visualize the completed trials in one scaling-study SQLite database."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import optuna


PARAMETERS = ("global_batch_size", "learning_rate")


def load_completed_trials(database: Path) -> tuple[str, list[optuna.trial.FrozenTrial]]:
    """Load finite, completed trials from the sole study in ``database``."""
    database = database.expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"Optuna database not found: {database}")

    storage = f"sqlite:///{database}"
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    if not summaries:
        raise ValueError(f"no Optuna studies found in {database}")
    if len(summaries) != 1:
        names = ", ".join(summary.study_name for summary in summaries)
        raise ValueError(f"expected one study in {database}, found: {names}")

    study_name = summaries[0].study_name
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(float(trial.value))
        and all(name in trial.params for name in PARAMETERS)
    ]
    if not trials:
        raise ValueError(f"no completed trials with all required parameters in {database}")
    return study_name, trials


def _annotation(trial: optuna.trial.FrozenTrial) -> str:
    return (
        f"#{trial.number}  batch={int(trial.params['global_batch_size'])}\n"
        f"lr={float(trial.params['learning_rate']):.3g}"
    )


def plot_study(database: Path, output: Path) -> Path:
    """Create batch-size and learning-rate versus loss plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    study_name, trials = load_completed_trials(database)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    plots = (
        ("global_batch_size", "Global batch size", False),
        ("learning_rate", "Learning rate", True),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    losses = [float(trial.value) for trial in trials]
    for axis, (parameter, label, log_scale) in zip(axes, plots, strict=True):
        values = [float(trial.params[parameter]) for trial in trials]
        axis.scatter(values, losses, s=42, alpha=0.85)
        for x, y, trial in zip(values, losses, trials, strict=True):
            axis.annotate(
                _annotation(trial),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                alpha=0.9,
            )
        if log_scale:
            axis.set_xscale("log")
        axis.set_xlabel(label)
        axis.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("Validation loss")
    fig.suptitle(f"Hyperparameter trials: {study_name}")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="path to a scaling study.db file")
    parser.add_argument(
        "--output",
        type=Path,
        help="output image (default: <database directory>/study_trials.png)",
    )
    args = parser.parse_args()
    output = args.output or args.database.parent / "study_trials.png"
    print(f"Plot written to {plot_study(args.database, output)}")


if __name__ == "__main__":
    main()
