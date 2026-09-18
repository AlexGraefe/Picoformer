"""Collect lm-eval metrics and plot every task metric over training steps."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


@dataclass(frozen=True)
class Metric:
    run: str
    suite: str
    task: str
    metric: str
    step: int
    value: float
    stderr: float | None
    checkpoint: str
    result_file: str


def _stderr_key(metric: str) -> str:
    name, separator, filter_name = metric.partition(",")
    return f"{name}_stderr{separator}{filter_name}"


def _is_metric(name: str, value: Any) -> bool:
    stem = name.partition(",")[0]
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and not stem.endswith("_stderr")
        and stem not in {"N", "n_samples"}
    )


def _latest_result(directory: Path) -> Path | None:
    files = list(directory.rglob("results_*.json"))
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def load_metrics(root: Path, *, phase: str = "full") -> list[Metric]:
    """Load numeric, non-stderr task metrics from completed evaluations."""
    rows: list[Metric] = []
    for manifest_path in root.rglob("evaluation.json"):
        try:
            with manifest_path.open(encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("status") != "complete" or manifest.get("phase") != phase:
            continue
        result_path = _latest_result(manifest_path.parent)
        if result_path is None:
            continue
        try:
            with result_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue
        for task, values in payload.get("results", {}).items():
            if not isinstance(values, dict):
                continue
            for metric, value in values.items():
                if not _is_metric(metric, value):
                    continue
                stderr_value = values.get(_stderr_key(metric))
                stderr = (
                    float(stderr_value)
                    if isinstance(stderr_value, (int, float))
                    and not isinstance(stderr_value, bool)
                    and math.isfinite(float(stderr_value))
                    else None
                )
                rows.append(
                    Metric(
                        run=str(manifest["run_name"]),
                        suite=str(manifest["suite"]),
                        task=str(task),
                        metric=str(metric),
                        step=int(manifest["training_step"]),
                        value=float(value),
                        stderr=stderr,
                        checkpoint=str(manifest["checkpoint"]),
                        result_file=str(result_path),
                    )
                )
    return sorted(rows, key=lambda row: (row.suite, row.task, row.metric, row.run, row.step))


def write_metrics_csv(rows: Iterable[Metric], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "run",
                "suite",
                "task",
                "metric",
                "step",
                "value",
                "stderr",
                "checkpoint",
                "result_file",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    return output


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.") or "metric"


def plot_evaluations(root: Path, *, phase: str = "full") -> list[Path]:
    """Write metrics.csv plus one PNG for every independent task metric."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = root.expanduser().resolve()
    rows = load_metrics(root, phase=phase)
    if not rows:
        raise ValueError(f"no completed {phase} evaluation results found below {root}")
    outputs = [write_metrics_csv(rows, root / f"metrics-{phase}.csv")]
    plot_dir = root / f"plots-{phase}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({(row.suite, row.task, row.metric) for row in rows})
    for suite, task, metric in keys:
        selected = [
            row
            for row in rows
            if (row.suite, row.task, row.metric) == (suite, task, metric)
        ]
        fig, axis = plt.subplots(figsize=(8, 5))
        for run in sorted({row.run for row in selected}):
            run_rows = sorted((row for row in selected if row.run == run), key=lambda row: row.step)
            x = [row.step for row in run_rows]
            y = [row.value for row in run_rows]
            if any(row.stderr is not None for row in run_rows):
                errors = [row.stderr or 0.0 for row in run_rows]
                axis.errorbar(x, y, yerr=errors, marker="o", capsize=3, label=run)
            else:
                axis.plot(x, y, marker="o", label=run)
        axis.set_xlabel("Completed optimizer steps")
        axis.set_ylabel(metric)
        axis.set_title(f"{suite}: {task} — {metric}")
        axis.grid(True, alpha=0.25)
        if len({row.run for row in selected}) > 1:
            axis.legend(fontsize=8)
        fig.tight_layout()
        output = plot_dir / _slug(f"{suite}--{task}--{metric}")
        output = output.with_suffix(".png")
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="evaluation results root")
    parser.add_argument("--phase", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    for output in plot_evaluations(args.results, phase=args.phase):
        print(output)


if __name__ == "__main__":
    main()
