"""Evaluate every checkpoint in one or more training runs with lm-eval."""

from __future__ import annotations

import argparse
import codecs
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import shutil
import subprocess
import sys
import time
from typing import Iterable, Sequence

from picoformer.consolidate_checkpoint import consolidate_checkpoint


SUITES: dict[str, tuple[str, ...]] = {
    "pretrain": (
        "wikitext",
        "lambada_openai",
        "hellaswag",
        "piqa",
        "arc_easy",
        "winogrande",
        "mutual",
        "coqa",
        "arithmetic",
        "asdiv",
        "gsm8k",
    ),
    "instruct": (
        "ifeval",
        "eq_bench",
        "coqa",
        "mutual_plus",
        "asdiv_cot_llama",
        "gsm8k",
    ),
    "instruct-retention": (
        "hellaswag",
        "piqa",
        "arc_easy",
        "winogrande",
        "arithmetic",
    ),
}

GENERATIVE_TASKS: dict[str, tuple[str, ...]] = {
    "pretrain": ("coqa", "gsm8k"),
    "instruct": ("ifeval", "eq_bench", "coqa", "asdiv_cot_llama", "gsm8k"),
    "instruct-retention": (),
}

_CHECKPOINT_RE = re.compile(r"^epoch_(?P<epoch>\d+)_step_(?P<step>\d+)$")


@dataclass(frozen=True)
class Checkpoint:
    run_name: str
    run_slug: str
    path: Path
    epoch: int
    checkpoint_step: int

    @property
    def training_step(self) -> int:
        """Number of completed optimizer steps (checkpoint names are zero-based)."""
        return self.checkpoint_step + 1


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")
    return value or "run"


def _checkpoint_dirs(root: Path) -> list[Path]:
    match = _CHECKPOINT_RE.fullmatch(root.name)
    if match and (root / "model").is_dir():
        return [root]
    return sorted(
        (
            path
            for path in root.rglob("epoch_*_step_*")
            if path.is_dir()
            and _CHECKPOINT_RE.fullmatch(path.name)
            and (path / "model").is_dir()
        ),
        key=lambda path: str(path),
    )


def discover_checkpoints(roots: Sequence[Path]) -> list[Checkpoint]:
    """Discover checkpoints and group them by their nearest checkpoints directory."""
    discovered: list[Checkpoint] = []
    used_slugs: dict[str, Path] = {}
    for supplied_root in roots:
        root = supplied_root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"run path not found: {root}")
        paths = _checkpoint_dirs(root)
        if not paths:
            raise ValueError(f"no epoch_<N>_step_<N> checkpoints found below {root}")
        containers = sorted({path.parent for path in paths}, key=str)
        for container in containers:
            run_path = container.parent if container.name == "checkpoints" else container
            if root == container or root == run_path or root.name == "checkpoints":
                relative_name = run_path.name
            else:
                try:
                    relative_name = str(run_path.relative_to(root))
                except ValueError:
                    relative_name = run_path.name
            run_name = relative_name.replace(os.sep, "/")
            run_slug = _slug(run_name.replace("/", "--"))
            if run_slug in used_slugs and used_slugs[run_slug] != run_path:
                raise ValueError(
                    f"run names collide after sanitizing: {used_slugs[run_slug]} and {run_path}"
                )
            used_slugs[run_slug] = run_path
            for path in paths:
                if path.parent != container:
                    continue
                match = _CHECKPOINT_RE.fullmatch(path.name)
                assert match is not None
                discovered.append(
                    Checkpoint(
                        run_name=run_name,
                        run_slug=run_slug,
                        path=path,
                        epoch=int(match.group("epoch")),
                        checkpoint_step=int(match.group("step")),
                    )
                )
    return sorted(
        discovered,
        key=lambda checkpoint: (
            checkpoint.run_slug,
            checkpoint.training_step,
            checkpoint.epoch,
        ),
    )


def requested_suites(suite: str) -> tuple[str, ...]:
    # An instruction checkpoint must be measured with both chat formatting and
    # the base-format retention suite described in the README.
    return ("instruct", "instruct-retention") if suite == "instruct" else (suite,)


def harness_command(
    *,
    executable: str,
    model_path: Path,
    tokenizer: str,
    dtype: str,
    tasks: Sequence[str],
    suite: str,
    output_path: Path,
    device: str,
    batch_size: str,
    phase: str,
    smoke_limit: int,
    extra_args: Sequence[str] = (),
) -> list[str]:
    model_args = f"pretrained={model_path},tokenizer={tokenizer},dtype={dtype}"
    command = [
        executable,
        "run",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--device",
        device,
        "--batch_size",
        batch_size,
        "--output_path",
        str(output_path),
        "--log_samples",
        "--gen_kwargs",
        "do_sample=False",
    ]
    if suite == "instruct":
        command.append("--apply_chat_template")
    if phase == "smoke":
        command.extend(("--limit", str(smoke_limit), "--write_out"))
    command.extend(extra_args)
    return command


def _result_files(output_path: Path) -> list[Path]:
    return sorted(output_path.rglob("results_*.json"), key=lambda path: path.stat().st_mtime)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _harness_version(executable: str) -> str:
    try:
        completed = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = (completed.stdout or completed.stderr).strip()
        if completed.returncode == 0 and output:
            return output
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return importlib.metadata.version("lm_eval")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _huggingface_cache_paths() -> tuple[Path, Path]:
    xdg_cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    hf_home = Path(os.environ.get("HF_HOME", xdg_cache_home / "huggingface"))
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))
    datasets_cache = Path(os.environ.get("HF_DATASETS_CACHE", hf_home / "datasets"))
    return hub_cache.expanduser(), datasets_cache.expanduser()


def _run_and_log(
    command: Sequence[str],
    log_path: Path,
    *,
    heartbeat_seconds: float,
) -> int:
    """Run a command while preserving progress-bar carriage-return updates."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(command)}\n")
        log.flush()
        environment = os.environ.copy()
        environment.setdefault("PYTHONUNBUFFERED", "1")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=environment,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        started_at = time.monotonic()
        last_output_at = started_at
        next_heartbeat_at = started_at + heartbeat_seconds
        try:
            while selector.get_map():
                events = selector.select(timeout=min(1.0, heartbeat_seconds))
                if events:
                    for key, _ in events:
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        decoded = decoder.decode(chunk)
                        sys.stdout.write(decoded)
                        sys.stdout.flush()
                        log.write(decoded)
                        log.flush()
                        last_output_at = time.monotonic()
                        next_heartbeat_at = last_output_at + heartbeat_seconds
                    continue

                now = time.monotonic()
                if process.poll() is None and now >= next_heartbeat_at:
                    heartbeat = (
                        "\n[picoformer] lm-eval is still running "
                        f"(elapsed {_format_duration(now - started_at)}, "
                        f"no output for {_format_duration(now - last_output_at)})\n"
                    )
                    sys.stdout.write(heartbeat)
                    sys.stdout.flush()
                    log.write(heartbeat)
                    log.flush()
                    next_heartbeat_at = now + heartbeat_seconds
        finally:
            selector.close()
            process.stdout.close()
        remainder = decoder.decode(b"", final=True)
        if remainder:
            sys.stdout.write(remainder)
            sys.stdout.flush()
            log.write(remainder)
            log.flush()
        return process.wait()


def _select_tasks(suite: str, phase: str) -> tuple[str, ...]:
    return GENERATIVE_TASKS[suite] if phase == "smoke" else SUITES[suite]


def evaluate(
    checkpoints: Iterable[Checkpoint],
    *,
    suites: Sequence[str],
    phase: str,
    output_root: Path,
    tokenizer: str,
    dtype: str,
    device: str,
    batch_size: str,
    executable: str,
    smoke_limit: int,
    num_threads: int,
    heartbeat_seconds: float,
    extra_args: Sequence[str],
    dry_run: bool,
    force: bool,
    fail_fast: bool,
) -> int:
    failures = 0
    version = "dry-run" if dry_run else _harness_version(executable)
    checkpoint_list = list(checkpoints)
    total_evaluations = len(checkpoint_list) * len(suites)
    if not dry_run:
        hub_cache, datasets_cache = _huggingface_cache_paths()
        print(f"Hugging Face download cache: {hub_cache}", flush=True)
        print(f"Prepared benchmark cache: {datasets_cache}", flush=True)
    print(
        f"Found {len(checkpoint_list)} checkpoint(s); "
        f"scheduled up to {total_evaluations} evaluation(s)",
        flush=True,
    )
    evaluation_number = 0
    for checkpoint_number, checkpoint in enumerate(checkpoint_list, start=1):
        predicted_model_path = checkpoint.path / "model" / "consolidated"
        model_path = predicted_model_path
        if not dry_run:
            print(
                f"\nCheckpoint {checkpoint_number}/{len(checkpoint_list)}: "
                f"consolidating {checkpoint.path}",
                flush=True,
            )
            model_path = consolidate_checkpoint(checkpoint.path, num_threads=num_threads)
        for suite in suites:
            evaluation_number += 1
            tasks = _select_tasks(suite, phase)
            if not tasks:
                print(f"Skipping {suite}: it has no generative tasks for smoke inspection")
                continue
            evaluation_dir = (
                output_root
                / checkpoint.run_slug
                / checkpoint.path.name
                / suite
                / phase
            )
            manifest_path = evaluation_dir / "evaluation.json"
            if (
                not force
                and manifest_path.is_file()
                and _result_files(evaluation_dir)
            ):
                try:
                    with manifest_path.open(encoding="utf-8") as stream:
                        if json.load(stream).get("status") == "complete":
                            print(f"Skipping completed evaluation: {evaluation_dir}")
                            continue
                except (json.JSONDecodeError, OSError):
                    pass
            command = harness_command(
                executable=executable,
                model_path=model_path,
                tokenizer=tokenizer,
                dtype=dtype,
                tasks=tasks,
                suite=suite,
                output_path=evaluation_dir,
                device=device,
                batch_size=batch_size,
                phase=phase,
                smoke_limit=smoke_limit,
                extra_args=extra_args,
            )
            if dry_run:
                print(shlex.join(command))
                continue
            manifest = {
                "batch_size": batch_size,
                "checkpoint": str(checkpoint.path),
                "checkpoint_step": checkpoint.checkpoint_step,
                "command": command,
                "device": device,
                "dtype": dtype,
                "epoch": checkpoint.epoch,
                "harness_version": version,
                "phase": phase,
                "run_name": checkpoint.run_name,
                "run_slug": checkpoint.run_slug,
                "status": "running",
                "suite": suite,
                "tasks": list(tasks),
                "tokenizer": tokenizer,
                "training_step": checkpoint.training_step,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(manifest_path, manifest)
            print(
                f"Evaluation {evaluation_number}/{total_evaluations}: "
                f"{checkpoint.run_name} step {checkpoint.training_step} "
                f"({suite}, {phase}); tasks: {', '.join(tasks)}",
                flush=True,
            )
            try:
                return_code = _run_and_log(
                    command,
                    evaluation_dir / "lm-eval.log",
                    heartbeat_seconds=heartbeat_seconds,
                )
            except OSError as error:
                return_code = 127
                manifest["error"] = str(error)
            result_files = _result_files(evaluation_dir)
            manifest.update(
                {
                    "return_code": return_code,
                    "result_files": [str(path) for path in result_files],
                    "status": "complete" if return_code == 0 and result_files else "failed",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            if return_code == 0 and not result_files:
                manifest["error"] = "lm-eval exited successfully but wrote no results_*.json"
            _write_json(manifest_path, manifest)
            if manifest["status"] == "failed":
                failures += 1
                print(f"Evaluation failed; see {evaluation_dir / 'lm-eval.log'}", file=sys.stderr)
                if fail_fast:
                    return 1
    return 1 if failures else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run or checkpoints directories")
    parser.add_argument(
        "--suite", choices=("pretrain", "instruct", "instruct-retention"), default="pretrain"
    )
    parser.add_argument("--phase", choices=("smoke", "full"), default="full")
    parser.add_argument("--output", type=Path, default=Path("results/evaluations"))
    parser.add_argument("--tokenizer", help="tokenizer path or Hugging Face ID")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--lm-eval-executable", default="lm-eval")
    parser.add_argument("--smoke-limit", type=int, default=20)
    parser.add_argument("--consolidation-threads", type=int, default=5)
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=15,
        help="print a status line after this many seconds without lm-eval output",
    )
    parser.add_argument(
        "--steps",
        type=lambda value: {int(item) for item in value.split(",")},
        help="comma-separated completed optimizer steps (for example 100,200,500)",
    )
    parser.add_argument(
        "--harness-arg",
        action="append",
        default=[],
        help="one additional lm-eval argument token; repeat as needed",
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands without writing")
    parser.add_argument("--force", action="store_true", help="rerun completed evaluations")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.smoke_limit < 1:
        parser.error("--smoke-limit must be at least 1")
    if args.consolidation_threads < 1:
        parser.error("--consolidation-threads must be at least 1")
    if args.heartbeat_seconds <= 0:
        parser.error("--heartbeat-seconds must be greater than zero")
    tokenizer = args.tokenizer
    if tokenizer is None:
        if args.suite != "pretrain":
            parser.error("--tokenizer is required for instruction checkpoints")
        tokenizer = "HuggingFaceTB/SmolLM2-135M"
    if not args.dry_run and shutil.which(args.lm_eval_executable) is None:
        parser.error(
            f"lm-eval executable not found: {args.lm_eval_executable}; "
            "install the pinned eval extra with `uv sync --extra eval`"
        )

    checkpoints = discover_checkpoints(args.runs)
    if args.steps is not None:
        checkpoints = [item for item in checkpoints if item.training_step in args.steps]
        if not checkpoints:
            parser.error("none of the discovered checkpoints match --steps")
    output_root = args.output.expanduser().resolve()
    suites = requested_suites(args.suite)
    return_code = evaluate(
        checkpoints,
        suites=suites,
        phase=args.phase,
        output_root=output_root,
        tokenizer=tokenizer,
        dtype=args.dtype,
        device=args.device,
        batch_size=args.batch_size,
        executable=args.lm_eval_executable,
        smoke_limit=args.smoke_limit,
        num_threads=args.consolidation_threads,
        heartbeat_seconds=args.heartbeat_seconds,
        extra_args=args.harness_arg,
        dry_run=args.dry_run,
        force=args.force,
        fail_fast=args.fail_fast,
    )
    if not args.dry_run:
        from picoformer.plot_evaluations import plot_evaluations

        try:
            outputs = plot_evaluations(output_root, phase=args.phase)
            print("Updated evaluation plots:")
            for output in outputs:
                print(f"  {output}")
        except ValueError as error:
            print(f"No plots updated: {error}", file=sys.stderr)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
