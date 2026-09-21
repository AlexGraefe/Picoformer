"""Launch NeMo AutoModel training with a token-budget-derived WSD schedule."""

from __future__ import annotations

import argparse
import re
import subprocess
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from uuid import uuid4

from omegaconf import DictConfig, OmegaConf

from picoformer.sweep import automodel_command, configure_token_schedule


def _tokens_from_billions(value: float | None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("--num-tokens-b must be finite and positive in ablation mode")
    billions = Decimal(str(value))
    if not billions.is_finite() or billions <= 0:
        raise ValueError("--num-tokens-b must be finite and positive in ablation mode")
    return int((billions * 1_000_000_000).to_integral_value(rounding=ROUND_CEILING))


def _ablation_directory(path: Path) -> Path:
    return path.with_name(path.name + "_ablation")


def _latest_checkpoint(root: Path) -> Path:
    """Follow NeMo's latest pointer, falling back to completed training states."""
    def complete(path: Path) -> bool:
        return (
            (path / "step_scheduler.pt").is_file()
            and (path / "model").is_dir()
            and (path / "optim").is_dir()
        )

    latest = root / "LATEST"
    if complete(latest):
        return latest.resolve()
    pointer = root / "LATEST.txt"
    if pointer.is_file():
        latest = root / pointer.read_text().strip()
        if complete(latest):
            return latest.resolve()
    candidates = []
    for path in root.glob("epoch_*_step_*"):
        match = re.fullmatch(r"epoch_(\d+)_step_(\d+)", path.name)
        if match and complete(path):
            candidates.append((int(match[2]), int(match[1]), path))
    if not candidates:
        raise ValueError(f"No training checkpoint found in {root}")
    return max(candidates)[2].resolve()


def _configure_ablation(
    cfg: DictConfig, destination: Path, num_tokens: int, resume_from: Path | None
) -> None:
    root = Path(cfg.checkpoint.checkpoint_dir).expanduser().resolve()
    checkpoint = resume_from.expanduser().resolve() if resume_from else _latest_checkpoint(root)
    if not all((checkpoint / name).exists() for name in ("step_scheduler.pt", "model", "optim")):
        raise ValueError(f"Checkpoint lacks full training state: {checkpoint}")

    import torch

    state = torch.load(checkpoint / "step_scheduler.pt", map_location="cpu", weights_only=True)
    completed_steps = int(state["step"])
    if completed_steps < 0:
        raise ValueError("Checkpoint step must be non-negative")
    global_batch_size = int(cfg.step_scheduler.global_batch_size)
    sequence_length = int(cfg.dataset.seq_len)
    if min(global_batch_size, sequence_length) <= 0:
        raise ValueError("global_batch_size and dataset.seq_len must be positive")
    tokens_per_step = global_batch_size * sequence_length
    additional_steps = (num_tokens + tokens_per_step - 1) // tokens_per_step
    decay_steps = int(cfg.lr_scheduler.get("wsd_decay_steps") or 0)
    if decay_steps <= 0:
        raise ValueError("Ablation requires a positive configured linear decay duration")
    end_step = completed_steps + additional_steps
    cfg.step_scheduler.max_steps = end_step
    # max_steps controls termination; allow enough epochs even when resuming
    # near the end of a finite dataloader or after the original final epoch.
    cfg.step_scheduler.num_epochs = max(
        int(cfg.step_scheduler.get("num_epochs") or 1), int(state["epoch"]) + additional_steps + 1
    )
    cfg.lr_scheduler.lr_decay_steps = end_step
    cfg.lr_scheduler.wsd_decay_steps = min(decay_steps, additional_steps)
    cfg.lr_scheduler.lr_warmup_steps = 0
    cfg.lr_scheduler.override_opt_param_scheduler = True
    cfg.lr_scheduler.use_checkpoint_opt_param_scheduler = False
    cfg.checkpoint.restore_from = str(checkpoint)
    cfg.checkpoint.enabled = True
    cfg.checkpoint.checkpoint_dir = str(_ablation_directory(root))

    if cfg.get("wandb") is not None:
        # Flatten extra kwargs so an old nested id/resume cannot win over the
        # new run settings in NeMo's WandbConfig.from_kwargs.
        extra = cfg.wandb.pop("extra", None)
        if extra:
            cfg.wandb = OmegaConf.merge(extra, cfg.wandb)
        cfg.wandb.name = str(cfg.wandb.get("name") or root.name) + "_ablation"
        cfg.wandb.id = uuid4().hex
        cfg.wandb.resume = "never"
        for key in ("resume_from", "fork_from"):
            cfg.wandb.pop(key, None)
        wandb_dir = cfg.wandb.get("dir")
        cfg.wandb.dir = str(
            _ablation_directory(Path(wandb_dir).expanduser().resolve())
            if wandb_dir else destination.parent.resolve()
        )


def write_automodel_config(
    cfg: DictConfig, destination: Path, *, resume_from: Path | None = None,
    mode: str = "train", num_tokens_b: float | None = None,
) -> None:
    """Resolve a standalone training YAML without changing the source config."""
    training_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if mode not in ("train", "ablation"):
        raise ValueError(f"Unknown training mode: {mode}")
    if mode == "train" and num_tokens_b is not None:
        raise ValueError("--num-tokens-b is only supported in ablation mode")
    if resume_from is not None:
        checkpoint = resume_from.expanduser().resolve()
        if not checkpoint.is_dir():
            raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
        # Use an absolute path so NeMo cannot interpret a bare directory name
        # relative to checkpoint.checkpoint_dir. Enable loading even if the
        # input config disabled checkpointing.
        OmegaConf.update(training_cfg, "checkpoint.restore_from", str(checkpoint))
        OmegaConf.update(training_cfg, "checkpoint.enabled", True)
    local_batch_size = int(training_cfg.step_scheduler.local_batch_size)
    if local_batch_size <= 0:
        raise ValueError("step_scheduler.local_batch_size must be positive")
    if int(training_cfg.step_scheduler.global_batch_size) % local_batch_size:
        raise ValueError("global_batch_size must be divisible by local_batch_size")
    training_cfg.dataloader.batch_size = local_batch_size
    if "validation_dataloader" in training_cfg:
        training_cfg.validation_dataloader.batch_size = local_batch_size

    configured_decay = training_cfg.lr_scheduler.get("wsd_decay_steps")
    if "num_tokens_B" in training_cfg or mode == "train":
        original_tokens_b = Decimal(str(training_cfg.num_tokens_B))
        if not original_tokens_b.is_finite() or original_tokens_b <= 0:
            raise ValueError("num_tokens_B must be finite and positive")
        original_tokens = int((original_tokens_b * 1_000_000_000).to_integral_value(rounding=ROUND_CEILING))
        configure_token_schedule(
            training_cfg,
            num_tokens=original_tokens,
            final_decay_ratio=float(training_cfg.get("final_decay_ratio", 0.1)),
            final_lr_percentage=float(training_cfg.get("final_lr_percentage", 1.0)),
        )
    if mode == "ablation":
        # An explicit decay duration takes precedence; otherwise retain the
        # duration derived from the ORIGINAL budget, preserving its slope.
        if configured_decay is not None and int(configured_decay) > 0:
            training_cfg.lr_scheduler.wsd_decay_steps = int(configured_decay)
        _configure_ablation(training_cfg, destination, _tokens_from_billions(num_tokens_b), resume_from)
    training_cfg.lr_scheduler.lr_decay_style = "WSD"
    training_cfg.lr_scheduler.lr_wsd_decay_style = "linear"
    # Resolve any references to launcher settings before removing those settings.
    OmegaConf.resolve(training_cfg)
    for key in ("num_tokens_B", "final_decay_ratio", "final_lr_percentage"):
        training_cfg.pop(key, None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(OmegaConf.to_yaml(training_cfg, resolve=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/optimal_training_config.yaml")
    )
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--mode", choices=("train", "ablation"), default="train")
    parser.add_argument(
        "--num-tokens-b", "--num-tokens", dest="num_tokens_b", type=float,
        help="Additional training tokens in billions, e.g. 0.5 (required for --mode ablation)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume training state from a NeMo checkpoint directory (path relative to cwd or absolute)",
    )
    parser.add_argument(
        "--output-config", type=Path, default=Path("outputs/train/automodel.yaml")
    )
    parser.add_argument("--dry-run", action="store_true", help="Write YAML without training")
    args = parser.parse_args()
    if args.mode == "ablation":
        args.output_config = _ablation_directory(args.output_config.parent.resolve()) / args.output_config.name
    if args.nproc_per_node <= 0:
        parser.error("--nproc-per-node must be positive")
    if args.config.resolve() == args.output_config.resolve():
        parser.error("--output-config must differ from the input --config")

    try:
        write_automodel_config(
            OmegaConf.load(args.config), args.output_config, resume_from=args.resume_from,
            mode=args.mode, num_tokens_b=args.num_tokens_b,
        )
    except ValueError as exc:
        parser.error(str(exc))
    generated = OmegaConf.load(args.output_config)
    print(f"Generated AutoModel config: {args.output_config}", flush=True)
    if args.resume_from is not None or args.mode == "ablation":
        print(f"Resuming from checkpoint: {generated.checkpoint.restore_from}", flush=True)
    print(
        f"Training until optimizer step {generated.step_scheduler.max_steps}; "
        f"local batch size: {generated.step_scheduler.local_batch_size}",
        flush=True,
    )
    if args.mode == "ablation":
        tokens_per_step = int(generated.step_scheduler.global_batch_size) * int(generated.dataset.seq_len)
        num_tokens = _tokens_from_billions(args.num_tokens_b)
        additional_steps = (num_tokens + tokens_per_step - 1) // tokens_per_step
        decay_steps = int(generated.lr_scheduler.wsd_decay_steps)
        print(
            f"Ablation: {additional_steps * tokens_per_step} additional tokens "
            f"({additional_steps} steps), {additional_steps - decay_steps} constant-LR steps, "
            f"{decay_steps} linear-decay steps; final LR: {generated.lr_scheduler.get('min_lr', 'default')}",
            flush=True,
        )
    if not args.dry_run:
        if args.mode == "ablation" and generated.get("wandb") is not None:
            Path(generated.wandb.dir).mkdir(parents=True, exist_ok=True)
        command = automodel_command(args.output_config, args.nproc_per_node)
        print(f"Launching: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
