"""Hydra launcher for NeMo AutoModel hyperparameter sweeps."""

from __future__ import annotations

import argparse
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


def write_automodel_config(cfg: DictConfig, destination: Path) -> None:
    """Write a resolved training config without Hydra launcher-only settings."""
    training_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    # Optimizer choices live outside the inherited AutoModel config so Hydra can
    # switch schemas cleanly (AdamW's ``eps`` vs Muon's ``epsilon``, for example).
    training_cfg["optimizer"] = OmegaConf.to_container(
        training_cfg["sweep_optimizer"], resolve=True
    )
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
