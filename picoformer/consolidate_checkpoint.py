"""Export a NeMo AutoModel checkpoint as Hugging Face safetensors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile


_INTERNAL_METADATA = {
    "fqn_to_file_index_mapping.json",
    "fqn_to_dtype_mapping.json",
}


def is_consolidated_model(path: Path) -> bool:
    """Return whether ``path`` contains a loadable-looking HF model bundle."""
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (
            (path / "model.safetensors").is_file()
            or (
                any(path.glob("*.safetensors"))
                and (path / "model.safetensors.index.json").is_file()
            )
        )
    )


def _repair_single_shard_index(output_dir: Path) -> None:
    """Add the index omitted by older AutoModel releases for a single shard."""
    shards = sorted(output_dir.glob("model-*-of-*.safetensors"))
    index_path = output_dir / "model.safetensors.index.json"
    if index_path.exists() or len(shards) != 1:
        return
    from safetensors import safe_open

    shard = shards[0]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    with shard.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), byteorder="little")
    payload = {
        "metadata": {"total_size": shard.stat().st_size - 8 - header_size},
        "weight_map": {key: shard.name for key in keys},
    }
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, index_path)


def consolidate_checkpoint(checkpoint: Path, *, num_threads: int = 5) -> Path:
    """Consolidate one checkpoint, returning its Hugging Face model directory."""
    checkpoint = checkpoint.expanduser().resolve()
    model_dir = checkpoint / "model"
    metadata_dir = model_dir / ".hf_metadata"
    output_dir = model_dir / "consolidated"

    if output_dir.is_dir():
        _repair_single_shard_index(output_dir)
    if is_consolidated_model(output_dir):
        return output_dir
    if output_dir.exists():
        raise RuntimeError(
            f"incomplete consolidated directory already exists: {output_dir}; "
            "move or remove it before retrying"
        )
    if not model_dir.is_dir():
        raise FileNotFoundError(f"checkpoint model directory not found: {model_dir}")
    shards = sorted(model_dir.glob("shard-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no sharded safetensors found in {model_dir}")

    index_path = metadata_dir / "fqn_to_file_index_mapping.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"consolidation index not found: {index_path}")
    with index_path.open(encoding="utf-8") as stream:
        fqn_to_index = {key: int(value) for key, value in json.load(stream).items()}

    dtype_path = metadata_dir / "fqn_to_dtype_mapping.json"
    fqn_to_dtype = None
    if dtype_path.is_file():
        with dtype_path.open(encoding="utf-8") as stream:
            fqn_to_dtype = json.load(stream)

    # Import lazily so discovery, plotting, and --dry-run do not initialize torch.
    from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
        consolidate_safetensors_files,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".consolidated-", dir=model_dir) as temporary:
        temporary_path = Path(temporary)
        consolidate_safetensors_files(
            input_dir=str(model_dir),
            output_dir=str(temporary_path),
            fqn_to_index_mapping=fqn_to_index,
            num_threads=num_threads,
            fqn_to_dtype_mapping=fqn_to_dtype,
        )
        for item in metadata_dir.iterdir():
            if item.name in _INTERNAL_METADATA:
                continue
            destination = temporary_path / item.name
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)
        if not is_consolidated_model(temporary_path):
            raise RuntimeError(f"consolidation did not create a complete model in {temporary_path}")
        os.replace(temporary_path, output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="epoch_<N>_step_<N> checkpoint")
    parser.add_argument("--num-threads", type=int, default=5)
    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error("--num-threads must be at least 1")
    print(consolidate_checkpoint(args.checkpoint, num_threads=args.num_threads))


if __name__ == "__main__":
    main()
