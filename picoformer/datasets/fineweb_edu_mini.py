"""Create and optionally tokenize a streamed subset of FineWeb-Edu-Dedup.

The source dataset is never downloaded in full. Examples are read from the
Hugging Face streaming API, shuffled with a bounded in-memory buffer, and
written incrementally to Parquet shards.

Example:
    python -m picoformer.datasets.fineweb_edu_mini \
        --output-dir data/datasets/fineweb_edu_mini \
        --num-tokens 5B \
        --tokenize

The resulting directory can be loaded with:
    load_dataset("parquet", data_dir="data/datasets/fineweb_edu_mini")

When ``--tokenize`` is used, the resulting ``train-*.bin`` files can be
passed directly to ``nemo_automodel``'s ``NanogptDataset``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset, Features, Value, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


DATASET_ID = "HuggingFaceTB/smollm-corpus"
DATASET_CONFIG = "fineweb-edu-dedup"
DATASET_SPLIT = "train"
DEFAULT_OUTPUT_DIR = Path("/data/datasets/fineweb_edu_mini")
DEFAULT_NUM_TOKENS = 5_000_000_000
PROGRESS_UPDATE_TOKENS = 10_000
DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM2-135M"
NANOGPT_MAGIC = 2788_95051
NANOGPT_VERSION = 1
NANOGPT_HEADER_SIZE = 256
MAX_HEADER_TOKENS = np.iinfo(np.int32).max


def parse_count(value: str) -> int:
    """Parse counts such as ``5000000000``, ``5B``, ``100M``, or ``25K``."""
    normalized = value.strip().upper().replace("_", "").replace(",", "")
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    multiplier = 1
    if normalized and normalized[-1] in multipliers:
        multiplier = multipliers[normalized[-1]]
        normalized = normalized[:-1]

    try:
        count = float(normalized) * multiplier
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid count: {value!r}") from error

    if not count.is_integer() or count <= 0:
        raise argparse.ArgumentTypeError("count must be a positive integer")
    return int(count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        "--save-folder",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for Parquet shards (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--num-tokens",
        type=parse_count,
        default=DEFAULT_NUM_TOKENS,
        help="Minimum number of GPT-2 tokens to save (default: 5B).",
    )
    parser.add_argument(
        "--tokens-per-shard",
        type=parse_count,
        default=500_000_000,
        help="Approximate tokens per Parquet shard (default: 500M).",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=parse_count,
        default=100_000,
        help="Number of streamed examples held for random shuffling (default: 100K).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--tokenize",
        action="store_true",
        help="Tokenize the saved Parquet dataset into NeMo-compatible .bin shards.",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"Hugging Face tokenizer name or path (default: {DEFAULT_TOKENIZER}).",
    )
    parser.add_argument(
        "--tokenization-batch-size",
        type=parse_count,
        default=1_024,
        help="Documents encoded per tokenizer batch (default: 1024).",
    )
    return parser.parse_args()


def record_token_count(record: dict[str, Any]) -> int:
    """Return the dataset-provided GPT-2 token count for one document."""
    metadata = record.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("token_count") is None:
        raise ValueError("source record is missing metadata.token_count")

    count = int(metadata["token_count"])
    if count < 0:
        raise ValueError(f"source record has a negative token count: {count}")
    return count


def write_shard(
    records: list[dict[str, Any]],
    output_dir: Path,
    shard_index: int,
    features: Features | None,
) -> Features:
    """Write one Parquet shard and return its fixed Hugging Face schema."""
    if features is None:
        # Infer the nested metadata schema from a single, small record, then use
        # 64-bit offsets for document text. Arrow's regular string type overflows
        # once a shard contains roughly 2 GiB of UTF-8 text.
        features = Features(Dataset.from_list(records[:1]).features)
        features["text"] = Value("large_string")

    shard = Dataset.from_list(records, features=features)
    output_path = output_dir / f"train-{shard_index:05d}.parquet"
    temporary_path = output_path.with_suffix(".parquet.incomplete")
    shard.to_parquet(temporary_path)
    temporary_path.replace(output_path)
    tqdm.write(f"Wrote {output_path} ({len(shard):,} documents)")
    return shard.features


def build_subset(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_shards = sorted(output_dir.glob("*.parquet"))
    if existing_shards:
        raise FileExistsError(
            f"{output_dir} already contains Parquet files; choose an empty output directory"
        )

    # For an IterableDataset, shuffle randomizes data-file order and uses a bounded
    # reservoir-like buffer. It does not materialize or download the full dataset.
    source = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)

    records: list[dict[str, Any]] = []
    shard_tokens = 0
    total_tokens = 0
    total_documents = 0
    shard_index = 0
    features: Features | None = None
    progress = tqdm(
        total=args.num_tokens,
        desc="Streaming FineWeb-Edu",
        unit="tok",
        unit_scale=True,
        mininterval=0,
        dynamic_ncols=True,
    )

    try:
        for record in source:
            token_count = record_token_count(record)
            if token_count == 0 or not record.get("text"):
                continue

            records.append(record)
            shard_tokens += token_count
            total_tokens += token_count
            total_documents += 1

            target_progress = min(total_tokens, args.num_tokens)
            pending_tokens = target_progress - int(progress.n)
            complete_updates = pending_tokens // PROGRESS_UPDATE_TOKENS
            if complete_updates:
                progress.update(complete_updates * PROGRESS_UPDATE_TOKENS)
            if target_progress == args.num_tokens and progress.n < args.num_tokens:
                progress.update(args.num_tokens - progress.n)

            if shard_tokens >= args.tokens_per_shard or total_tokens >= args.num_tokens:
                features = write_shard(records, output_dir, shard_index, features)
                records = []
                shard_tokens = 0
                shard_index += 1
                progress.set_postfix(
                    documents=f"{total_documents:,}",
                    shards=shard_index,
                    refresh=True,
                )

            if total_tokens >= args.num_tokens:
                break
        else:
            raise RuntimeError(
                f"stream ended after {total_tokens:,} tokens, before the "
                f"requested {args.num_tokens:,} tokens"
            )
    finally:
        progress.close()

    if records:
        features = write_shard(records, output_dir, shard_index, features)
        shard_index += 1

    manifest = {
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "streaming": True,
        "target_tokens": args.num_tokens,
        "actual_tokens": total_tokens,
        "tokenizer_for_counts": "gpt2 (counts supplied by the source dataset)",
        "documents": total_documents,
        "parquet_shards": shard_index,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "seed": args.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": features.to_dict() if features is not None else None,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Done: saved {total_documents:,} documents and {total_tokens:,} tokens "
        f"across {shard_index:,} Parquet shards in {output_dir}",
        flush=True,
    )


def has_saved_subset(output_dir: Path) -> bool:
    """Return whether a completed local Hugging Face subset is present."""
    manifest_path = output_dir / "manifest.json"
    parquet_paths = sorted(output_dir.glob("*.parquet"))
    if not manifest_path.is_file() or not parquet_paths:
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read existing manifest {manifest_path}: {error}") from error

    expected_shards = manifest.get("parquet_shards")
    if expected_shards is not None and int(expected_shards) != len(parquet_paths):
        raise ValueError(
            f"{output_dir} contains {len(parquet_paths)} Parquet shards, but its "
            f"manifest records {expected_shards}"
        )
    return True


def token_dtype(tokenizer: Any) -> np.dtype[Any]:
    """Choose the narrowest unsigned dtype capable of storing every token ID."""
    vocabulary = tokenizer.get_vocab()
    largest_token_id = max(vocabulary.values(), default=-1)
    if largest_token_id <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if largest_token_id <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    raise ValueError(f"token ID {largest_token_id:,} is too large for uint32")


def write_tokenized_shard(
    parquet_path: Path,
    tokenizer: Any,
    dtype: np.dtype[Any],
    batch_size: int,
) -> tuple[Path, int, int]:
    """Tokenize one Parquet shard into the NanoGPT format used by NeMo."""
    output_path = parquet_path.with_suffix(".bin")
    index_path = parquet_path.with_suffix(".bos.idx")
    if output_path.exists() or index_path.exists():
        raise FileExistsError(
            f"tokenized output already exists for {parquet_path.name}; "
            "remove the corresponding .bin/.bos.idx files to regenerate it"
        )

    temporary_path = output_path.with_suffix(".bin.incomplete")
    temporary_index_path = index_path.with_suffix(".idx.incomplete")
    token_count = 0
    document_count = 0

    try:
        with temporary_path.open("wb+") as output_file, temporary_index_path.open("wb") as index_file:
            np.zeros(NANOGPT_HEADER_SIZE, dtype=np.int32).tofile(output_file)

            parquet = pq.ParquetFile(parquet_path)
            for record_batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
                texts = record_batch.column(0).to_pylist()
                encoded = tokenizer(
                    texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]

                bos_positions: list[int] = []
                batch_tokens: list[int] = []
                for token_ids in encoded:
                    bos_positions.append(token_count + len(batch_tokens))
                    batch_tokens.append(tokenizer.bos_token_id)
                    batch_tokens.extend(token_ids)
                    document_count += 1

                next_token_count = token_count + len(batch_tokens)
                if next_token_count > MAX_HEADER_TOKENS:
                    raise ValueError(
                        f"{parquet_path.name} produces more than {MAX_HEADER_TOKENS:,} tokens; "
                        "use smaller Parquet shards"
                    )
                np.asarray(batch_tokens, dtype=dtype).tofile(output_file)
                np.asarray(bos_positions, dtype=np.int32).tofile(index_file)
                token_count = next_token_count

            header = np.zeros(NANOGPT_HEADER_SIZE, dtype=np.int32)
            header[0] = NANOGPT_MAGIC
            header[1] = NANOGPT_VERSION
            header[2] = token_count
            header[3] = dtype.itemsize
            output_file.seek(0)
            header.tofile(output_file)

        temporary_path.replace(output_path)
        temporary_index_path.replace(index_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        temporary_index_path.unlink(missing_ok=True)
        raise

    return output_path, token_count, document_count


def tokenize_subset(args: argparse.Namespace) -> None:
    """Tokenize all saved Parquet shards and record the result in the manifest."""
    output_dir = args.output_dir.expanduser().resolve()
    parquet_paths = sorted(output_dir.glob("*.parquet"))
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.bos_token_id is None:
        raise ValueError(
            f"tokenizer {args.tokenizer!r} has no bos_token_id; choose a tokenizer with a BOS token"
        )
    dtype = token_dtype(tokenizer)

    total_tokens = 0
    total_documents = 0
    progress = tqdm(parquet_paths, desc="Tokenizing FineWeb-Edu", unit="shard", dynamic_ncols=True)
    for parquet_path in progress:
        output_path, shard_tokens, shard_documents = write_tokenized_shard(
            parquet_path, tokenizer, dtype, args.tokenization_batch_size
        )
        total_tokens += shard_tokens
        total_documents += shard_documents
        progress.set_postfix(tokens=f"{total_tokens:,}", refresh=True)
        tqdm.write(f"Wrote {output_path} ({shard_tokens:,} tokens)")

    manifest["tokenization"] = {
        "tokenizer": args.tokenizer,
        "bos_token_id": tokenizer.bos_token_id,
        "dtype": dtype.name,
        "tokens": total_tokens,
        "documents": total_documents,
        "bin_shards": len(parquet_paths),
        "format": "nanogpt_data_processor",
        "magic": NANOGPT_MAGIC,
        "version": NANOGPT_VERSION,
        "tokenized_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary_manifest = manifest_path.with_suffix(".json.incomplete")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    print(
        f"Done: tokenized {total_documents:,} documents into {total_tokens:,} tokens "
        f"across {len(parquet_paths):,} binary shards in {output_dir}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if args.tokenize and has_saved_subset(output_dir):
        print(f"Found existing Parquet dataset and manifest in {output_dir}; skipping streaming.")
    else:
        build_subset(args)
    if args.tokenize:
        tokenize_subset(args)


if __name__ == "__main__":
    main()
