"""NeMo AutoModel dataset factory for the local FineWeb-Edu mini subset.

The Parquet files consumed here are produced by ``fineweb_edu_mini.py``.
Unlike that script's optional NanoGPT output, this factory keeps the Hugging
Face dataset as its backing store and tokenizes documents lazily.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import load_dataset

from nemo_automodel.components.datasets.lazy_mapped_dataset import LazyMappedDataset
from nemo_automodel.components.datasets.llm.formatting_utils import (
    _add_pad_token,
    format_prompt_completion,
)


DEFAULT_DATASET_PATH = Path("/data/datasets/fineweb_edu_mini")


def _format_document(
    example: dict[str, Any],
    tokenizer: Any,
    eos_token_id: int,
    pad_token_id: int,
    seq_length: int | None,
    padding: str | bool,
    truncation: str | bool,
) -> dict[str, Any]:
    """Tokenize one document for standard shifted next-token prediction."""
    text = example.get("text")
    if not isinstance(text, str):
        raise ValueError("FineWeb-Edu examples must contain a string 'text' field")

    # SmolLM2's tokenizer does not insert special tokens by default. Adding
    # boundaries here makes the first and final token of each independent
    # document trainable, just as the producer's NanoGPT representation does.
    bos_token = getattr(tokenizer, "bos_token", None) or getattr(tokenizer, "eos_token", "")
    eos_token = getattr(tokenizer, "eos_token", None) or bos_token

    return format_prompt_completion(
        tokenizer=tokenizer,
        prompt=bos_token,
        answer=text + eos_token,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        seq_length=seq_length,
        padding=padding,
        truncation=truncation,
        answer_only_loss_mask=False,
    )


def make_fineweb_edu(
    tokenizer: Any,
    seq_length: int | None = None,
    limit_dataset_samples: int | None = None,
    fp8: bool = False,
    split: str = "train",
    dataset_name: str | Path = DEFAULT_DATASET_PATH,
    padding: str | bool = False,
    truncation: str | bool = False,
    cache_size: int | None = 10_000,
    *,
    dataset_path: str | Path | None = None,
) -> LazyMappedDataset:
    """Load the local FineWeb-Edu mini Parquet dataset for NeMo AutoModel.

    Args:
        tokenizer: Hugging Face-compatible tokenizer used by the model.
        seq_length: Optional tokenization length used for padding/truncation.
        limit_dataset_samples: Optionally load at most this many documents.
        fp8: Accepted for compatibility with NeMo dataset factories; unused.
        split: Hugging Face split expression, normally ``"train"``.
        dataset_name: Directory containing the ``train-*.parquet`` files made
            by ``fineweb_edu_mini.py``.
        padding: Hugging Face tokenizer padding strategy.
        truncation: Hugging Face tokenizer truncation strategy.
        cache_size: Number of lazily tokenized samples cached in memory. Use
            zero to disable caching or ``None`` to cache the complete dataset.
        dataset_path: More explicit alias for ``dataset_name``. Do not pass
            both with different values.

    Returns:
        A map-style dataset whose samples contain ``input_ids``, ``labels``,
        ``attention_mask``, and NeMo padding metadata.
    """
    del fp8  # Kept in the signature to match other NeMo LLM factories.

    if tokenizer is None:
        raise ValueError("tokenizer is required")
    if dataset_path is not None:
        if dataset_name != DEFAULT_DATASET_PATH and Path(dataset_name) != Path(dataset_path):
            raise ValueError("dataset_name and dataset_path refer to different locations")
        dataset_name = dataset_path
    if limit_dataset_samples is not None:
        if not isinstance(limit_dataset_samples, int) or isinstance(limit_dataset_samples, bool):
            raise TypeError("limit_dataset_samples must be an int")
        if limit_dataset_samples < 0:
            raise ValueError("limit_dataset_samples must be non-negative")
        if "[" not in split:
            split = f"{split}[:{limit_dataset_samples}]"
        else:
            logging.warning(
                "Dataset split %s already has a slice; ignoring limit_dataset_samples",
                split,
            )

    dataset_dir = Path(dataset_name).expanduser()
    parquet_files = sorted(dataset_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no Parquet shards found in {dataset_dir!s}")
    dataset = load_dataset("parquet", data_files=[str(path) for path in parquet_files], split=split)
    if "text" not in dataset.column_names:
        raise ValueError(
            f"FineWeb-Edu dataset at {dataset_name!s} has no 'text' column; "
            f"found {dataset.column_names}"
        )

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    pad_token_id = _add_pad_token(tokenizer)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    def format_document(example: dict[str, Any]) -> dict[str, Any]:
        return _format_document(
            example,
            tokenizer,
            eos_token_id,
            pad_token_id,
            seq_length,
            padding,
            truncation,
        )

    return LazyMappedDataset(dataset, format_document, cache_size=cache_size)
