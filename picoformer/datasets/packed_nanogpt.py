"""Bounded-memory NanoGPT reader with document-aware packed attention.

The standard NeMo ``NanogptDataset`` yields contiguous fixed-length token
slices, but it does not describe document boundaries inside those slices.
This subclass scans only the current slice for BOS/EOS markers and emits the
indexed attention mask and reset position IDs used by NeMo's NEAT packing.
The token shards remain memory-mapped; only ``seq_len + 1`` tokens are copied
into RAM for each yielded sample.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from nemo_automodel.components.datasets.llm.nanogpt_dataset import (
    NanogptDataset,
    load_bin_shard,
)
from nemo_automodel.components.datasets.utils import neat_packed_collater
from nemo_automodel.components.models.common.packing import configure_packing


CROSS_ENTROPY_IGNORE_IDX = -100


def _normalize_token_ids(token_ids: int | Sequence[int] | None) -> frozenset[int]:
    if token_ids is None:
        return frozenset()
    if isinstance(token_ids, int):
        return frozenset((token_ids,))
    return frozenset(int(token_id) for token_id in token_ids)


def build_packed_metadata(
    input_ids: Sequence[int],
    *,
    bos_token: int | Sequence[int] | None,
    eos_token: int | Sequence[int] | None,
) -> tuple[list[int], list[int]]:
    """Build NEAT-style indexed attention and per-document position IDs.

    BOS belongs to the sequence it starts, while a distinct EOS belongs to
    the sequence it ends. When BOS and EOS are the same token (as in SmolLM),
    each occurrence starts a new sequence; the first occurrence in a slice
    remains sequence 1.
    """
    bos_tokens = _normalize_token_ids(bos_token)
    eos_tokens = _normalize_token_ids(eos_token)
    if not bos_tokens and not eos_tokens:
        raise ValueError("at least one BOS or EOS token ID is required")

    distinct_eos_tokens = eos_tokens - bos_tokens
    attention_mask: list[int] = []
    position_ids: list[int] = []
    sequence_id = 1
    sequence_position = 0

    previous_token: int | None = None
    for index, token_id in enumerate(input_ids):
        # A BOS marker or the token after a distinct EOS begins a new document.
        # If EOS is followed by BOS these conditions still create one segment.
        if index > 0 and (token_id in bos_tokens or previous_token in distinct_eos_tokens):
            sequence_id += 1
            sequence_position = 0

        attention_mask.append(sequence_id)
        position_ids.append(sequence_position)
        sequence_position += 1

        previous_token = token_id

    return attention_mask, position_ids


def mask_cross_document_labels(
    input_ids: Sequence[int],
    labels: list[int],
    *,
    bos_token: int | Sequence[int] | None,
    eos_token: int | Sequence[int] | None,
) -> None:
    """Ignore next-token targets that cross a distinct EOS/BOS boundary.

    When BOS and EOS share an ID, predicting the next document's marker is
    also the valid end-of-document target and is intentionally retained.
    """
    bos_tokens = _normalize_token_ids(bos_token)
    eos_tokens = _normalize_token_ids(eos_token)
    distinct_eos_tokens = eos_tokens - bos_tokens
    distinct_bos_tokens = bos_tokens - eos_tokens
    for index, (input_token, target_token) in enumerate(zip(input_ids, labels, strict=True)):
        if input_token in distinct_eos_tokens or target_token in distinct_bos_tokens:
            labels[index] = CROSS_ENTROPY_IGNORE_IDX


class PackedNanogptDataset(NanogptDataset):
    """Stream fixed-size, document-isolated packs from NanoGPT shards.

    This deliberately bypasses NeMo's eager ``neat_pack_dataset`` conversion:
    the source is already densely packed and fixed-width, so scanning each
    memory-mapped slice is enough to reconstruct the same boundary metadata.
    """

    def __init__(
        self,
        file_pattern: str | Sequence[str],
        seq_len: int,
        *,
        bos_token: int | Sequence[int] | None = None,
        eos_token: int | Sequence[int] | None = None,
        shuffle_files: bool = False,
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if bos_token is None and eos_token is None:
            raise ValueError("at least one of bos_token or eos_token must be provided")

        # ``align_to_bos`` would discard the tail of every document longer than
        # seq_len. Contiguous chunks retain all complete training transitions.
        super().__init__(
            file_pattern=file_pattern,
            seq_len=seq_len,
            bos_token=None,
            shuffle_files=shuffle_files,
            align_to_bos=False,
        )
        self.document_bos_tokens = bos_token
        self.document_eos_tokens = eos_token
        self.attn_implementation = attn_implementation

        # The recipe normally applies this patch only while eagerly running
        # NEAT. Our data is pre-packed, so configure the same model-side support
        # when the dataset is instantiated (after the model has been built).
        configure_packing(attn_implementation=attn_implementation)

    def _process_file_tokens(
        self,
        file: str,
        split_single_file: bool,
        file_start_pos: int,
        file_end_pos: int | None,
    ) -> Iterator[dict[str, list[int]]]:
        tokens = load_bin_shard(file)
        pos = file_start_pos if split_single_file else 0
        max_pos = (
            min(file_end_pos, len(tokens))
            if split_single_file and file_end_pos is not None
            else len(tokens)
        )

        while pos + self.seq_len < max_pos:
            # Copy one bounded chunk. The final token is look-ahead for the
            # shifted labels and becomes the first input token of the next pack.
            chunk = tokens[pos : pos + self.seq_len + 1].tolist()
            input_ids = [int(token_id) for token_id in chunk[:-1]]
            labels = [int(token_id) for token_id in chunk[1:]]
            attention_mask, position_ids = build_packed_metadata(
                input_ids,
                bos_token=self.document_bos_tokens,
                eos_token=self.document_eos_tokens,
            )
            mask_cross_document_labels(
                input_ids,
                labels,
                bos_token=self.document_bos_tokens,
                eos_token=self.document_eos_tokens,
            )
            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
            pos += self.seq_len


def packed_nanogpt_collater(batch: list[dict]) -> dict:
    """Pickleable FlashAttention-2 collater for pre-packed NanoGPT samples."""
    return neat_packed_collater(batch, attn_implementation="flash_attention_2")


__all__ = [
    "PackedNanogptDataset",
    "build_packed_metadata",
    "mask_cross_document_labels",
    "packed_nanogpt_collater",
]
