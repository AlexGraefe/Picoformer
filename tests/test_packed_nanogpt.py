from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from picoformer.datasets.packed_nanogpt import (
    PackedNanogptDataset,
    build_packed_metadata,
    mask_cross_document_labels,
    packed_nanogpt_collater,
)


def write_nanogpt_shard(path: Path, tokens: list[int]) -> None:
    header = np.zeros(256, dtype=np.int32)
    header[:4] = [278_895_051, 1, len(tokens), 2]
    with path.open("wb") as output:
        header.tofile(output)
        np.asarray(tokens, dtype=np.uint16).tofile(output)


class PackedMetadataTest(unittest.TestCase):
    def test_shared_bos_eos_starts_new_indexed_segment(self) -> None:
        mask, positions = build_packed_metadata(
            [0, 10, 11, 0, 20, 0],
            bos_token=0,
            eos_token=0,
        )
        self.assertEqual(mask, [1, 1, 1, 2, 2, 3])
        self.assertEqual(positions, [0, 1, 2, 0, 1, 0])

    def test_distinct_eos_ends_a_segment(self) -> None:
        mask, positions = build_packed_metadata(
            [1, 10, 2, 1, 20, 2],
            bos_token=1,
            eos_token=2,
        )
        self.assertEqual(mask, [1, 1, 1, 2, 2, 2])
        self.assertEqual(positions, [0, 1, 2, 0, 1, 2])

    def test_masks_only_true_cross_document_targets(self) -> None:
        shared_labels = [10, 0, 20]
        mask_cross_document_labels(
            [0, 10, 0], shared_labels, bos_token=0, eos_token=0
        )
        self.assertEqual(shared_labels, [10, 0, 20])

        distinct_labels = [10, 2, 1, 20]
        mask_cross_document_labels(
            [1, 10, 2, 1], distinct_labels, bos_token=1, eos_token=2
        )
        self.assertEqual(distinct_labels, [10, 2, -100, 20])


class PackedNanogptDatasetTest(unittest.TestCase):
    def test_streams_overlapping_lookahead_without_skipping_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            shard = Path(temporary_dir) / "train-00000.bin"
            write_nanogpt_shard(shard, [0, 10, 11, 0, 20, 21, 0, 30, 31])
            dataset = PackedNanogptDataset(
                shard,
                seq_len=4,
                bos_token=0,
                eos_token=0,
                attn_implementation="sdpa",
            )
            iterator = dataset._process_file_tokens(str(shard), False, 0, None)

            first = next(iterator)
            second = next(iterator)

            self.assertEqual(first["input_ids"], [0, 10, 11, 0])
            self.assertEqual(first["labels"], [10, 11, 0, 20])
            self.assertEqual(first["attention_mask"], [1, 1, 1, 2])
            self.assertEqual(first["position_ids"], [0, 1, 2, 0])
            self.assertEqual(second["input_ids"], [20, 21, 0, 30])
            self.assertEqual(second["labels"], [21, 0, 30, 31])
            self.assertEqual(second["attention_mask"], [1, 1, 2, 2])

    def test_collater_preserves_indexed_flash_attention_mask(self) -> None:
        sample = {
            "input_ids": [0, 10, 0, 20],
            "labels": [10, 0, 20, 21],
            "attention_mask": [1, 1, 2, 2],
            "position_ids": [0, 1, 0, 1],
        }
        batch = packed_nanogpt_collater([sample])
        self.assertEqual(tuple(batch["input_ids"].shape), (1, 4))
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 2, 2]])
        self.assertEqual(batch["position_ids"].tolist(), [[0, 1, 0, 1]])
        self.assertEqual(batch["_packed_seq_ids"].tolist(), [[1, 1, 2, 2]])


if __name__ == "__main__":
    unittest.main()
