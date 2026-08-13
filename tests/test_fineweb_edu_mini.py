from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from picoformer.datasets.fineweb_edu_mini import build_subset


class _Stream:
    def __init__(self, records):
        self.records = records

    def shuffle(self, **_kwargs):
        return iter(self.records)


class FineWebEduMiniTest(unittest.TestCase):
    def test_builds_disjoint_training_then_validation_splits(self) -> None:
        records = [
            {"text": f"document-{index}", "metadata": {"token_count": 3}}
            for index in range(5)
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            args = argparse.Namespace(
                output_dir=output_dir,
                num_tokens=5,
                validation_tokens=4,
                tokens_per_shard=100,
                shuffle_buffer_size=10,
                seed=42,
            )
            with patch(
                "picoformer.datasets.fineweb_edu_mini.load_dataset",
                return_value=_Stream(records),
            ):
                build_subset(args)

            train = pq.read_table(output_dir / "train-00000.parquet").column("text").to_pylist()
            validation = pq.read_table(output_dir / "validation-00000.parquet").column("text").to_pylist()
            self.assertEqual(train, ["document-0", "document-1"])
            self.assertEqual(validation, ["document-2", "document-3"])
            self.assertTrue(set(train).isdisjoint(validation))

            manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertEqual(manifest["splits"]["train"]["actual_tokens"], 6)
            self.assertEqual(manifest["splits"]["validation"]["actual_tokens"], 6)


if __name__ == "__main__":
    unittest.main()
