from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from nemo_automodel.components.datasets.llm.nanogpt_dataset import load_bin_shard


SCRIPT_PATH = Path(__file__).parents[1] / "datasets" / "fineweb_edu_mini.py"
SPEC = importlib.util.spec_from_file_location("fineweb_edu_mini", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
fineweb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fineweb)


class FakeTokenizer:
    bos_token_id = 7

    def __len__(self) -> int:
        return 100

    def get_vocab(self) -> dict[str, int]:
        return {"token": 99}

    def __call__(self, texts: list[str], **_: object) -> dict[str, list[list[int]]]:
        return {"input_ids": [[ord(character) for character in text] for text in texts]}


class FineWebTokenizationTest(unittest.TestCase):
    def test_writes_new_nanogpt_format_loadable_by_nemo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            parquet_path = Path(temporary_dir) / "train-00000.parquet"
            pq.write_table(pa.table({"text": ["ab", "c"]}), parquet_path)

            output_path, token_count, document_count = fineweb.write_tokenized_shard(
                parquet_path,
                FakeTokenizer(),
                np.dtype(np.uint16),
                batch_size=1,
            )

            header = np.fromfile(output_path, dtype=np.int32, count=256)
            self.assertEqual(header[:4].tolist(), [278_895_051, 1, 5, 2])
            self.assertEqual(token_count, 5)
            self.assertEqual(document_count, 2)
            self.assertEqual(load_bin_shard(output_path).tolist(), [7, 97, 98, 7, 99])
            self.assertEqual(
                np.fromfile(output_path.with_suffix(".bos.idx"), dtype=np.int32).tolist(),
                [0, 3],
            )

    def test_recognizes_complete_existing_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            pq.write_table(pa.table({"text": ["text"]}), output_dir / "train-00000.parquet")
            (output_dir / "manifest.json").write_text(
                json.dumps({"parquet_shards": 1}), encoding="utf-8"
            )

            self.assertTrue(fineweb.has_saved_subset(output_dir))


if __name__ == "__main__":
    unittest.main()
