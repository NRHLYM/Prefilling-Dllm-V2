import json
from pathlib import Path

import numpy as np
import zstandard as zstd

from scripts.prefilling_v2.build_dream_packed_corpus import build_packed_corpus


class FakeTokenizer:
    eos_token_id = 99
    vocab_size = 1000

    def __len__(self):
        return self.vocab_size

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [int(token) for token in text.split()]}

    def decode(self, token_ids):
        return " ".join(map(str, token_ids))


def _write_zstd_jsonl(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstd.ZstdCompressor()
    with path.open("wb") as raw:
        with compressor.stream_writer(raw) as compressed:
            for line in lines:
                compressed.write(line.encode("utf-8") + b"\n")


def test_single_document_keeps_only_full_one_doc_windows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shard = source / "shard.jsonl.zst"
    short = " ".join(str(i) for i in range(1, 4))
    long = " ".join(str(i) for i in range(1, 20))
    _write_zstd_jsonl(
        shard,
        [json.dumps({"text": short}), json.dumps({"text": long})],
    )
    manifest_path = source / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "repo": "test",
                "resolved_revision": "x",
                "shards": [
                    {
                        "path": shard.name,
                        "bytes": shard.stat().st_size,
                        "sha256": __import__("hashlib")
                        .sha256(shard.read_bytes())
                        .hexdigest(),
                    }
                ],
            }
        )
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "tokenizer_config.json").write_text("{}")
    (model_path / "vocab.json").write_text("{}")

    metadata = build_packed_corpus(
        manifest_path=manifest_path,
        model_path=model_path,
        output_dir=tmp_path / "packed",
        physical_seq_len=8,
        chunk_size=4,
        tokenizer=FakeTokenizer(),
        pack_mode="single_document",
        write_buffer_packs=1,
    )
    # 8 physical = 6 text slots. long doc is 19 tokens + EOS = 20, so 3 windows.
    assert metadata["text_slots_per_pack"] == 6
    assert metadata["pack_count"] == 3
    assert metadata["skipped_short_documents"] == 1
    assert metadata["long_document_windows"] == 3
    assert metadata["padding_tokens"] == 0
    tokens = np.fromfile(tmp_path / "packed" / "train.bin", dtype="<u4").reshape(3, 6)
    valid = np.fromfile(tmp_path / "packed" / "train.valid.bin", dtype="u1")
    segments = np.fromfile(tmp_path / "packed" / "train.segments.bin", dtype="<u4")
    np.testing.assert_array_equal(valid, np.ones(18, dtype="u1"))
    np.testing.assert_array_equal(segments, np.ones(18, dtype="<u4"))
    np.testing.assert_array_equal(tokens[0], np.arange(1, 7, dtype="<u4"))
    np.testing.assert_array_equal(tokens[1], np.arange(7, 13, dtype="<u4"))
    np.testing.assert_array_equal(tokens[2], np.arange(13, 19, dtype="<u4"))
