#!/usr/bin/env python3
"""Tokenize manifested Dolma3 JSONL shards into fixed Dream uint32 packs."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import zstandard as zstd
from transformers import AutoTokenizer


_TOKENIZER_FILES = {
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenization_dream.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
_WORKER_TOKENIZER = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_fingerprint(model_path: Path | str) -> str:
    model_path = Path(model_path)
    files = sorted(
        path
        for path in model_path.iterdir()
        if path.is_file() and path.name in _TOKENIZER_FILES
    )
    if not files:
        raise ValueError(f"no tokenizer files found in {model_path}")

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_and_validate_manifest(manifest_path: Path) -> tuple[dict, list[Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("source manifest must contain a non-empty shards list")

    paths: list[Path] = []
    for index, entry in enumerate(shards):
        if not isinstance(entry, dict) or "path" not in entry:
            raise ValueError(f"invalid source manifest shard at index {index}")
        path = manifest_path.parent / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(entry.get("bytes", -1)):
            raise ValueError(f"source shard byte count mismatch: {path}")
        if _sha256(path) != entry.get("sha256"):
            raise ValueError(f"source shard checksum mismatch: {path}")
        paths.append(path)
    return manifest, paths


def _iter_text_records(paths: Iterable[Path], stats: dict[str, int]) -> Iterator[str]:
    for path in paths:
        with path.open("rb") as raw:
            with zstd.ZstdDecompressor().stream_reader(raw) as reader:
                buffered = bytearray()
                while True:
                    block = reader.read(1024 * 1024)
                    if not block:
                        break
                    buffered.extend(block)
                    while True:
                        newline = buffered.find(b"\n")
                        if newline < 0:
                            break
                        line = bytes(buffered[:newline])
                        del buffered[: newline + 1]
                        text = _parse_text_line(line, stats)
                        if text is not None:
                            yield text
                if buffered:
                    text = _parse_text_line(bytes(buffered), stats)
                    if text is not None:
                        yield text


def _parse_text_line(line: bytes, stats: dict[str, int]) -> str | None:
    try:
        record = json.loads(line.decode("utf-8"))
        text = record["text"]
        if not isinstance(text, str):
            raise TypeError("text is not a string")
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        stats["malformed_records"] += 1
        return None
    stats["records"] += 1
    return text


def _load_tokenizer(model_path: Path | str):
    return AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )


def _init_tokenizer_worker(model_path: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = _load_tokenizer(model_path)


def _tokenize_in_worker(text: str) -> list[int]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    return list(
        _WORKER_TOKENIZER(text, add_special_tokens=False)["input_ids"]
    )


def _tokenize_sequential(texts: Iterable[str], tokenizer) -> Iterator[list[int]]:
    for text in texts:
        yield list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _tokenized_records(
    texts: Iterable[str],
    *,
    tokenizer,
    model_path: Path,
    num_workers: int,
) -> Iterator[list[int]]:
    if num_workers <= 1:
        yield from _tokenize_sequential(texts, tokenizer)
        return
    context = mp.get_context("spawn")
    with context.Pool(
        processes=num_workers,
        initializer=_init_tokenizer_worker,
        initargs=(str(model_path),),
    ) as pool:
        yield from pool.imap(_tokenize_in_worker, texts, chunksize=32)


def build_packed_corpus(
    *,
    manifest_path: Path | str,
    model_path: Path | str,
    output_dir: Path | str,
    physical_seq_len: int,
    chunk_size: int,
    num_workers: int = 1,
    write_buffer_packs: int = 32,
    tokenizer=None,
    tokenizer_revision: str = "local",
    pack_mode: str = "concat_documents",
    max_packs: int | None = None,
) -> dict[str, object]:
    manifest_path = Path(manifest_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    if physical_seq_len <= 0 or physical_seq_len % chunk_size != 0:
        raise ValueError("physical_seq_len must be positive and divisible by chunk_size")
    if chunk_size < 2:
        raise ValueError("chunk_size must reserve at least one text and one landmark slot")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if write_buffer_packs <= 0:
        raise ValueError("write_buffer_packs must be positive")
    if pack_mode not in {"concat_documents", "single_document"}:
        raise ValueError(f"unsupported pack_mode={pack_mode}")
    if max_packs is not None and max_packs <= 0:
        raise ValueError("max_packs must be positive when set")
    if tokenizer is not None and num_workers > 1:
        raise ValueError("an injected tokenizer can only be used with num_workers=1")

    manifest, shard_paths = _load_and_validate_manifest(manifest_path)
    tokenizer = tokenizer or _load_tokenizer(model_path)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Dream tokenizer must define eos_token_id")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    physical_chunks = physical_seq_len // chunk_size
    text_slots_per_pack = physical_chunks * (chunk_size - 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_path = output_dir / "train.bin"
    temporary_binary = output_dir / "train.bin.incomplete"
    valid_path = output_dir / "train.valid.bin"
    temporary_valid = output_dir / "train.valid.bin.incomplete"
    segments_path = output_dir / "train.segments.bin"
    temporary_segments = output_dir / "train.segments.bin.incomplete"
    stats = {"records": 0, "malformed_records": 0}
    pending_tokens: list[int] = []
    pending_valid: list[int] = []
    pending_segments: list[int] = []
    next_segment = 1
    packs_to_write: list[list[int]] = []
    valid_to_write: list[list[int]] = []
    segments_to_write: list[list[int]] = []
    pack_count = 0
    token_count = 0
    padding_tokens = 0
    dropped_tokens = 0
    skipped_short_documents = 0
    long_document_windows = 0

    def flush(token_handle, valid_handle, segment_handle) -> None:
        nonlocal packs_to_write, valid_to_write, segments_to_write
        if not packs_to_write:
            return
        token_handle.write(
            np.asarray(packs_to_write, dtype="<u4").tobytes(order="C")
        )
        valid_handle.write(
            np.asarray(valid_to_write, dtype="u1").tobytes(order="C")
        )
        segment_handle.write(
            np.asarray(segments_to_write, dtype="<u4").tobytes(order="C")
        )
        packs_to_write = []
        valid_to_write = []
        segments_to_write = []

    def finish_pack(token_handle, valid_handle, segment_handle) -> None:
        nonlocal pending_tokens, pending_valid, pending_segments
        nonlocal next_segment, pack_count
        if len(pending_tokens) != text_slots_per_pack:
            raise RuntimeError("attempted to finish a partial pack")
        packs_to_write.append(pending_tokens)
        valid_to_write.append(pending_valid)
        segments_to_write.append(pending_segments)
        pending_tokens = []
        pending_valid = []
        pending_segments = []
        next_segment = 1
        pack_count += 1
        if len(packs_to_write) == write_buffer_packs:
            flush(token_handle, valid_handle, segment_handle)

    texts = _iter_text_records(shard_paths, stats)
    tokenized = _tokenized_records(
        texts,
        tokenizer=tokenizer,
        model_path=model_path,
        num_workers=num_workers,
    )
    with (
        temporary_binary.open("wb") as token_output,
        temporary_valid.open("wb") as valid_output,
        temporary_segments.open("wb") as segment_output,
    ):
        try:
            for token_ids in tokenized:
                document = [int(token_id) for token_id in token_ids]
                document.append(int(eos_token_id))
                token_count += len(document)
                if pack_mode == "single_document":
                    offset = 0
                    emitted = 0
                    while offset + text_slots_per_pack <= len(document):
                        window = document[offset : offset + text_slots_per_pack]
                        pending_tokens = list(window)
                        pending_valid = [1] * text_slots_per_pack
                        pending_segments = [1] * text_slots_per_pack
                        finish_pack(token_output, valid_output, segment_output)
                        offset += text_slots_per_pack
                        emitted += text_slots_per_pack
                        long_document_windows += 1
                        if max_packs is not None and pack_count >= max_packs:
                            break
                    remainder = len(document) - emitted
                    dropped_tokens += remainder
                    if emitted == 0:
                        skipped_short_documents += 1
                    if max_packs is not None and pack_count >= max_packs:
                        break
                    continue

                offset = 0
                while offset < len(document):
                    remaining = text_slots_per_pack - len(pending_tokens)
                    take = min(remaining, len(document) - offset)
                    pending_tokens.extend(document[offset : offset + take])
                    pending_valid.extend([1] * take)
                    pending_segments.extend([next_segment] * take)
                    offset += take

                    if offset == len(document):
                        padding = (-take) % (chunk_size - 1)
                        if padding > text_slots_per_pack - len(pending_tokens):
                            raise RuntimeError("document padding crossed a pack boundary")
                        pending_tokens.extend([int(pad_token_id)] * padding)
                        pending_valid.extend([0] * padding)
                        pending_segments.extend([0] * padding)
                        padding_tokens += padding
                        next_segment += 1

                    if len(pending_tokens) == text_slots_per_pack:
                        finish_pack(token_output, valid_output, segment_output)
                        if max_packs is not None and pack_count >= max_packs:
                            break
                if max_packs is not None and pack_count >= max_packs:
                    break
        finally:
            closer = getattr(tokenized, "close", None)
            if closer is not None:
                closer()

        if pack_mode == "concat_documents":
            dropped_tokens = int(sum(pending_valid))
        flush(token_output, valid_output, segment_output)
        for handle in (token_output, valid_output, segment_output):
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(temporary_binary, binary_path)
    os.replace(temporary_valid, valid_path)
    os.replace(temporary_segments, segments_path)

    metadata: dict[str, object] = {
        "format_version": 2,
        "dtype": "uint32",
        "valid_dtype": "uint8",
        "segment_dtype": "uint32",
        "physical_seq_len": physical_seq_len,
        "chunk_size": chunk_size,
        "real_slots_per_pack": text_slots_per_pack,
        "text_slots_per_pack": text_slots_per_pack,
        "pack_count": pack_count,
        "token_count": token_count,
        "dropped_tokens": dropped_tokens,
        "padding_tokens": padding_tokens,
        "skipped_short_documents": skipped_short_documents,
        "long_document_windows": long_document_windows,
        "records": stats["records"],
        "malformed_records": stats["malformed_records"],
        "tokenizer_path": str(model_path),
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_fingerprint": tokenizer_fingerprint(model_path),
        "tokenizer_length": len(tokenizer),
        "eos_token_id": int(eos_token_id),
        "pad_token_id": int(pad_token_id),
        "source_repo": manifest.get("repo"),
        "source_revision": manifest.get("resolved_revision"),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_shards": [entry["path"] for entry in manifest["shards"]],
        "binary_sha256": _sha256(binary_path),
        "valid_binary": valid_path.name,
        "valid_binary_sha256": _sha256(valid_path),
        "segments_binary": segments_path.name,
        "segments_binary_sha256": _sha256(segments_path),
        "build_parameters": {
            "physical_seq_len": physical_seq_len,
            "chunk_size": chunk_size,
            "num_workers": num_workers,
            "write_buffer_packs": write_buffer_packs,
            "pack_mode": pack_mode,
            "max_packs": max_packs,
        },
        "pack_mode": pack_mode,
        "max_packs": max_packs,
    }
    _write_json_atomic(output_dir / "train.meta.json", metadata)
    return metadata


def validate_packed_corpus(
    binary_path: Path | str,
    metadata_path: Path | str,
    *,
    model_path: Path | str,
    tokenizer=None,
) -> dict[str, object]:
    binary_path = Path(binary_path)
    metadata_path = Path(metadata_path)
    model_path = Path(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    current_fingerprint = tokenizer_fingerprint(model_path)
    if metadata.get("tokenizer_fingerprint") != current_fingerprint:
        raise ValueError("tokenizer fingerprint mismatch")

    expected_bytes = (
        int(metadata["pack_count"]) * int(metadata["real_slots_per_pack"]) * 4
    )
    actual_bytes = binary_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"packed corpus binary size is {actual_bytes}, expected {expected_bytes}"
        )
    tokenizer = tokenizer or _load_tokenizer(model_path)
    tokens = np.memmap(binary_path, mode="r", dtype="<u4")
    max_token_id = int(tokens.max()) if tokens.size else -1
    if max_token_id >= len(tokenizer):
        raise ValueError(
            f"packed corpus token id {max_token_id} exceeds tokenizer length {len(tokenizer)}"
        )
    if _sha256(binary_path) != metadata.get("binary_sha256"):
        raise ValueError("packed corpus binary checksum mismatch")
    if int(metadata.get("format_version", 1)) >= 2:
        valid_path = binary_path.with_name(str(metadata["valid_binary"]))
        segments_path = binary_path.with_name(str(metadata["segments_binary"]))
        expected_elements = int(metadata["pack_count"]) * int(
            metadata["real_slots_per_pack"]
        )
        if valid_path.stat().st_size != expected_elements:
            raise ValueError("valid binary size mismatch")
        if segments_path.stat().st_size != expected_elements * 4:
            raise ValueError("segments binary size mismatch")
        if _sha256(valid_path) != metadata.get("valid_binary_sha256"):
            raise ValueError("valid binary checksum mismatch")
        if _sha256(segments_path) != metadata.get("segments_binary_sha256"):
            raise ValueError("segments binary checksum mismatch")
        valid = np.memmap(valid_path, mode="r", dtype="u1")
        segments = np.memmap(segments_path, mode="r", dtype="<u4")
        if np.any((valid == 0) & (segments != 0)):
            raise ValueError("invalid token slots must have segment id zero")
        if np.any((valid != 0) & (segments == 0)):
            raise ValueError("valid token slots must have a nonzero segment id")
    eos_token_id = int(metadata["eos_token_id"])
    eos_count = int(np.count_nonzero(tokens == eos_token_id))
    preview = tokenizer.decode(tokens[: min(32, tokens.size)].tolist()) if tokens.size else ""
    return {
        "pack_count": int(metadata["pack_count"]),
        "token_count": int(tokens.size),
        "max_token_id": max_token_id,
        "eos_count": eos_count,
        "preview": preview,
        "binary_sha256": metadata["binary_sha256"],
        "tokenizer_fingerprint": current_fingerprint,
        "format_version": int(metadata.get("format_version", 1)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--physical_seq_len", type=int, default=8192)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--write_buffer_packs", type=int, default=32)
    parser.add_argument(
        "--pack_mode",
        choices=("concat_documents", "single_document"),
        default="single_document",
        help="single_document keeps only contiguous one-document windows of "
        "exactly text_slots_per_pack tokens; shorter documents are dropped.",
    )
    parser.add_argument(
        "--max_packs",
        type=int,
        default=0,
        help="Stop after this many packs. 0 means no limit.",
    )
    parser.add_argument("--tokenizer_revision", default="local")
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.validate_only:
        result = validate_packed_corpus(
            args.output_dir / "train.bin",
            args.output_dir / "train.meta.json",
            model_path=args.model_path,
        )
    else:
        result = build_packed_corpus(
            manifest_path=args.manifest,
            model_path=args.model_path,
            output_dir=args.output_dir,
            physical_seq_len=args.physical_seq_len,
            chunk_size=args.chunk_size,
            num_workers=args.num_workers,
            write_buffer_packs=args.write_buffer_packs,
            tokenizer_revision=args.tokenizer_revision,
            pack_mode=args.pack_mode,
            max_packs=(args.max_packs or None),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
