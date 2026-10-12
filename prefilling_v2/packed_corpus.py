"""Read-only fixed-length token packs for Dream full-text training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


_REQUIRED_METADATA = {
    "format_version",
    "dtype",
    "real_slots_per_pack",
    "pack_count",
    "tokenizer_path",
    "tokenizer_fingerprint",
    "eos_token_id",
    "source_manifest_sha256",
}


class DreamPackedCorpus(torch.utils.data.Dataset):
    """Memory-map a validated `[pack_count, real_slots]` uint32 corpus."""

    def __init__(
        self,
        bin_path: str | Path,
        metadata_path: str | Path,
        *,
        expected_real_slots: Optional[int] = None,
        expected_tokenizer_fingerprint: Optional[str] = None,
    ) -> None:
        self.bin_path = Path(bin_path)
        self.metadata_path = Path(metadata_path)
        if not self.bin_path.is_file():
            raise FileNotFoundError(self.bin_path)
        if not self.metadata_path.is_file():
            raise FileNotFoundError(self.metadata_path)

        with self.metadata_path.open("r", encoding="utf-8") as handle:
            metadata: Dict[str, Any] = json.load(handle)
        missing = sorted(_REQUIRED_METADATA - metadata.keys())
        if missing:
            raise ValueError(f"packed corpus metadata is missing: {missing}")
        if metadata["format_version"] not in {1, 2}:
            raise ValueError(
                f"unsupported packed corpus format_version={metadata['format_version']}"
            )
        if metadata["dtype"] != "uint32":
            raise ValueError(
                f"packed corpus dtype must be uint32, got {metadata['dtype']}"
            )

        self.real_slots_per_pack = int(metadata["real_slots_per_pack"])
        self.pack_count = int(metadata["pack_count"])
        if self.real_slots_per_pack <= 0 or self.pack_count < 0:
            raise ValueError(
                "real_slots_per_pack must be positive and pack_count non-negative"
            )
        if (
            expected_real_slots is not None
            and self.real_slots_per_pack != int(expected_real_slots)
        ):
            raise ValueError(
                f"expected {expected_real_slots} real slots, got "
                f"{self.real_slots_per_pack}"
            )
        if (
            expected_tokenizer_fingerprint is not None
            and metadata["tokenizer_fingerprint"]
            != expected_tokenizer_fingerprint
        ):
            raise ValueError(
                "tokenizer fingerprint mismatch: "
                f"metadata={metadata['tokenizer_fingerprint']} "
                f"expected={expected_tokenizer_fingerprint}"
            )

        expected_bytes = self.pack_count * self.real_slots_per_pack * 4
        actual_bytes = self.bin_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"packed corpus binary size is {actual_bytes} bytes, "
                f"expected {expected_bytes}"
            )
        self.metadata = metadata
        self._tokens = np.memmap(
            self.bin_path,
            mode="r",
            dtype=np.uint32,
            shape=(self.pack_count, self.real_slots_per_pack),
        )
        self._valid = None
        self._segments = None
        if metadata["format_version"] >= 2:
            valid_path = self.bin_path.with_name(metadata["valid_binary"])
            segments_path = self.bin_path.with_name(
                metadata["segments_binary"]
            )
            expected_valid_bytes = self.pack_count * self.real_slots_per_pack
            if not valid_path.is_file() or valid_path.stat().st_size != expected_valid_bytes:
                raise ValueError("packed corpus valid binary size mismatch")
            if not segments_path.is_file() or segments_path.stat().st_size != expected_bytes:
                raise ValueError("packed corpus segments binary size mismatch")
            self._valid = np.memmap(
                valid_path,
                mode="r",
                dtype=np.uint8,
                shape=(self.pack_count, self.real_slots_per_pack),
            )
            self._segments = np.memmap(
                segments_path,
                mode="r",
                dtype=np.uint32,
                shape=(self.pack_count, self.real_slots_per_pack),
            )

    def __len__(self) -> int:
        return self.pack_count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += self.pack_count
        if index < 0 or index >= self.pack_count:
            raise IndexError(index)
        clean_ids = torch.from_numpy(
            np.asarray(self._tokens[index], dtype=np.int64)
        )
        valid_tokens = (
            torch.ones(self.real_slots_per_pack, dtype=torch.bool)
            if self._valid is None
            else torch.from_numpy(
                np.asarray(self._valid[index], dtype=np.bool_)
            )
        )
        segment_ids = (
            torch.ones(self.real_slots_per_pack, dtype=torch.long)
            if self._segments is None
            else torch.from_numpy(
                np.asarray(self._segments[index], dtype=np.int64)
            )
        )
        return {
            "clean_ids": clean_ids,
            "valid_tokens": valid_tokens,
            "segment_ids": segment_ids,
            "sample_id": index,
        }
