"""Compact layer caches for Dream+route Fast-dLLM inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerKVCache:
    key: torch.Tensor
    value: torch.Tensor
    key_valid: torch.Tensor
    landmark_keys: torch.Tensor | None = None
    prior_bias: torch.Tensor | None = None
    dsa_index_keys: torch.Tensor | None = None
    nsa_raw_key: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.key.ndim != 4 or self.value.shape != self.key.shape:
            raise ValueError(
                "key and value must have identical [B,L,Hkv,D] shape"
            )
        if self.key.shape[0] != 1:
            raise ValueError("Fast-dLLM cache currently requires batch size one")
        if self.key.shape[2] != 4:
            raise ValueError(
                "Dream cache must store four compact KV heads, "
                f"got {self.key.shape[2]}"
            )
        if self.key.dtype != self.value.dtype:
            raise ValueError("key and value dtype must match")
        if self.key.device != self.value.device:
            raise ValueError("key and value device must match")
        if self.key_valid.shape != self.key.shape[:2]:
            raise ValueError(
                "key_valid shape must match the cache batch and sequence"
            )
        if self.key_valid.dtype != torch.bool:
            raise ValueError("key_valid must be boolean")
        if self.key_valid.device != self.key.device:
            raise ValueError("key_valid and K/V device must match")

        if self.dsa_index_keys is not None:
            if self.dsa_index_keys.shape != self.key.shape[:2] + (self.key.shape[-1],):
                raise ValueError("DSA index keys must have shape [B,L,D]")
            if self.dsa_index_keys.device != self.key.device:
                raise ValueError("DSA index keys and attention cache must share device")

        if self.nsa_raw_key is not None:
            if self.nsa_raw_key.shape != self.key.shape:
                raise ValueError("NSA raw keys must match RoPE key shape")
            if self.nsa_raw_key.dtype != self.key.dtype:
                raise ValueError("NSA raw keys dtype must match the cache")
            if self.nsa_raw_key.device != self.key.device:
                raise ValueError("NSA raw keys and attention cache must share device")

        paired = (self.landmark_keys is None) == (self.prior_bias is None)
        if not paired:
            raise ValueError(
                "landmark_keys and prior_bias must be provided together"
            )
        if self.landmark_keys is None:
            return

        batch, _, _, dim = self.key.shape
        landmark_shape = self.landmark_keys.shape
        if (
            self.landmark_keys.ndim != 5
            or landmark_shape[0] != batch
            or landmark_shape[2:] != (4, 7, dim)
        ):
            raise ValueError(
                "landmark_keys must have shape [B,C,4,7,D]"
            )
        expected_prior = landmark_shape[:-1]
        if self.prior_bias.shape != expected_prior:
            raise ValueError(
                f"prior_bias must have shape {expected_prior}, "
                f"got {tuple(self.prior_bias.shape)}"
            )
        if self.landmark_keys.device != self.key.device:
            raise ValueError("landmark_keys and K/V device must match")
        if self.landmark_keys.dtype != self.key.dtype:
            raise ValueError("landmark_keys and K/V dtype must match")
        if self.prior_bias.device != self.key.device:
            raise ValueError("prior_bias and K/V device must match")

    @property
    def physical_length(self) -> int:
        return int(self.key.shape[1])

    def replace(
        self,
        positions: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        positions = positions.to(device=self.key.device, dtype=torch.long)
        if positions.ndim != 1:
            raise ValueError("cache update positions must be one-dimensional")
        if torch.unique(positions).numel() != positions.numel():
            raise ValueError("cache update positions must be unique")
        if bool(
            ((positions < 0) | (positions >= self.physical_length)).any()
        ):
            raise ValueError("cache update positions must be in range")

        expected = (
            self.key.shape[0],
            positions.numel(),
            self.key.shape[2],
            self.key.shape[3],
        )
        if key.shape != expected or value.shape != expected:
            raise ValueError(
                f"replacement K/V shape must be {expected}, got "
                f"{tuple(key.shape)} and {tuple(value.shape)}"
            )
        if key.dtype != self.key.dtype or value.dtype != self.value.dtype:
            raise ValueError("replacement K/V dtype must match the cache")
        if key.device != self.key.device or value.device != self.value.device:
            raise ValueError("replacement K/V device must match the cache")

        self.key.index_copy_(1, positions, key)
        self.value.index_copy_(1, positions, value)


@dataclass
class RouteModelCache:
    layers: tuple[LayerKVCache, ...]
    position_ids: torch.Tensor
    physical_length: int
    prefill_count: int = 1

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("model cache must contain at least one layer")
        if self.physical_length <= 0:
            raise ValueError("physical_length must be positive")
        first = self.layers[0]
        expected_positions = (first.key.shape[0], self.physical_length)
        if self.position_ids.shape != expected_positions:
            raise ValueError(
                f"position_ids must match physical_length with shape "
                f"{expected_positions}, got {tuple(self.position_ids.shape)}"
            )
        if self.position_ids.dtype != torch.long:
            raise ValueError("position_ids must use torch.long")
        if self.position_ids.device != first.key.device:
            raise ValueError("position_ids and cache device must match")
        if self.prefill_count <= 0:
            raise ValueError("prefill_count must be positive")
        for layer in self.layers:
            if layer.key.shape[0] != first.key.shape[0]:
                raise ValueError("all cache layers must share batch size")
            if layer.physical_length != self.physical_length:
                raise ValueError(
                    "all cache layers must match physical_length"
                )
