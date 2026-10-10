"""Configuration and weight containers for the Prefilling-dLLM v2 Dream stack."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

ADAPTER_FILE = "prefilling_v2_adapter.safetensors"
CONFIG_FILE = "prefilling_v2_config.json"
_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class PrefillingV2Config:
    num_layers: int
    route_layers: tuple
    chunk_size: int
    route_local_radius: int
    route_top_k: int
    sliding_radius: int
    lora_scaling: float
    entropy_prior_scale: float
    rope_theta: float
    rope_factor: float
    rope_original_max_positions: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    rms_norm_eps: float
    mask_token_id: int
    pad_token_id: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def group_size(self) -> int:
        return self.num_heads // self.num_kv_heads

    @staticmethod
    def from_files(model_dir: str | Path, hf_config) -> PrefillingV2Config:
        raw = json.loads((Path(model_dir) / CONFIG_FILE).read_text())
        rope = raw["rope_scaling"]
        rope_type = rope.get("rope_type", rope.get("type"))
        if rope_type != "yarn":
            raise ValueError(
                f"Prefilling-dLLM v2 expects YaRN rotary scaling, got {rope_type!r}"
            )
        if set(raw["lora_targets"]) != set(_LORA_TARGETS):
            raise ValueError(f"unsupported LoRA targets {raw['lora_targets']}")
        return PrefillingV2Config(
            num_layers=int(raw["num_layers"]),
            route_layers=tuple(int(i) for i in raw["route_layers"]),
            chunk_size=int(raw["chunk_size"]),
            route_local_radius=int(raw["route_local_radius"]),
            route_top_k=int(raw["route_top_k"]),
            sliding_radius=int(raw["sliding_radius"]),
            lora_scaling=float(raw["lora_scaling"]),
            entropy_prior_scale=float(raw["entropy_prior_scale"]),
            rope_theta=float(raw["rope_theta"]),
            rope_factor=float(rope["factor"]),
            rope_original_max_positions=int(rope["original_max_position_embeddings"]),
            hidden_size=int(hf_config.hidden_size),
            num_heads=int(hf_config.num_attention_heads),
            num_kv_heads=int(hf_config.num_key_value_heads),
            rms_norm_eps=float(hf_config.rms_norm_eps),
            mask_token_id=int(hf_config.mask_token_id),
            pad_token_id=int(hf_config.pad_token_id),
        )


@dataclass
class LowRank:
    """A LoRA pair, kept in the float32 the adapter was trained in."""

    down: torch.Tensor  # [rank, in]
    up: torch.Tensor  # [out, rank]


@dataclass
class RouteQuery:
    """Low-rank residual and per-head RMSNorm weight of a route layer's query."""

    down: torch.Tensor  # [rank, hidden]
    up: torch.Tensor  # [num_heads * head_dim, rank]
    norm_weight: torch.Tensor  # [head_dim]


@dataclass
class LayerWeights:
    qkv_weight: torch.Tensor
    qkv_bias: Optional[torch.Tensor]
    o_weight: torch.Tensor
    gate_up_weight: torch.Tensor
    down_weight: torch.Tensor
    input_norm_weight: torch.Tensor
    post_norm_weight: torch.Tensor
    lora: Dict[str, LowRank] = field(default_factory=dict)
    route_query: Optional[RouteQuery] = None


@dataclass
class PrefillingV2Weights:
    embed: torch.Tensor
    final_norm_weight: torch.Tensor
    lm_head: torch.Tensor
    role_offset: torch.Tensor
    layers: List[LayerWeights]


def attach_adapter(
    weights: PrefillingV2Weights,
    config: PrefillingV2Config,
    adapter: Dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    """Fill the adapter slots of ``weights`` from a converted adapter file."""

    def take(name: str) -> torch.Tensor:
        return adapter.pop(name).to(device=device, dtype=torch.float32)

    adapter = dict(adapter)
    weights.role_offset = take("prefilling_v2.role_offset")
    for index, layer in enumerate(weights.layers):
        prefix = f"prefilling_v2.layers.{index}"
        for target in _LORA_TARGETS:
            layer.lora[target] = LowRank(
                down=take(f"{prefix}.{target}.lora_A"),
                up=take(f"{prefix}.{target}.lora_B"),
            )
        if index in config.route_layers:
            layer.route_query = RouteQuery(
                down=take(f"{prefix}.route_query.down"),
                up=take(f"{prefix}.route_query.up"),
                norm_weight=take(f"{prefix}.route_query.norm_weight"),
            )
    if adapter:
        raise ValueError(f"unused adapter tensors: {sorted(adapter)[:8]}")


def load_adapter_file(model_dir: str | Path) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(Path(model_dir) / ADAPTER_FILE))


def load_from_hf_dir(
    model_dir: str | Path,
    config: PrefillingV2Config,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> PrefillingV2Weights:
    """Load the base weights straight from a Hugging Face Dream checkpoint.

    The q/k/v and gate/up projections are stacked the way SGLang's merged
    linear layers hold them, so this path and the served model run the same
    matrix products.
    """

    from safetensors import safe_open

    model_dir = Path(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    handles = {}

    def get(name: str) -> torch.Tensor:
        shard = index["weight_map"][name]
        if shard not in handles:
            handles[shard] = safe_open(
                str(model_dir / shard), framework="pt", device="cpu"
            )
        return handles[shard].get_tensor(name).to(device=device, dtype=dtype)

    layers = []
    for i in range(config.num_layers):
        prefix = f"model.layers.{i}"
        attn = f"{prefix}.self_attn"
        layers.append(
            LayerWeights(
                qkv_weight=torch.cat(
                    [get(f"{attn}.{p}_proj.weight") for p in "qkv"], dim=0
                ),
                qkv_bias=torch.cat(
                    [get(f"{attn}.{p}_proj.bias") for p in "qkv"], dim=0
                ),
                o_weight=get(f"{attn}.o_proj.weight"),
                gate_up_weight=torch.cat(
                    [
                        get(f"{prefix}.mlp.gate_proj.weight"),
                        get(f"{prefix}.mlp.up_proj.weight"),
                    ],
                    dim=0,
                ),
                down_weight=get(f"{prefix}.mlp.down_proj.weight"),
                input_norm_weight=get(f"{prefix}.input_layernorm.weight"),
                post_norm_weight=get(f"{prefix}.post_attention_layernorm.weight"),
            )
        )
    weights = PrefillingV2Weights(
        embed=get("model.embed_tokens.weight"),
        final_norm_weight=get("model.norm.weight"),
        lm_head=get("lm_head.weight"),
        role_offset=torch.empty(0),
        layers=layers,
    )
    attach_adapter(weights, config, load_adapter_file(model_dir), device)
    return weights
