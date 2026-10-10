"""Dream continued-trained with Prefilling-dLLM v2.

The model directory is a Dream checkpoint plus ``prefilling_v2_adapter.safetensors`` and
``prefilling_v2_config.json`` (see ``benchmark/prefilling_v2/convert_checkpoint.py``). The base
weights load as Dream's do. Generation does not go through ``forward``: the
``PrefillingDreamV2`` dLLM algorithm drives ``prefilling_v2_stack`` directly, because Prefilling-dLLM v2's
attention reads a per-request cache of its own instead of the KV pool.
"""

from typing import Dict, Iterable, Optional, Tuple

import torch

from sglang.srt.dllm.prefilling_v2.stack import PrefillingV2Stack
from sglang.srt.dllm.prefilling_v2.weights import (
    LayerWeights,
    PrefillingV2Config,
    PrefillingV2Weights,
    attach_adapter,
    load_adapter_file,
)
from sglang.srt.models.dream import DreamModel
from sglang.srt.runtime_context import get_parallel

_ADAPTER_PREFIX = "prefilling_v2."


class PrefillingDreamV2Model(DreamModel):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config, prefix)
        if get_parallel().tp_size != 1:
            raise ValueError(
                "PrefillingDreamV2Model supports tensor parallel size 1 only"
            )
        if quant_config is not None:
            raise ValueError(
                "PrefillingDreamV2Model does not support quantized weights"
            )
        self.prefilling_v2_model_dir = getattr(config, "_name_or_path", None)
        if not self.prefilling_v2_model_dir:
            raise ValueError(
                "PrefillingDreamV2Model needs the model directory of its config"
            )
        self.prefilling_v2_config = PrefillingV2Config.from_files(
            self.prefilling_v2_model_dir, config
        )
        self._prefilling_v2_adapter: Dict[str, torch.Tensor] = {}
        self._prefilling_v2_stack: Optional[PrefillingV2Stack] = None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        def base_weights():
            for name, tensor in weights:
                if name.startswith(_ADAPTER_PREFIX):
                    self._prefilling_v2_adapter[name] = tensor
                else:
                    yield name, tensor

        return super().load_weights(base_weights())

    @property
    def prefilling_v2_stack(self) -> PrefillingV2Stack:
        """The Prefilling-dLLM v2 stack over this model's loaded parameters, built on first use."""

        if self._prefilling_v2_stack is None:
            # The weight loader skips files the checkpoint index does not list.
            adapter = self._prefilling_v2_adapter or load_adapter_file(
                self.prefilling_v2_model_dir
            )
            layers = [
                LayerWeights(
                    qkv_weight=layer.self_attn.qkv_proj.weight.data,
                    qkv_bias=layer.self_attn.qkv_proj.bias.data,
                    o_weight=layer.self_attn.o_proj.weight.data,
                    gate_up_weight=layer.mlp.gate_up_proj.weight.data,
                    down_weight=layer.mlp.down_proj.weight.data,
                    input_norm_weight=layer.input_layernorm.weight.data,
                    post_norm_weight=layer.post_attention_layernorm.weight.data,
                )
                for layer in self.model.layers
            ]
            weights = PrefillingV2Weights(
                embed=self.model.embed_tokens.weight.data,
                final_norm_weight=self.model.norm.weight.data,
                lm_head=self.lm_head.weight.data,
                role_offset=torch.empty(0),
                layers=layers,
            )
            attach_adapter(
                weights, self.prefilling_v2_config, adapter, weights.embed.device
            )
            self._prefilling_v2_adapter = {}
            self._prefilling_v2_stack = PrefillingV2Stack(
                self.prefilling_v2_config, weights
            )
        return self._prefilling_v2_stack


EntryClass = PrefillingDreamV2Model
