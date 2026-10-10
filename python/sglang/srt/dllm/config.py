import json
from pathlib import Path
from typing import Any, Optional

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.dllm.prefilling_v2.weights import (
    CONFIG_FILE as PREFILLING_V2_CONFIG_FILE,
)
from sglang.srt.server_args import ServerArgs


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        block_size: Optional[int],
        mask_id: int,
        max_running_requests: int,
        first_done_first_out_mode: bool = False,
        needs_full_prefill: bool = False,
        routing_chunk_size: Optional[int] = None,
        max_physical_length: Optional[int] = None,
        handoff_slots_per_chunk: int = 0,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests
        self.first_done_first_out_mode = first_done_first_out_mode
        self.needs_full_prefill = needs_full_prefill
        # Prefilling-dLLM v2 only: positions per chunk, the last of which is a routing slot.
        self.routing_chunk_size = routing_chunk_size
        self.max_physical_length = max_physical_length
        # Prefilling-dLLM v2 under prefill-decode disaggregation: extra KV slots per chunk
        # that carry the chunk summaries to the decode server.
        self.handoff_slots_per_chunk = handoff_slots_per_chunk

    def physical_length(self, num_tokens: int) -> int:
        """KV slots a request of ``num_tokens`` text tokens occupies.

        Prefilling-dLLM v2 stores one routing slot after every ``routing_chunk_size - 1``
        text tokens and fills its last chunk; under prefill-decode
        disaggregation each chunk takes ``handoff_slots_per_chunk`` more. A
        request whose layout would not fit the context length keeps its token
        count; the model then holds that request's keys and values outside
        the KV pool, and such a request cannot be handed to a decode server.
        """

        chunk = self.routing_chunk_size
        if not chunk or num_tokens <= 0:
            return num_tokens
        num_chunks = -(-num_tokens // (chunk - 1))
        physical = num_chunks * (chunk + self.handoff_slots_per_chunk)
        if self.max_physical_length is not None and physical > self.max_physical_length:
            return num_tokens
        return physical

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
    ):

        cfg = resolving_view(server_args)
        if cfg.dllm_algorithm is None:
            return None

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=cfg.model_path,
            model_revision=cfg.revision,
        )
        DLLM_PARAMS = {
            "LLaDA2MoeModelLM": {"block_size": 32, "mask_id": 156895},
            "SDARForCausalLM": {"block_size": 4, "mask_id": 151669},
            "SDARMoeForCausalLM": {"block_size": 4, "mask_id": 151669},
            "DreamModel": {"block_size": None, "mask_id": 151666},
            "PrefillingDreamV2Model": {"block_size": None, "mask_id": 151666},
        }

        arch = model_config.hf_config.architectures[0]
        if arch in DLLM_PARAMS:
            params = DLLM_PARAMS[arch]
            block_size = params["block_size"]
            mask_id = params["mask_id"]
        else:
            raise RuntimeError(f"Unknown diffusion LLM: {arch}")
        needs_full_prefill = arch in ("DreamModel", "PrefillingDreamV2Model")
        routing_chunk_size = None
        handoff_slots_per_chunk = 0
        if arch == "PrefillingDreamV2Model":
            routing_chunk_size = int(
                json.loads(
                    (Path(cfg.model_path) / PREFILLING_V2_CONFIG_FILE).read_text()
                )["chunk_size"]
            )
            if cfg.disaggregation_mode != "null":
                # One row per query head of a KV group; see prefilling_v2/handoff.py.
                hf_config = model_config.hf_config
                handoff_slots_per_chunk = (
                    hf_config.num_attention_heads // hf_config.num_key_value_heads
                )

        max_running_requests = (
            1 if cfg.max_running_requests is None else cfg.max_running_requests
        )

        algorithm_config = {}
        if cfg.dllm_algorithm_config is not None:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(cfg.dllm_algorithm_config, "r") as f:
                algorithm_config = yaml.safe_load(f)

            # Parse common algorithm configurations
            block_size = algorithm_config.get("block_size", block_size)

        return DllmConfig(
            algorithm=cfg.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
            first_done_first_out_mode=cfg.dllm_fdfo,
            needs_full_prefill=needs_full_prefill,
            routing_chunk_size=routing_chunk_size,
            handoff_slots_per_chunk=handoff_slots_per_chunk,
            max_physical_length=min(
                model_config.context_len,
                cfg.max_total_tokens or model_config.context_len,
            ),
        )
