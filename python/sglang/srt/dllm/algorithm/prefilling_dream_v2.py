import logging
from typing import Any, List, Optional, Tuple

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm, DllmRunOutput
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.prefilling_v2.decoder import generate
from sglang.srt.dllm.prefilling_v2.weights import PrefillingV2Config
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def kv_pool_buffers(
    pool: Any, slots: torch.Tensor, physical_length: int, config: PrefillingV2Config
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Per-layer views of the KV pool over a request's slots.

    The Prefilling-dLLM v2 kernels read keys and values as one contiguous tensor per layer,
    so the views exist only when the request's first ``physical_length`` slots
    are one ascending run of a pool that stores plain BF16 keys and values per
    layer. Otherwise returns None.
    """

    keys = getattr(pool, "k_buffer", None)
    values = getattr(pool, "v_buffer", None)
    if not isinstance(keys, list) or not isinstance(values, list):
        return None
    if len(keys) != config.num_layers or len(values) != config.num_layers:
        return None
    row_shape = (config.num_kv_heads, config.head_dim)
    if any(
        buffer.dtype != torch.bfloat16 or tuple(buffer.shape[1:]) != row_shape
        for buffer in (keys[0], values[0])
    ):
        return None
    if slots.numel() < physical_length:
        return None
    start = int(slots[0])
    run = torch.arange(
        start, start + physical_length, dtype=slots.dtype, device=slots.device
    )
    if not torch.equal(slots[:physical_length], run):
        return None
    stop = start + physical_length
    return [(key[start:stop], value[start:stop]) for key, value in zip(keys, values)]


class PrefillingDreamV2(DllmAlgorithm):
    """Confidence-threshold denoising for a Prefilling-dLLM v2 Dream model.

    A request is answered in one call: the model's Prefilling-dLLM v2 stack runs the full
    forward and the cached forwards of every block itself, one request at a
    time, so nothing is carried between scheduling rounds. The keys and values
    of a request live in the KV pool slots the scheduler reserved for it.

    Algorithm config keys: ``block_size`` (default 32), ``threshold``
    (default 0.9) and ``rope_factor``. The YaRN factor follows the request
    length unless ``rope_factor`` fixes it.
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.block_size = int(config.block_size or 32)
        self.threshold = float(config.algorithm_config.get("threshold", 0.9))
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        rope_factor = config.algorithm_config.get("rope_factor")
        self.rope_factor = None if rope_factor is None else float(rope_factor)
        if self.rope_factor is not None and self.rope_factor < 1.0:
            raise ValueError(f"rope_factor must be at least 1, got {rope_factor}")
        self._reported_kv_in_pool: Optional[bool] = None

    def _keep_free_slots_sorted(self, model_runner: ModelRunner) -> None:
        # Every Prefilling-dLLM v2 request is finished and its slots are freed within its
        # round. An allocator that merges and sorts what was freed therefore
        # always hands out one ascending run, which kv_pool_buffers needs.
        allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)
        if allocator is not None and getattr(allocator, "need_sort", None) is False:
            allocator.need_sort = True

    def _report_kv_placement(self, in_pool: bool, physical_length: int) -> None:
        if in_pool == self._reported_kv_in_pool:
            return
        self._reported_kv_in_pool = in_pool
        if in_pool:
            logger.info(
                "Prefilling-dLLM v2 requests keep their keys and values in the KV pool."
            )
        else:
            logger.warning(
                "A Prefilling-dLLM v2 request of %d positions could not use its KV pool slots; "
                "its keys and values are held outside the pool.",
                physical_length,
            )

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states: Optional[List[Any]] = None,
    ) -> DllmRunOutput:
        lengths = forward_batch.extend_seq_lens_cpu
        if lengths is None or algo_states is None:
            raise RuntimeError(
                "PrefillingDreamV2 requires per-request lengths and states"
            )
        stack = getattr(model_runner.model, "prefilling_v2_stack", None)
        if stack is None:
            raise RuntimeError("PrefillingDreamV2 requires an PrefillingDreamV2Model")
        self._keep_free_slots_sorted(model_runner)
        pool = getattr(model_runner, "token_to_kv_pool", None)
        slots = forward_batch.out_cache_loc
        if slots is None or slots.numel() != sum(lengths):
            slot_parts = [None] * len(lengths)
        else:
            slot_parts = slots.split(lengths)

        answers, sequences = [], []
        for token_ids, request_slots, state in zip(
            forward_batch.input_ids.split(lengths), slot_parts, algo_states
        ):
            prompt_len = int(state["prompt_len"])
            # The request may carry trailing filler that only reserves slots.
            token_ids = token_ids[: int(state.get("num_tokens", token_ids.numel()))]
            if prompt_len >= token_ids.numel():
                answer = token_ids.new_empty(0)
            else:

                def kv_buffers_for(physical_length, request_slots=request_slots):
                    if request_slots is None:
                        return None
                    return kv_pool_buffers(
                        pool, request_slots, physical_length, stack.config
                    )

                answer, stats = generate(
                    stack,
                    token_ids,
                    answer_start=prompt_len,
                    block_size=self.block_size,
                    threshold=self.threshold,
                    rope_factor=self.rope_factor,
                    kv_buffers_for=kv_buffers_for,
                )
                self._report_kv_placement(stats.kv_in_buffers, stats.physical_length)
                answer = answer.to(dtype=token_ids.dtype)
            answers.append(answer)
            sequences.append(torch.cat((token_ids[:prompt_len], answer)))

        logits_output = LogitsProcessorOutput(next_token_logits=None)
        if self.fdfo:
            # Every request is complete after its one call.
            return (
                logits_output,
                [sequence.tolist() for sequence in sequences],
                None,
                [None] * len(sequences),
                False,
                [True] * len(sequences),
            )
        if all(answer.numel() == 0 for answer in answers):
            return logits_output, [], None, None, False, None
        return logits_output, answers, None, None, False, None


Algorithm = PrefillingDreamV2
