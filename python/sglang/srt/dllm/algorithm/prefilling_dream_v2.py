import logging
from typing import Any, List, Optional, Sequence, Tuple

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm, DllmRunOutput
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.prefilling_v2 import handoff
from sglang.srt.dllm.prefilling_v2.decoder import (
    close_at_stop_token,
    finish_block,
    generate,
    plan_request,
    start_block,
)
from sglang.srt.dllm.prefilling_v2.stack import PrefillingV2Stack
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_disagg

logger = logging.getLogger(__name__)

KVBuffers = List[Tuple[torch.Tensor, torch.Tensor]]


def kv_pool_layers(
    pool: Any, stack: PrefillingV2Stack
) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    """The pool's per-layer key and value buffers, if Prefilling-dLLM v2 can keep a cache in them.

    That takes a pool of plain BF16 rows of shape [kv_heads, head_dim], one
    buffer per layer.
    """

    config = stack.config
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
    return keys, values


class PrefillingDreamV2(DllmAlgorithm):
    """Confidence-threshold denoising for a Prefilling-dLLM v2 Dream model.

    A request is answered in one call: the model's Prefilling-dLLM v2 stack runs the full
    forward and the cached forwards of every block itself, one request at a
    time, so nothing is carried between scheduling rounds. The keys and values
    of a request live in the KV pool slots the scheduler reserved for it.

    Under prefill-decode disaggregation the prefill server stops after the
    first block's full forward and leaves the request's cache in its KV slots
    (``prefilling_v2/handoff.py``); the decode server resumes from them in ``resume``.

    Algorithm config keys: ``block_size`` (default 32), ``threshold``
    (default 0.9), ``rope_factor`` and ``stop_at_eos`` (default true). The
    YaRN factor follows the request length unless ``rope_factor`` fixes it.
    With ``stop_at_eos`` a block that holds the end-of-sequence token ends the
    answer, so the blocks after it cost no full forward.
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
        self.stop_at_eos = bool(config.algorithm_config.get("stop_at_eos", True))
        self.hands_off = get_disagg().disaggregation_mode == "prefill"
        if self.hands_off and not self.fdfo:
            raise ValueError("a Prefilling-dLLM v2 prefill server requires --dllm-fdfo")
        self._reported_kv_in_pool: Optional[bool] = None

    @staticmethod
    def _stack(model_runner: ModelRunner) -> PrefillingV2Stack:
        stack = getattr(model_runner.model, "prefilling_v2_stack", None)
        if stack is None:
            raise RuntimeError("PrefillingDreamV2 requires an PrefillingDreamV2Model")
        return stack

    def _keep_free_slots_sorted(self, model_runner: ModelRunner) -> None:
        # Every Prefilling-dLLM v2 request is finished and its slots are freed within its
        # round. An allocator that merges and sorts what was freed therefore
        # always hands out one ascending run, which lets the stack keep a
        # request's keys and values in the pool in place.
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

    def _kv_buffers(
        self,
        model_runner: ModelRunner,
        slots: Optional[torch.Tensor],
        physical_length: int,
    ) -> Optional[KVBuffers]:
        layers = kv_pool_layers(
            model_runner.token_to_kv_pool, self._stack(model_runner)
        )
        if layers is None or slots is None:
            return None
        return handoff.contiguous_views(*layers, slots, physical_length)

    def _first_forward(
        self,
        model_runner: ModelRunner,
        token_ids: torch.Tensor,
        prompt_len: int,
        slots: Optional[torch.Tensor],
    ) -> Optional[str]:
        """Prefill server: run the first full forward and store the handoff.

        Returns an error message when the request cannot be handed off.
        """

        stack = self._stack(model_runner)
        config = stack.config
        if prompt_len >= token_ids.numel():
            return "a Prefilling-dLLM v2 request needs at least one token to generate"
        plan = plan_request(
            stack,
            token_ids,
            answer_start=prompt_len,
            block_size=self.block_size,
            rope_factor=self.rope_factor,
        )
        length = plan.physical_length
        layers = kv_pool_layers(model_runner.token_to_kv_pool, stack)
        needed = length + handoff.side_slots(
            length, config.chunk_size, config.group_size
        )
        if layers is None or slots is None or slots.numel() < needed:
            return (
                "the request does not fit a prefill-decode handoff: its "
                f"{needed} KV slots exceed the model's context length"
            )
        block = plan.blocks[0]
        cache = start_block(
            stack,
            plan,
            block,
            threshold=self.threshold,
            kv_buffers=handoff.contiguous_views(*layers, slots, length),
        )
        handoff.write_handoff(
            *layers,
            slots,
            cache,
            plan.input_ids[block.answer_positions.to(stack.device)],
            config,
        )
        return None

    @torch.inference_mode()
    def resume(
        self,
        model_runner: ModelRunner,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        slots: torch.Tensor,
        *,
        transferred: bool,
    ) -> List[int]:
        """Decode server: finish a request whose handoff sits in ``slots``.

        Without a usable handoff (a health check has no prefill peer) the
        request is generated here from scratch.
        """

        stack = self._stack(model_runner)
        config = stack.config
        if max_new_tokens <= 0 or len(prompt_ids) == 0:
            return []
        token_ids = torch.tensor(
            list(prompt_ids) + [config.mask_token_id] * max_new_tokens,
            dtype=torch.long,
        )
        plan = plan_request(
            stack,
            token_ids,
            answer_start=len(prompt_ids),
            block_size=self.block_size,
            rope_factor=self.rope_factor,
        )
        length = plan.physical_length
        layers = kv_pool_layers(model_runner.token_to_kv_pool, stack)
        kv_buffers = None
        if layers is not None:
            kv_buffers = handoff.contiguous_views(*layers, slots, length)

        first = block_before = plan.blocks[0]
        received = None
        if transferred and layers is not None:
            cos, sin = stack.rotary(plan.layout.position_ids, plan.rope_factor)
            received = handoff.read_handoff(
                *layers,
                slots,
                length,
                int(first.answer_positions.numel()),
                plan.layout.key_valid.to(stack.device).unsqueeze(0),
                cos,
                sin,
                config,
            )
        if received is None:
            if transferred:
                logger.warning(
                    "No Prefilling-dLLM v2 handoff found in a transferred request's KV slots; "
                    "generating it on the decode server."
                )
            cache = start_block(
                stack, plan, first, threshold=self.threshold, kv_buffers=kv_buffers
            )
        else:
            cache, answer_ids = received
            plan.input_ids[first.answer_positions.to(stack.device)] = answer_ids
        finish_block(stack, plan, first, cache, threshold=self.threshold)
        del cache
        # A later block starts with a full forward of its own, here.
        for block in plan.blocks[1:]:
            if self.stop_at_eos and close_at_stop_token(stack, plan, block_before):
                break
            block_before = block
            cache = start_block(
                stack, plan, block, threshold=self.threshold, kv_buffers=kv_buffers
            )
            finish_block(stack, plan, block, cache, threshold=self.threshold)
            del cache
        return plan.input_ids[plan.answer_positions].tolist()

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
        stack = self._stack(model_runner)
        self._keep_free_slots_sorted(model_runner)
        slots = forward_batch.out_cache_loc
        if slots is None or slots.numel() != sum(lengths):
            slot_parts = [None] * len(lengths)
        else:
            slot_parts = slots.split(lengths)
        logits_output = LogitsProcessorOutput(next_token_logits=None)

        answers, sequences, states = [], [], []
        for token_ids, request_slots, state in zip(
            forward_batch.input_ids.split(lengths), slot_parts, algo_states
        ):
            prompt_len = int(state["prompt_len"])
            # The request may carry trailing filler that only reserves slots.
            token_ids = token_ids[: int(state.get("num_tokens", token_ids.numel()))]
            if self.hands_off:
                error = self._first_forward(
                    model_runner, token_ids, prompt_len, request_slots
                )
                sequences.append(token_ids)
                states.append(dict(state, handoff_error=error))
                continue
            if prompt_len >= token_ids.numel():
                answer = token_ids.new_empty(0)
            else:
                answer, stats = generate(
                    stack,
                    token_ids,
                    answer_start=prompt_len,
                    block_size=self.block_size,
                    threshold=self.threshold,
                    rope_factor=self.rope_factor,
                    stop_at_eos=self.stop_at_eos,
                    kv_buffers_for=lambda length, s=request_slots: self._kv_buffers(
                        model_runner, s, length
                    ),
                )
                self._report_kv_placement(stats.kv_in_buffers, stats.physical_length)
                answer = answer.to(dtype=token_ids.dtype)
            answers.append(answer)
            sequences.append(torch.cat((token_ids[:prompt_len], answer)))
            states.append(None)

        if self.fdfo:
            # A request is complete after its one call, unless this is a
            # prefill server, whose scheduler hands it to the decode server.
            return (
                logits_output,
                [sequence.tolist() for sequence in sequences],
                None,
                states,
                False,
                [not self.hands_off] * len(sequences),
            )
        if all(answer.numel() == 0 for answer in answers):
            return logits_output, [], None, None, False, None
        return logits_output, answers, None, None, False, None


Algorithm = PrefillingDreamV2
