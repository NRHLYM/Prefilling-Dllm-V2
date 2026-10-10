from typing import Any, List, Optional

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm, DllmRunOutput
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.prefilling_v2.decoder import generate
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class PrefillingDreamV2(DllmAlgorithm):
    """Confidence-threshold denoising for a Prefilling-dLLM v2 Dream model.

    A request is answered in one call: the model's Prefilling-dLLM v2 stack runs the full
    forward and the cached forwards of every block itself, one request at a
    time, so nothing is carried between scheduling rounds.

    Algorithm config keys: ``block_size`` (default 32) and ``threshold``
    (default 0.9).
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.block_size = int(config.block_size or 32)
        self.threshold = float(config.algorithm_config.get("threshold", 0.9))
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")

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

        answers, sequences = [], []
        for token_ids, state in zip(
            forward_batch.input_ids.split(lengths), algo_states
        ):
            prompt_len = int(state["prompt_len"])
            if prompt_len >= token_ids.numel():
                answer = token_ids.new_empty(0)
            else:
                answer, _ = generate(
                    stack,
                    token_ids,
                    answer_start=prompt_len,
                    block_size=self.block_size,
                    threshold=self.threshold,
                )
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
