#!/usr/bin/env python3
"""Fast-dLLM prefill + decode latency at 16k/32k/64k/128k.

One synthetic sequence, 128 masked answer tokens, one Fast-dLLM block.
Prefill = first cached capture. Decode = cached_forwards until 128 slots fill
(threshold 0.9). Warmup 3, then 10 timed trials; report medians.
YaRN = L/2048. Do not launch from the notebook.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


def _load_ruler_eval():
    path = Path(__file__).resolve().parent / "eval_ruler.py"
    spec = importlib.util.spec_from_file_location("_ruler_eval_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sync() -> None:
    for idx in range(torch.cuda.device_count()):
        torch.cuda.synchronize(idx)


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs))


_HAYSTACK = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again.\n"
)


def _prompt_ids(tokenizer, n_text: int, seed: int) -> list[int]:
    ids = tokenizer(_HAYSTACK, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError("empty latency haystack")
    ids = (ids * ((n_text // len(ids)) + 2))[:n_text]
    vocab = max(int(getattr(tokenizer, "vocab_size", 1) or 1), 1)
    ids[0] = (int(ids[0]) + int(seed) * 1315423911) % vocab
    return ids


def _decode_until_filled(
    *,
    decoder,
    generated: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    block,
    cache,
    landmark_positions,
    mask_token_id: int,
) -> tuple[int, int]:
    from prefilling_v2.fastdllm_v1 import _rows_for_positions, select_confidence_transfers

    answer_positions = block.answer_positions.to(generated.device)
    predictor_positions = block.predictor_positions.to(generated.device)
    cached_forwards = 0
    routing_calls = 0
    while torch.any(generated[0, answer_positions] == mask_token_id):
        cached = decoder.cached_forward(
            generated,
            block,
            cache,
            landmark_positions=landmark_positions,
        )
        cached_forwards += 1
        routing_calls += int(cached.routing_calls)
        predictor_rows = _rows_for_positions(
            cached.query_positions, predictor_positions
        )
        candidate_logits = cached.logits.index_select(1, predictor_rows)
        active_rows = torch.where(generated[0, answer_positions] == mask_token_id)[0]
        selected_rows, token_ids, _ = select_confidence_transfers(
            candidate_logits,
            active_rows,
            threshold=decoder.threshold,
            mask_token_id=mask_token_id,
        )
        generated[0, answer_positions[selected_rows]] = token_ids
        if cached_forwards > 4096:
            raise RuntimeError("decode loop exceeded 4096 cached_forwards")
    return cached_forwards, routing_calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("prefilling_v2", "dense", "hybrid"))
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--max_seq_len", type=int, required=True)
    parser.add_argument("--answer_tokens", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer_parallel_gpus", type=int, default=1)
    args = parser.parse_args()
    if args.answer_tokens % args.block_length:
        raise SystemExit("answer_tokens must be a multiple of block_length")
    if args.max_seq_len not in {16384, 32768, 65536, 131072}:
        raise SystemExit("this sweep is 16k/32k/64k/128k only")

    ruler = _load_ruler_eval()
    from prefilling_v2.fastdllm_v1 import DreamRouteFastDLLM
    from prefilling_v2.longbench_eval import (
        build_fastdllm_block_layouts,
        build_generation_layout,
        build_plain_fastdllm_block_layouts,
        build_plain_generation_layout,
    )
    from prefilling_v2.train_fulltext import (
        _build_model_and_tokenizer,
        _kernel_fallback_count,
        _set_seed,
        parse_args as parse_training_args,
        resolve_landmark_token_id,
    )
    from scripts.prefilling_v2.eval_longbench_mfen import (
        configure_eval_trainables,
        load_trainables,
    )

    os.environ.pop("RULER_KEEP_TRAIN_YARN", None)
    os.environ.pop("RULER_YARN_FACTOR", None)
    training_args = parse_training_args(
        ["--config", args.training_config, "--no_gradient_checkpointing"]
    )
    ruler.apply_yarn(training_args, args.max_seq_len)
    is_route = training_args.attention_mode == "route"
    _set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, tokenizer, plan = _build_model_and_tokenizer(training_args, device)
    configure_eval_trainables(model, training_args)
    n_train = load_trainables(model, Path(args.checkpoint))
    if int(args.layer_parallel_gpus) >= 2:
        ruler.shard_dream_layers_two_gpus(model)
    model.eval()
    wrap_idx = None
    if str(training_args.attention_mode) == "dense":
        sliding = list(getattr(plan, "sliding_window_layers", []) or [])
        dense_slots = list(getattr(plan, "dense_layers", []) or [])
        wrap_idx = dense_slots if sliding and dense_slots else None
        ruler.enable_dense_fastdllm_cache(
            model,
            local_window=int(args.max_seq_len),
            chunk_size=int(training_args.chunk_size),
            layer_indices=wrap_idx,
        )
    decoder = DreamRouteFastDLLM(
        model=model,
        mask_token_id=int(tokenizer.mask_token_id),
        threshold=args.threshold,
        bootstrap="confidence",
        use_cache=True,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    slots = ruler.text_slot_budget(
        args.max_seq_len,
        is_route=is_route,
        chunk_size=int(training_args.chunk_size),
    )
    prompt_ids = _prompt_ids(
        tokenizer, slots - int(args.answer_tokens), args.seed
    )
    if is_route:
        layout = build_generation_layout(
            prompt_ids=prompt_ids,
            answer_tokens=args.answer_tokens,
            physical_length=args.max_seq_len,
            chunk_size=int(training_args.chunk_size),
            mask_token_id=int(tokenizer.mask_token_id),
            pad_token_id=int(pad_token_id),
            landmark_token_id=int(tokenizer.mask_token_id),
        )
        blocks = build_fastdllm_block_layouts(
            layout,
            logical_block_size=int(args.answer_tokens),
            chunk_size=int(training_args.chunk_size),
        )
        landmark_positions = layout.landmark_positions.to(device)
    else:
        layout = build_plain_generation_layout(
            prompt_ids=prompt_ids,
            answer_tokens=args.answer_tokens,
            physical_length=args.max_seq_len,
            mask_token_id=int(tokenizer.mask_token_id),
            pad_token_id=int(pad_token_id),
        )
        blocks = build_plain_fastdllm_block_layouts(
            layout, logical_block_size=int(args.answer_tokens)
        )
        landmark_positions = None
    if len(blocks) != 1:
        raise SystemExit(f"expected one Fast-dLLM block, got {len(blocks)}")
    block = blocks[0]
    base_ids = layout.input_ids.unsqueeze(0).to(device)
    attention_mask = layout.attention_mask.unsqueeze(0).to(device)
    position_ids = layout.position_ids.unsqueeze(0).to(device)
    mask_token_id = int(tokenizer.mask_token_id)

    def run_prefill(generated: torch.Tensor):
        return decoder.prefill(
            generated,
            attention_mask,
            position_ids,
            landmark_positions=landmark_positions,
        )

    def run_decode(generated: torch.Tensor, cache):
        return _decode_until_filled(
            decoder=decoder,
            generated=generated,
            attention_mask=attention_mask,
            position_ids=position_ids,
            block=block,
            cache=cache,
            landmark_positions=landmark_positions,
            mask_token_id=mask_token_id,
        )

    for _ in range(int(args.warmup)):
        generated = base_ids.clone()
        pref = run_prefill(generated)
        run_decode(generated, pref.cache)
        del pref
        _sync()
        if _kernel_fallback_count(model):
            raise RuntimeError("kernel fallback during warmup")

    prefill_ms: list[float] = []
    decode_ms: list[float] = []
    n_cached: list[int] = []
    peak = 0
    torch.cuda.reset_peak_memory_stats()
    for _ in range(int(args.repeats)):
        generated = base_ids.clone()
        _sync()
        t0 = time.perf_counter()
        pref = run_prefill(generated)
        _sync()
        prefill_ms.append((time.perf_counter() - t0) * 1000.0)
        _sync()
        t1 = time.perf_counter()
        cached_forwards, _routing = run_decode(generated, pref.cache)
        _sync()
        decode_ms.append((time.perf_counter() - t1) * 1000.0)
        n_cached.append(int(cached_forwards))
        del pref
        if _kernel_fallback_count(model):
            raise RuntimeError("kernel fallback during timed trial")
        for idx in range(torch.cuda.device_count()):
            peak = max(peak, int(torch.cuda.max_memory_allocated(idx)))

    payload = {
        "arm": args.arm,
        "max_seq_len": int(args.max_seq_len),
        "answer_tokens": int(args.answer_tokens),
        "yarn_factor": float((training_args.model_rope_scaling or {}).get("factor") or 0),
        "attention_mode": training_args.attention_mode,
        "trainable_tensors": int(n_train),
        "layer_parallel_gpus": int(args.layer_parallel_gpus),
        "dense_wrap_layers": wrap_idx,
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "cached_forwards": n_cached,
        "prefill_ms_median": _median(prefill_ms),
        "decode_ms_median": _median(decode_ms),
        "decode_ms_per_token_median": _median(decode_ms) / float(args.answer_tokens),
        "cached_forwards_median": _median([float(x) for x in n_cached]),
        "peak_memory_bytes": peak,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
