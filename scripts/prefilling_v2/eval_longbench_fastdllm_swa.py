#!/usr/bin/env python3
"""Evaluate trained Dream SWA-2048 with Fast-dLLM KV-cache decoding."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from prefilling_v2.fastdllm_v1 import (
    DreamRouteFastDLLM,
    FastDLLMGenerationStats,
)
from prefilling_v2.longbench_eval import (
    append_jsonl_fsync,
    build_plain_fastdllm_block_layouts,
    build_plain_generation_layout,
    load_resumable_jsonl,
    merge_evaluation_shards,
    qa_f1_score,
    shard_indices,
)
from prefilling_v2.train_fulltext import (
    _build_model_and_tokenizer,
    _kernel_fallback_count,
    _set_seed,
    parse_args as parse_training_args,
)
from scripts.prefilling_v2.eval_longbench_mfen import (
    decode_answer,
    load_trainables,
    tokenize_prompt,
)


DEFAULT_DATA = "data/LongBench/data/multifieldqa_en.jsonl"
DEFAULT_DATA_ROOT = "data/LongBench/data"
DEFAULT_PROMPTS = (
    "data/LongBench/config/dataset2prompt.json"
)
DEFAULT_CHECKPOINT = "outputs/swa-yarn16-nolmk-step500/step-500"
DEFAULT_OUTPUT = (
    "outputs/swa-yarn16-nolmk-step500/longbench_fastdllm_all21"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training_config",
        default="configs/prefilling_v2/dolma3_8k_dual_gpu.json",
    )
    parser.add_argument("--task", default="multifieldqa_en")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", default=None)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prompt_config", default=DEFAULT_PROMPTS)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical_length", type=int, default=8192)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--local_window", type=int, default=1024)
    parser.add_argument("--answer_tokens", type=int, default=64)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--bootstrap", choices=("first_token", "confidence"), default="confidence")
    parser.add_argument("--qcal_scale", type=float, default=1.0)
    parser.add_argument(
        "--eval_trainable_scope",
        choices=("full", "qcal_only"),
        default=None,
        help="Override only the checkpoint parameter scope used by evaluation.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--allow_empty_predictions", action="store_true")
    parser.set_defaults(model_variant="dream_swa2048_nolmk_fastdllm_32k")
    return parser


def rouge_l_score(prediction: str, answers: list[str]) -> float:
    from rouge import Rouge
    scorer = Rouge()
    best = 0.0
    for answer in answers:
        try:
            score = scorer.get_scores([prediction], [answer], avg=True)["rouge-l"]["f"]
        except Exception:
            score = 0.0
        best = max(best, float(score))
    return best


def score_prediction(task: str, prediction: str, answers: list[str]) -> tuple[float, str]:
    if task in {"gov_report", "qmsum", "multi_news", "samsum"}:
        return rouge_l_score(prediction, answers), "rouge_l"
    return qa_f1_score(prediction, answers), "qa_f1"


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_record(
    *,
    args: argparse.Namespace,
    rank: int,
    index: int,
    example: dict[str, object],
    prediction: str,
    prompt_metadata: dict[str, int | str],
    seconds: float,
    generation_stats: FastDLLMGenerationStats,
    fallback_count: int,
) -> dict[str, object]:
    answers = [str(answer) for answer in example.get("answers", [])]
    score, metric = score_prediction(args.task, prediction, answers)
    return {
        "task": args.task,
        "rank": int(rank),
        "index": int(index),
        "example_id": example.get("_id", index),
        "question": example.get("input", ""),
        "prediction": prediction,
        "answers": answers,
        "score": score,
        "metric": metric,
        "length": example.get("length"),
        "prompt_metadata": prompt_metadata,
        "seconds": float(seconds),
        "model_variant": args.model_variant,
        "cache_mode": "native_route_compact_gqa",
        "physical_length": int(args.physical_length),
        "answer_tokens": int(args.answer_tokens),
        "block_length": int(args.block_length),
        "threshold": float(args.threshold),
        "bootstrap": args.bootstrap,
        "chunk_size": int(args.chunk_size),
        "local_window": int(args.local_window),
        "qcal_scale": float(args.qcal_scale),
        "compact_kv_heads": 4,
        "full_prefills": generation_stats.full_prefills,
        "cached_forwards": generation_stats.cached_forwards,
        "routing_calls": generation_stats.routing_calls,
        "recomputed_tokens": generation_stats.recomputed_tokens,
        "peak_memory_bytes": generation_stats.peak_memory_bytes,
        "fallback_count": int(fallback_count),
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.physical_length <= 0 or args.physical_length % args.chunk_size:
        raise ValueError("physical_length must be positive and divisible by chunk_size")
    if args.chunk_size < 2:
        raise ValueError("chunk_size must be at least 2")
    if args.local_window <= 0:
        raise ValueError("local_window must be positive")
    if args.answer_tokens <= 0 or args.block_length <= 0:
        raise ValueError("answer_tokens and block_length must be positive")
    if args.answer_tokens % args.block_length:
        raise ValueError("answer_tokens must be divisible by block_length")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    if not math.isfinite(args.qcal_scale):
        raise ValueError("qcal_scale must be finite")
    if not args.merge and args.rank is None:
        raise ValueError("--rank is required unless --merge is used")


def _model_variant(args: argparse.Namespace, training_args: argparse.Namespace) -> str:
    return (
        f"dream_swa_total{2 * training_args.local_window}_nolmk_"
        f"fastdllm_v1_{args.physical_length}"
    )


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    if args.data is None:
        args.data = str(Path(args.data_root) / f"{args.task}.jsonl")
    examples = _read_jsonl(args.data)
    expected_examples = len(examples)
    if args.limit > 0:
        expected_examples = min(expected_examples, args.limit * args.world_size)
    if args.merge:
        metrics = merge_evaluation_shards(
            args.output_dir,
            total_examples=expected_examples,
            expected_variant=None,
            require_nonempty_predictions=not args.allow_empty_predictions,
        )
        records = load_resumable_jsonl(Path(args.output_dir) / "merged.jsonl")
        metric = next(
            iter({str(record.get("metric", "qa_f1")) for record in records}),
            "unknown",
        )
        metrics["task"] = args.task
        metrics["metric"] = metric
        metrics["score_avg"] = metrics["qa_f1"]
        Path(args.output_dir, "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, sort_keys=True), flush=True)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rank-{args.rank}.jsonl"
    completed = {
        int(record["index"]) for record in load_resumable_jsonl(output_path)
    }
    assigned = shard_indices(len(examples), args.rank, args.world_size)
    if args.limit > 0:
        assigned = assigned[: args.limit]
    pending = [index for index in assigned if index not in completed]
    templates = json.loads(Path(args.prompt_config).read_text(encoding="utf-8"))
    if args.task not in templates:
        raise KeyError(f"missing LongBench prompt template for task={args.task}")
    template = templates[args.task]

    training_args = parse_training_args(
        ["--config", args.training_config, "--no_gradient_checkpointing"]
    )
    if args.eval_trainable_scope is not None:
        training_args.route_trainable_scope = args.eval_trainable_scope
    if (
        training_args.max_length != args.physical_length
        or training_args.chunk_size != args.chunk_size
        or training_args.local_window != args.local_window
        or training_args.attention_mode != "swa"
        or not training_args.no_kernel_fallback
    ):
        raise ValueError("training config does not match SWA fast-cache evaluation")
    args.model_variant = _model_variant(args, training_args)
    _set_seed(args.seed + args.rank)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, tokenizer, plan = _build_model_and_tokenizer(training_args, device)
    if plan.route_layers or not plan.sliding_window_layers:
        raise ValueError("SWA evaluation requires all layers to be sliding-window layers")
    # The all-layer ablation intentionally has a different layer plan.
    from scripts.prefilling_v2.eval_longbench_mfen import configure_eval_trainables
    configure_eval_trainables(model, training_args)
    trainable_tensors = load_trainables(model, Path(args.checkpoint))
    qcal_modules = 0
    model.eval()
    decoder = DreamRouteFastDLLM(
        model=model,
        mask_token_id=int(tokenizer.mask_token_id),
        threshold=args.threshold,
        bootstrap=args.bootstrap,
    )

    max_prompt_tokens = args.physical_length - args.answer_tokens
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    print(
        json.dumps(
            {
                "rank": args.rank,
                "assigned": len(assigned),
                "pending": len(pending),
                "model_variant": args.model_variant,
                "device": str(device),
                "swa_layers": plan.sliding_window_layers,
                "trainable_tensors": trainable_tensors,
                "qcal_modules": qcal_modules,
                "qcal_scale": args.qcal_scale,
                "prompt_capacity": max_prompt_tokens,
            }
        ),
        flush=True,
    )

    for ordinal, index in enumerate(pending, start=1):
        example = examples[index]
        prompt_ids, prompt_metadata = tokenize_prompt(
            tokenizer,
            template,
            example,
            max_prompt_tokens,
        )
        layout = build_plain_generation_layout(
            prompt_ids=prompt_ids,
            answer_tokens=args.answer_tokens,
            physical_length=args.physical_length,
            mask_token_id=int(tokenizer.mask_token_id),
            pad_token_id=int(pad_token_id),
        )
        blocks = build_plain_fastdllm_block_layouts(
            layout,
            args.block_length,
        )
        input_ids = layout.input_ids.unsqueeze(0).to(device)
        attention_mask = layout.attention_mask.unsqueeze(0).to(device)
        position_ids = layout.position_ids.unsqueeze(0).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        generated, generation_stats = decoder.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            blocks=blocks,
            landmark_positions=layout.landmark_positions.to(device),
        )
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        answer_ids = generated[0, layout.answer_positions.to(device)].tolist()
        prediction = decode_answer(tokenizer, answer_ids)
        fallback_count = _kernel_fallback_count(model)
        if fallback_count:
            raise RuntimeError(f"kernel fallback count became {fallback_count}")
        expected_prefills = args.answer_tokens // args.block_length
        if (
            generation_stats.full_prefills != expected_prefills
            or generation_stats.cached_forwards < expected_prefills
        ):
            raise RuntimeError(f"unexpected generation stats: {generation_stats}")
        record = make_record(
            args=args,
            rank=args.rank,
            index=index,
            example=example,
            prediction=prediction,
            prompt_metadata=prompt_metadata,
            seconds=seconds,
            generation_stats=generation_stats,
            fallback_count=fallback_count,
        )
        record["raw_answer_ids"] = answer_ids
        record["raw_decoded"] = tokenizer.decode(
            answer_ids, skip_special_tokens=False
        )
        record["eos_offset"] = (
            answer_ids.index(tokenizer.eos_token_id)
            if tokenizer.eos_token_id in answer_ids
            else None
        )
        append_jsonl_fsync(output_path, record)
        print(
            json.dumps(
                {
                    "rank": args.rank,
                    "progress": f"{ordinal}/{len(pending)}",
                    "index": index,
                    "score": record["score"],
                    "seconds": seconds,
                    "full_prefills": generation_stats.full_prefills,
                    "cached_forwards": generation_stats.cached_forwards,
                    "routing_calls": generation_stats.routing_calls,
                    "prediction": prediction,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    shard_records = [
        record
        for record in load_resumable_jsonl(output_path)
        if int(record["index"]) in set(assigned)
    ]
    summary = {
        "rank": args.rank,
        "assigned": len(assigned),
        "completed": len(shard_records),
        "model_variant": args.model_variant,
        "score_avg": (
            100.0 * sum(float(record["score"]) for record in shard_records) / len(shard_records)
            if shard_records
            else 0.0
        ),
        "metric": next(iter({str(record.get("metric", "qa_f1")) for record in shard_records}), "unknown"),
    }
    (output_dir / f"rank-{args.rank}.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
