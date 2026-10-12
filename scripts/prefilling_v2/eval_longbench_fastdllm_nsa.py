#!/usr/bin/env python3
"""Evaluate trained Dream NSA with Fast-dLLM v1 blockwise DualCache."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from prefilling_v2.longbench_eval import (
    build_fastdllm_block_layouts,
    build_generation_layout,
    load_resumable_jsonl,
    longbench_score,
    merge_evaluation_shards,
    shard_indices,
)
from prefilling_v2.fastdllm_v1 import DreamRouteFastDLLM
from prefilling_v2.train_fulltext import (
    _build_model_and_tokenizer,
    _kernel_fallback_count,
    _set_seed,
    parse_args as parse_training_args,
    resolve_landmark_token_id,
)
from scripts.prefilling_v2.eval_longbench_mfen import (
    append_jsonl, decode_answer, load_trainables, read_jsonl, tokenize_prompt,
)


DEFAULT_DATA_ROOT = (
    "data/LongBench/data"
)
DEFAULT_TASK = "multifieldqa_en"
DEFAULT_PROMPTS = (
    "data/LongBench/config/"
    "dataset2prompt_raw.json"
)
DEFAULT_CHECKPOINT = "outputs/dream-route-dolma3-8k-pilot/step-500"
DEFAULT_OUTPUT = "outputs/dream-route-dolma3-8k-pilot/longbench_mfen_exact"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training_config",
        default="configs/prefilling_v2/dolma3_8k_dual_gpu.json",
    )
    parser.add_argument(
        "--model_path",
        help="Override the model path recorded in the training config.",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--data")
    parser.add_argument("--prompt_config", default=DEFAULT_PROMPTS)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical_length", type=int, default=8192)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--answer_tokens", type=int, default=64)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--cache_mode", choices=("dual_block", "no_cache_block"), default="dual_block")
    parser.add_argument(
        "--bootstrap",
        choices=("first_token", "confidence"),
        default="confidence",
        help="Initial token-transfer policy for each Fast-dLLM block.",
    )
    parser.add_argument(
        "--route_query_source",
        choices=("per_position", "question_mean"),
        default="per_position",
        help="Retained for evaluator CLI compatibility; NSA always uses per-position queries.",
    )
    parser.add_argument(
        "--position_encoding",
        choices=("trained", "zero"),
        default="trained",
        help="NSA evaluation requires the positions used at training time.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--indices_file",
        type=Path,
        help="Optional JSON list of dataset indices to replay.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum examples for this rank; zero evaluates its complete shard.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Validate and merge complete rank shards instead of evaluating.",
    )
    return parser




def main() -> None:
    args = build_parser().parse_args()
    if args.physical_length % args.chunk_size:
        raise ValueError("physical_length must be divisible by chunk_size")
    if args.answer_tokens <= 0:
        raise ValueError("answer_tokens must be positive")
    if args.block_length <= 0 or args.answer_tokens % args.block_length:
        raise ValueError("answer_tokens must be divisible by block_length")
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    if args.position_encoding != "trained" or args.route_query_source != "per_position":
        raise ValueError("NSA cached evaluation requires trained positions and per-position queries")
    if not args.merge and args.rank is None:
        raise ValueError("--rank is required unless --merge is used")

    task = str(args.task)
    data_path = (
        Path(args.data)
        if args.data
        else Path(DEFAULT_DATA_ROOT) / f"{task}.jsonl"
    )
    examples = read_jsonl(data_path)
    selected_indices = None
    if args.indices_file is not None:
        selected_indices = [
            int(index)
            for index in json.loads(args.indices_file.read_text(encoding="utf-8"))
        ]
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError("indices_file contains duplicates")
        if any(index < 0 or index >= len(examples) for index in selected_indices):
            raise ValueError("indices_file contains an out-of-range index")
        selected_indices.sort()
        if args.merge:
            raise ValueError("subset replay does not support --merge")
    training_argv = [
        "--config",
        args.training_config,
        "--no_gradient_checkpointing",
    ]
    if args.model_path:
        training_argv.extend(["--model_path", args.model_path])
    training_args = parse_training_args(training_argv)
    if training_args.attention_mode != "nsa" or training_args.nsa_backend != "tilelang":
        raise ValueError("this evaluator requires Dream NSA with TileLang selected attention")
    if training_args.max_length != args.physical_length:
        raise ValueError(
            "training max_length does not match evaluation physical_length"
        )
    if training_args.chunk_size != args.chunk_size:
        raise ValueError("training and evaluation chunk sizes differ")
    model_variant = (
        f"dream_nsa_paper_l{getattr(training_args, 'nsa_compress_block', 32)}"
        f"d{getattr(training_args, 'nsa_compress_stride', 16)}"
        f"_n{training_args.nsa_block_count}_nolmk_{args.physical_length}"
    )
    model_variant += (
        f"_fastdllm_v1_{args.cache_mode}_{args.bootstrap}_"
        f"b{args.block_length}_t{args.threshold:g}"
    )
    expected_examples = (
        len(selected_indices) if selected_indices is not None else len(examples)
    )
    if args.limit > 0:
        expected_examples = min(expected_examples, args.limit * args.world_size)

    if args.merge:
        metrics = merge_evaluation_shards(
            args.output_dir,
            total_examples=expected_examples,
            expected_variant=model_variant,
            require_nonempty_predictions=False,
        )
        records = load_resumable_jsonl(Path(args.output_dir) / "merged.jsonl")
        metric = next(
            iter({str(record.get("metric", "qa_f1")) for record in records}),
            "unknown",
        )
        metrics["task"] = task
        metrics["metric"] = metric
        score_avg = float(metrics.pop("qa_f1"))
        metrics["score_avg"] = score_avg
        metrics[metric] = score_avg
        metrics["dataset_examples"] = len(examples)
        metrics["limit_per_rank"] = args.limit
        Path(args.output_dir, "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, sort_keys=True), flush=True)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rank-{args.rank}.jsonl"
    summary_path = output_dir / f"rank-{args.rank}.summary.json"
    existing = load_resumable_jsonl(output_path)
    for record in existing:
        if record.get("model_variant") != model_variant:
            raise ValueError("resume record uses a different decoding configuration")
    completed = {int(record["index"]) for record in existing}

    assigned = (
        selected_indices[args.rank :: args.world_size]
        if selected_indices is not None
        else shard_indices(len(examples), args.rank, args.world_size)
    )
    if args.limit > 0:
        assigned = assigned[: args.limit]
    pending = [index for index in assigned if index not in completed]
    prompt_templates = json.loads(
        Path(args.prompt_config).read_text(encoding="utf-8")
    )
    if task not in prompt_templates:
        raise KeyError(f"missing LongBench prompt template for task={task}")
    template = prompt_templates[task]

    _set_seed(args.seed + args.rank)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, tokenizer, plan = _build_model_and_tokenizer(training_args, device)
    trainable_tensors = load_trainables(model, Path(args.checkpoint))
    model.eval()
    decoder = DreamRouteFastDLLM(
        model=model, mask_token_id=int(tokenizer.mask_token_id),
        threshold=args.threshold, use_cache=args.cache_mode == "dual_block",
        bootstrap=args.bootstrap,
    )
    landmark_token_id = resolve_landmark_token_id(training_args, tokenizer)

    real_slots = args.physical_length
    max_prompt_tokens = real_slots - args.answer_tokens
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    print(
        json.dumps(
            {
                "rank": args.rank,
                "task": task,
                "assigned": len(assigned),
                "completed": len(completed & set(assigned)),
                "pending": len(pending),
                "device": str(device),
                "decoding_schedule": "confidence_threshold",
                "bootstrap": args.bootstrap,
                "max_nfe": args.answer_tokens,
                "answer_tokens": args.answer_tokens,
                "landmark_token_id": landmark_token_id,
                "lmk_token_mode": training_args.lmk_token_mode,
                "fast_dllm": True,
                "use_cache": args.cache_mode == "dual_block",
                "model_variant": model_variant,
                "cache_mode": args.cache_mode,
                "block_length": args.block_length,
                "threshold": args.threshold,
                "route_query_source": args.route_query_source,
                "position_encoding": args.position_encoding,
                "nsa_layers": plan.nsa_layers,
                "nsa_block_count": training_args.nsa_block_count,
                "nsa_remote_physical_budget": training_args.nsa_block_count
                * int(getattr(training_args, "nsa_select_block", args.chunk_size)),
                "nsa_remote_text_budget": training_args.nsa_block_count
                * int(getattr(training_args, "nsa_select_block", args.chunk_size)),
                "nsa_insert_landmarks": False,
                "trainable_tensors": trainable_tensors,
            }
        ),
        flush=True,
    )

    for ordinal, index in enumerate(pending, start=1):
        example = examples[index]
        prompt_ids, prompt_metadata = tokenize_prompt(
            tokenizer, template, example, max_prompt_tokens
        )
        layout = build_generation_layout(
            prompt_ids=prompt_ids,
            answer_tokens=args.answer_tokens,
            physical_length=args.physical_length,
            chunk_size=args.chunk_size,
            mask_token_id=int(tokenizer.mask_token_id),
            pad_token_id=int(pad_token_id),
            landmark_token_id=int(landmark_token_id),
            insert_landmarks=False,
        )
        blocks = build_fastdllm_block_layouts(layout, args.block_length, args.chunk_size)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated, stats = decoder.generate(
                input_ids=layout.input_ids.unsqueeze(0).to(device),
                attention_mask=layout.attention_mask.unsqueeze(0).to(device),
                position_ids=layout.position_ids.unsqueeze(0).to(device),
                blocks=blocks,
                landmark_positions=layout.landmark_positions.to(device),
            )
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        answer_ids = generated[
            0, layout.answer_positions.to(device)
        ].tolist()
        prediction = decode_answer(tokenizer, answer_ids)
        nfe = stats.full_prefills + stats.cached_forwards
        if args.cache_mode == "dual_block" and stats.full_prefills != len(blocks):
            raise RuntimeError("DualCache must refresh exactly once per block")
        answers = [str(answer) for answer in example.get("answers", [])]
        all_classes = example.get("all_classes")
        score, metric = longbench_score(
            task,
            prediction,
            answers,
            all_classes=(
                [str(label) for label in all_classes]
                if isinstance(all_classes, list)
                else None
            ),
        )
        fallback_count = _kernel_fallback_count(model)
        if fallback_count:
            raise RuntimeError(f"kernel fallback count became {fallback_count}")
        record: dict[str, object] = {
            "task": task,
            "rank": args.rank,
            "index": index,
            "example_id": example.get("_id", index),
            "question": example.get("input", ""),
            "prediction": prediction,
            "raw_answer_ids": answer_ids,
            "raw_decoded": tokenizer.decode(
                answer_ids, skip_special_tokens=False
            ),
            "eos_offset": (
                answer_ids.index(tokenizer.eos_token_id)
                if tokenizer.eos_token_id in answer_ids
                else None
            ),
            "answers": answers,
            "score": score,
            "metric": metric,
            "length": example.get("length"),
            "prompt_metadata": prompt_metadata,
            "physical_length": args.physical_length,
            "answer_tokens": args.answer_tokens,
            "nfe": nfe,
            "seconds": seconds,
            "fallback_count": fallback_count,
            "fast_dllm": True,
            "use_cache": args.cache_mode == "dual_block",
            "model_variant": model_variant,
            "cache_mode": args.cache_mode,
            "block_length": args.block_length,
            "threshold": args.threshold,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "training_config": str(Path(args.training_config).resolve()),
            "route_query_source": args.route_query_source,
            "position_encoding": args.position_encoding,
            "full_prefills": stats.full_prefills,
            "cached_forwards": stats.cached_forwards,
            "routing_calls": stats.routing_calls,
            "recomputed_tokens": stats.recomputed_tokens,
            "peak_memory_bytes": stats.peak_memory_bytes,
        }
        append_jsonl(output_path, record)
        print(
            json.dumps(
                {
                    "rank": args.rank,
                    "progress": f"{ordinal}/{len(pending)}",
                    "index": index,
                    "score": record["score"],
                    "seconds": seconds,
                    "prediction": prediction,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    records = load_resumable_jsonl(output_path)
    shard_records = [
        record for record in records if int(record["index"]) in set(assigned)
    ]
    summary = {
        "rank": args.rank,
        "task": task,
        "world_size": args.world_size,
        "assigned": len(assigned),
        "completed": len(shard_records),
        "score": 100
        * sum(float(record["score"]) for record in shard_records)
        / len(shard_records),
        "mean_seconds": sum(float(record["seconds"]) for record in shard_records)
        / len(shard_records),
        "fallback_max": max(
            int(record["fallback_count"]) for record in shard_records
        ),
        "fast_dllm": True,
        "use_cache": args.cache_mode == "dual_block",
        "decoding_schedule": "confidence_threshold",
        "max_nfe": args.answer_tokens,
        "answer_tokens": args.answer_tokens,
        "model_variant": model_variant,
        "cache_mode": args.cache_mode,
        "block_length": args.block_length,
        "threshold": args.threshold,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
