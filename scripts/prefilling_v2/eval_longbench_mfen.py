#!/usr/bin/env python3
"""Run resumable exact sparse-attention evaluation on LongBench v1."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from prefilling_v2.longbench_eval import (
    GenerationLayout,
    build_generation_layout,
    build_plain_generation_layout,
    load_resumable_jsonl,
    longbench_score,
    physical_text_positions,
    merge_evaluation_shards,
    shard_indices,
    truncate_prompt_parts,
)
from prefilling_v2.train_fulltext import (
    _build_model_and_tokenizer,
    _kernel_fallback_count,
    _landmark_inputs_embeds,
    _set_seed,
    parse_args as parse_training_args,
    resolve_landmark_token_id,
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
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument(
        "--route_query_source",
        choices=("per_position", "question_mean"),
        default="per_position",
        help="Use each token Q or pooled LongBench question Q for route chunk routing.",
    )
    parser.add_argument(
        "--position_encoding",
        choices=("trained", "zero"),
        default="trained",
        help="Use trained position_ids or zero all position_ids for a no-position eval ablation.",
    )
    parser.add_argument("--seed", type=int, default=7)
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


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def configure_eval_trainables(
    model: torch.nn.Module, training_args: argparse.Namespace
) -> None:
    """Match the eval loader's requires_grad view to the training scope.

    Q-Cal-only checkpoints intentionally contain only the Q-Cal parameters;
    the other LoRA weights are frozen during training and must not be treated
    as missing checkpoint entries during evaluation.
    """
    if str(getattr(training_args, "route_trainable_scope", "full")) in {
        "qcal_only",
        "lora_qcal",
        "qcal_lmk",
        "lora_qcal_lmk",
    }:
        from prefilling_v2.train_fulltext import initialize_and_configure_trainables
        initialize_and_configure_trainables(model, training_args)


def load_trainables(model: torch.nn.Module, checkpoint: Path) -> int:
    saved = torch.load(
        checkpoint / "trainable_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    current = dict(model.named_parameters())
    missing = sorted(set(saved) - set(current))
    if missing:
        raise ValueError(f"checkpoint trainables missing from model: {missing[:8]}")
    with torch.no_grad():
        for name, tensor in saved.items():
            parameter = current[name]
            parameter.copy_(tensor.to(parameter.device, parameter.dtype))
    return len(saved)


def prompt_parts(template: str, example: dict[str, object]) -> tuple[str, str, str]:
    sentinel = "__LONGBENCH_CONTEXT_SENTINEL__"
    rendered = template.format(context=sentinel, input=example.get("input", ""))
    if rendered.count(sentinel) != 1:
        raise ValueError("LongBench template must contain one context slot")
    prefix, query = rendered.split(sentinel)
    return prefix, str(example.get("context", "")), query


def tokenize_prompt(
    tokenizer,
    template: str,
    example: dict[str, object],
    max_prompt_tokens: int,
) -> tuple[list[int], dict[str, int | str]]:
    prefix, context, query = prompt_parts(template, example)
    encode = lambda text: tokenizer(
        text, add_special_tokens=False
    ).input_ids
    return truncate_prompt_parts(
        prefix_ids=encode(prefix),
        context_ids=encode(context),
        query_ids=encode(query),
        max_prompt_tokens=max_prompt_tokens,
        bos_token_id=tokenizer.bos_token_id,
    )


def set_route_query_positions(
    model: torch.nn.Module,
    positions: torch.Tensor | None,
) -> None:
    for module in model.modules():
        if hasattr(module, "route_query_positions"):
            module.route_query_positions = positions


def query_route_positions(
    prompt_metadata: dict[str, int | str],
    *,
    physical_length: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    start = int(prompt_metadata["query_start_token"])
    end = int(prompt_metadata["query_end_token"])
    if end <= start:
        raise ValueError("LongBench question span is empty")
    real_slots = (physical_length // chunk_size) * (chunk_size - 1)
    mapping = physical_text_positions(real_slots, chunk_size).to(device=device)
    return mapping[torch.arange(start, end, device=device, dtype=torch.long)]


def decode_answer(tokenizer, token_ids: list[int]) -> str:
    if tokenizer.eos_token_id in token_ids:
        token_ids = token_ids[: token_ids.index(tokenizer.eos_token_id)]
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    for stop in ("\n\n\n", "\n\nQuestion:", "\n\nRead the following"):
        if stop in text:
            text = text.split(stop, 1)[0].strip()
    return text


@torch.inference_mode()
def denoise_answer(
    *,
    model: torch.nn.Module,
    tokenizer,
    layout: GenerationLayout,
    landmark_token_id: int,
    steps: int,
    route_query_positions: torch.Tensor | None = None,
    position_encoding: str = "trained",
) -> tuple[str, int, float]:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    device = next(model.parameters()).device
    input_ids = layout.input_ids.unsqueeze(0).to(device)
    attention_mask = layout.attention_mask.unsqueeze(0).to(device)
    position_ids = layout.position_ids.unsqueeze(0).to(device)
    if position_encoding == "zero":
        position_ids = torch.zeros_like(position_ids)
    answer_positions = layout.answer_positions.to(device)
    predictor_positions = layout.predictor_positions.to(device)
    landmark_positions = layout.landmark_positions.to(device)
    remaining = torch.ones(
        answer_positions.numel(), device=device, dtype=torch.bool
    )

    started = time.perf_counter()
    completed_steps = 0
    for step in range(steps):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model_input_ids, inputs_embeds = _landmark_inputs_embeds(
                model,
                input_ids,
                input_ids.eq(int(landmark_token_id)),
            )
            set_route_query_positions(model, route_query_positions)
            try:
                outputs = model(
                    input_ids=model_input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
            finally:
                set_route_query_positions(model, None)
        candidate_logits = outputs.logits[0, predictor_positions].float()
        candidate_logits[:, int(tokenizer.mask_token_id)] = float("-inf")
        active = torch.where(remaining)[0]
        probabilities = torch.softmax(candidate_logits[active], dim=-1)
        confidence, candidate_ids = probabilities.max(dim=-1)
        transfer_count = math.ceil(active.numel() / (steps - step))
        selected_active = torch.topk(confidence, k=transfer_count).indices
        selected = active[selected_active]
        input_ids[0, answer_positions[selected]] = candidate_ids[selected_active]
        remaining[selected] = False
        completed_steps = step + 1

        if not torch.all(
            input_ids[0, landmark_positions] == int(landmark_token_id)
        ):
            raise RuntimeError("a fixed landmark token was modified")
        if not remaining.any():
            break

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if remaining.any():
        raise RuntimeError(f"{int(remaining.sum())} answer masks remain")
    generated = input_ids[0, answer_positions].tolist()
    return decode_answer(tokenizer, generated), completed_steps, elapsed


def main() -> None:
    args = build_parser().parse_args()
    if args.physical_length <= 0:
        raise ValueError("physical_length must be positive")
    if args.answer_tokens <= 0 or args.steps <= 0:
        raise ValueError("answer_tokens and steps must be positive")
    if not args.merge and args.rank is None:
        raise ValueError("--rank is required unless --merge is used")

    task = str(args.task)
    data_path = (
        Path(args.data)
        if args.data
        else Path(DEFAULT_DATA_ROOT) / f"{task}.jsonl"
    )
    examples = read_jsonl(data_path)
    training_argv = [
        "--config",
        args.training_config,
        "--no_gradient_checkpointing",
    ]
    if args.model_path:
        training_argv.extend(["--model_path", args.model_path])
    training_args = parse_training_args(training_argv)
    if training_args.max_length != args.physical_length:
        raise ValueError(
            "training max_length does not match evaluation physical_length"
        )
    if training_args.attention_mode != "dsa" and training_args.chunk_size != args.chunk_size:
        raise ValueError("training and evaluation chunk sizes differ")
    if training_args.attention_mode != "dsa" and args.physical_length % args.chunk_size:
        raise ValueError("physical_length must be divisible by chunk_size")
    if (
        training_args.attention_mode == "route"
        and training_args.route_token_budget > 0
    ):
        model_variant = (
            f"dream_route_{training_args.route_token_policy}_"
            f"{training_args.route_token_budget}_{args.physical_length}"
        )
    elif training_args.attention_mode == "dsa":
        model_variant = (
            f"dream_dsa_topk{training_args.dsa_topk}_"
            f"{args.physical_length}"
        )
    else:
        model_variant = (
            f"dream_{training_args.attention_mode}_{args.physical_length}"
        )
    expected_examples = len(examples)
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
    completed = {int(record["index"]) for record in existing}

    assigned = shard_indices(len(examples), args.rank, args.world_size)
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
    configure_eval_trainables(model, training_args)
    trainable_tensors = load_trainables(model, Path(args.checkpoint))
    model.eval()
    landmark_token_id = resolve_landmark_token_id(training_args, tokenizer)

    if training_args.attention_mode == "dsa":
        max_prompt_tokens = args.physical_length - args.answer_tokens
    else:
        real_slots = (args.physical_length // args.chunk_size) * (
            args.chunk_size - 1
        )
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
                "steps": args.steps,
                "answer_tokens": args.answer_tokens,
                "landmark_token_id": landmark_token_id,
                "lmk_token_mode": training_args.lmk_token_mode,
                "fast_dllm": False,
                "use_cache": False,
                "model_variant": model_variant,
                "cache_mode": "no_cache_exact",
                "route_query_source": args.route_query_source,
                "position_encoding": args.position_encoding,
                "route_layers": plan.route_layers,
                "dsa_layers": plan.dsa_layers,
                "dsa_topk": training_args.dsa_topk,
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
        if training_args.attention_mode == "dsa":
            layout = build_plain_generation_layout(
                prompt_ids=prompt_ids,
                answer_tokens=args.answer_tokens,
                physical_length=args.physical_length,
                mask_token_id=int(tokenizer.mask_token_id),
                pad_token_id=int(pad_token_id),
            )
        else:
            layout = build_generation_layout(
                prompt_ids=prompt_ids,
                answer_tokens=args.answer_tokens,
                physical_length=args.physical_length,
                chunk_size=args.chunk_size,
                mask_token_id=int(tokenizer.mask_token_id),
                pad_token_id=int(pad_token_id),
                landmark_token_id=int(landmark_token_id),
            )
        route_positions = None
        if args.route_query_source == "question_mean":
            route_positions = query_route_positions(
                prompt_metadata,
                physical_length=args.physical_length,
                chunk_size=args.chunk_size,
                device=device,
            )
        prediction, nfe, seconds = denoise_answer(
            model=model,
            tokenizer=tokenizer,
            layout=layout,
            landmark_token_id=int(landmark_token_id),
            steps=args.steps,
            route_query_positions=route_positions,
            position_encoding=args.position_encoding,
        )
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
            "fast_dllm": False,
            "use_cache": False,
            "model_variant": model_variant,
            "cache_mode": "no_cache_exact",
            "route_query_source": args.route_query_source,
            "position_encoding": args.position_encoding,
            "full_prefills": nfe,
            "cached_forwards": 0,
            "routing_calls": nfe if plan.route_layers else 0,
            "recomputed_tokens": args.physical_length * nfe,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
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
        "fast_dllm": False,
        "use_cache": False,
        "steps": args.steps,
        "answer_tokens": args.answer_tokens,
        "model_variant": model_variant,
        "cache_mode": "no_cache_exact",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
