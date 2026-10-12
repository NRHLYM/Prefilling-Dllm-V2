#!/usr/bin/env python3
"""RULER length-extrapolation probes for the Dream variants.

Table 2: S-N / MK-MQ / VT at 16k, 32k, 64k (128k optional). Checkpoints stay
at the 16k LoRA; only YaRN factor = L/2048 changes. Needle counts match
training task 0/1/2, not the stock NVIDIA 13-task YAML.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

# The three training task families, plus the stock RULER task names.
PROBE_TASKS = ["ruler_sn", "ruler_mkmq", "ruler_vt"]
ALL_TASKS = [
    *PROBE_TASKS,
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]
TASKS = PROBE_TASKS

TASK_FAMILY = {
    "ruler_sn": "niah",
    "ruler_mkmq": "niah",
    "ruler_vt": "variable_tracking",
    "niah_single_1": "niah",
    "niah_single_2": "niah",
    "niah_single_3": "niah",
    "niah_multikey_1": "niah",
    "niah_multikey_2": "niah",
    "niah_multikey_3": "niah",
    "niah_multivalue": "niah",
    "niah_multiquery": "niah",
    "vt": "variable_tracking",
    "cwe": "common_words_extraction",
    "fwe": "freq_words_extraction",
    "qa_1": "qa",
    "qa_2": "qa",
}

TOKENS_TO_GENERATE = {
    "niah": 128,
    "variable_tracking": 32,
    "common_words_extraction": 128,
    "freq_words_extraction": 64,
    "qa": 32,
}

LENGTHS = (16384, 32768, 65536, 131072)


def yarn_factor(max_seq_len: int) -> float:
    return float(max_seq_len) / 2048.0


def string_match_part(preds, refs):
    score = (
        sum(
            max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref] or [0.0])
            for pred, ref in zip(preds, refs)
        )
        / max(len(preds), 1)
        * 100
    )
    return round(score, 2)


def _move_tree(value, device):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tree(item, device) for key, item in value.items()}
    return value


def shard_dream_layers_two_gpus(model: torch.nn.Module) -> None:
    """Split embed+first-half layers onto cuda:0, rest+norm onto cuda:1.

    One process, one 128k sequence. Tied lm_head stays with embeddings.
    Prefill hooks move each module's args onto that module's device.
    """
    from prefilling_v2.fastdllm_v1 import _unwrap_dream_model

    if torch.cuda.device_count() < 2:
        raise RuntimeError("layer_parallel_gpus=2 needs two visible CUDA devices")
    core = _unwrap_dream_model(model)
    layers = core.model.layers
    mid = max(1, len(layers) // 2)
    d0 = torch.device("cuda:0")
    d1 = torch.device("cuda:1")
    embed = core.model.embed_tokens
    embed.to(d0)
    rotary = getattr(core.model, "rotary_emb", None)
    if rotary is not None:
        rotary.to(d0)
    root = model.module if hasattr(model, "module") else model
    holders = [root, core]
    get_base = getattr(root, "get_base_model", None)
    if callable(get_base):
        holders.append(get_base())
    for holder in holders:
        if holder is None:
            continue
        for name in ("dream_route_lmk_type_embed", "dream_route_lmk_embed"):
            param = getattr(holder, name, None)
            if torch.is_tensor(param):
                param.data = param.data.to(d0)
    for idx, layer in enumerate(layers):
        layer.to(d0 if idx < mid else d1)
    core.model.norm.to(d1)
    head = getattr(core, "lm_head", None)
    embed_ids = {id(p) for p in embed.parameters()}
    if head is not None and not ({id(p) for p in head.parameters()} & embed_ids):
        head.to(d1)

    def _pre_hook(module, args, kwargs):
        device = next(module.parameters()).device
        return _move_tree(args, device), _move_tree(kwargs, device)

    for layer in layers:
        layer.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    core.model.norm.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    if head is not None:
        head.register_forward_pre_hook(_pre_hook, with_kwargs=True)


def string_match_all(preds, refs):
    score = (
        sum(
            sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref)
            / max(len(ref), 1)
            for pred, ref in zip(preds, refs)
        )
        / max(len(preds), 1)
        * 100
    )
    return round(score, 2)


METRIC = {
    "niah": string_match_all,
    "variable_tracking": string_match_all,
    "common_words_extraction": string_match_all,
    "freq_words_extraction": string_match_all,
    "qa": string_match_part,
}


def ensure_ruler(ruler_dir: Path) -> Path:
    if (ruler_dir / "scripts" / "data" / "prepare.py").is_file():
        return ruler_dir
    ruler_dir.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("GIT_HTTP_VERSION", "HTTP/1.1")
    subprocess.check_call(
        [
            "git",
            "-c",
            "http.version=HTTP/1.1",
            "clone",
            "--depth",
            "1",
            "https://github.com/NVIDIA/RULER.git",
            str(ruler_dir),
        ],
        env=env,
    )
    return ruler_dir


def install_probe_yaml(ruler: Path) -> None:
    import yaml

    overlay_path = Path(__file__).with_name("ruler_probes.yaml")
    dest = ruler / "scripts" / "synthetic.yaml"
    base = yaml.safe_load(dest.read_text()) or {}
    extra = yaml.safe_load(overlay_path.read_text())
    base.update(extra)
    dest.write_text(yaml.safe_dump(base, sort_keys=False))
    print(f"installed probe yaml -> {dest} tasks={list(extra)}", flush=True)


def tasks_need_essays(tasks: list[str]) -> bool:
    return any(task not in PROBE_TASKS for task in tasks)


def ensure_synthetic_assets(ruler: Path, python: str, tasks: list[str]) -> None:
    if not tasks_need_essays(tasks):
        return
    jsondir = ruler / "scripts" / "data" / "synthetic" / "json"
    essay = jsondir / "PaulGrahamEssays.json"
    squad = jsondir / "squad.json"
    hotpot = jsondir / "hotpotqa.json"
    if not essay.is_file() or essay.stat().st_size < 100_000:
        print(f"download Paul Graham essays -> {essay}", flush=True)
        subprocess.check_call(
            [python, str(jsondir / "download_paulgraham_essay.py")],
            cwd=str(jsondir),
        )
        if not essay.is_file() or essay.stat().st_size < 100_000:
            raise RuntimeError(f"missing or tiny {essay}")
    if not squad.is_file() or not hotpot.is_file():
        print(f"download SQuAD/HotpotQA -> {jsondir}", flush=True)
        subprocess.check_call(
            ["bash", str(jsondir / "download_qa_dataset.sh")],
            cwd=str(jsondir),
        )
        if not squad.is_file() or not hotpot.is_file():
            raise RuntimeError(f"missing {squad} or {hotpot}")


def prepare_split(args: argparse.Namespace) -> None:
    ruler = ensure_ruler(Path(args.ruler_dir))
    python = sys.executable
    subprocess.check_call(
        [
            python,
            "-m",
            "pip",
            "install",
            "nltk",
            "wonderwords",
            "pyyaml",
            "tenacity",
            "beautifulsoup4",
            "html2text",
        ]
    )
    install_probe_yaml(ruler)
    tasks = list(args.tasks) if args.tasks else list(PROBE_TASKS)
    ensure_synthetic_assets(ruler, python, tasks)
    subprocess.check_call(
        [
            python,
            "-c",
            "import os, nltk\n"
            "d=os.environ.get('NLTK_DATA')\n"
            "if d:\n"
            "    nltk.data.path.insert(0, d)\n"
            "ok=True\n"
            "for res in ('tokenizers/punkt','tokenizers/punkt_tab'):\n"
            "    try:\n"
            "        nltk.data.find(res)\n"
            "    except LookupError:\n"
            "        ok=False\n"
            "        nltk.download(res.split('/')[-1], download_dir=d)\n"
            "print('nltk_ok' if ok else 'nltk_downloaded', flush=True)",
        ]
    )
    prepare = ruler / "scripts" / "data" / "prepare.py"
    tokenizer = args.model_path
    for max_seq_len in args.lengths:
        save_dir = Path(args.data_dir) / f"len{max_seq_len}"
        save_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            out = save_dir / task / "validation.jsonl"
            if out.is_file() and sum(1 for _ in out.open()) >= args.num_samples:
                print(f"skip prepare {out}", flush=True)
                continue
            cmd = [
                python,
                str(prepare),
                "--save_dir",
                str(save_dir),
                "--benchmark",
                "synthetic",
                "--task",
                task,
                "--tokenizer_path",
                tokenizer,
                "--tokenizer_type",
                "hf",
                "--max_seq_length",
                str(max_seq_len),
                "--model_template_type",
                "base",
                "--num_samples",
                str(args.num_samples),
                "--random_seed",
                "42",
            ]
            env = os.environ.copy()
            env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
            print(" ".join(cmd), flush=True)
            subprocess.check_call(cmd, cwd=str(ruler / "scripts"), env=env)
            if not out.is_file() or sum(1 for _ in out.open()) < args.num_samples:
                raise RuntimeError(
                    f"RULER prepare failed for {task} len={max_seq_len}: {out}"
                )


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def job_list(lengths: list[int], tasks: list[str] | None = None) -> list[tuple[int, str]]:
    names = tasks or list(PROBE_TASKS)
    return [(length, task) for length in lengths for task in names]


def enable_dense_fastdllm_cache(
    model: torch.nn.Module,
    *,
    local_window: int,
    chunk_size: int,
    layer_indices: list[int] | None = None,
) -> bool:
    """Wrap dense layers as full-window FA-SWA so Fast-dLLM can capture KV.

    For 3 SWA + 1 dense, pass the 7 dense slot indices. Wrapping every layer
    would replace radius-1280 SWA with a full-window kernel and void the
    ablation.
    """
    from prefilling_v2.attention import KernelDreamSlidingWindowAttention
    from prefilling_v2.fastdllm_v1 import _unwrap_dream_model

    core = _unwrap_dream_model(model)
    layers = core.model.layers
    want = None if layer_indices is None else set(int(i) for i in layer_indices)
    replaced = 0
    for idx, layer in enumerate(layers):
        if want is not None and idx not in want:
            continue
        source = getattr(layer.self_attn, "source_attn", layer.self_attn)
        layer.self_attn = KernelDreamSlidingWindowAttention(
            source,
            int(local_window),
            int(chunk_size),
            allow_fallback=False,
            skip_inert_slots=False,
        )
        replaced += 1
    if replaced == 0:
        raise RuntimeError("dense Fast-dLLM cache wrap replaced zero layers")
    return True


def apply_yarn(training_args, max_seq_len: int) -> None:
    training_args.max_length = int(max_seq_len)
    training_args.model_max_position_embeddings = int(max_seq_len)
    rope = dict(getattr(training_args, "model_rope_scaling", None) or {})
    rope_type = str(rope.get("rope_type") or rope.get("type") or "yarn")
    orig = int(rope.get("original_max_position_embeddings") or 2048)
    if rope_type == "hope":
        training_args.model_rope_scaling = {
            "rope_type": "hope",
            "original_max_position_embeddings": orig,
            "period_multiplier": float(rope.get("period_multiplier") or 1.0),
        }
        return
    keep = os.environ.get("RULER_KEEP_TRAIN_YARN", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if keep:
        factor = float(rope.get("factor") or 8.0)
    else:
        factor = yarn_factor(max_seq_len)
        orig = 2048
    override = os.environ.get("RULER_YARN_FACTOR", "").strip()
    if override:
        factor = float(override)
    training_args.model_rope_scaling = {
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": orig,
    }


TASK_SYNTH_ID = {
    "ruler_sn": 0,
    "ruler_mkmq": 1,
    "ruler_vt": 2,
}

NOISE_HAYSTACK = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again.\n"
)


def text_slot_budget(physical_length: int, *, is_route: bool, chunk_size: int) -> int:
    if is_route:
        return (physical_length // chunk_size) * (chunk_size - 1)
    return physical_length


def padded_answer_tokens(n_gold: int, block_length: int) -> int:
    """Keep the Fast-dLLM mask span equal to the gold answer (no pad to 32)."""
    del block_length
    return max(1, int(n_gold))


def haystack_token_ids(tokenizer, example: dict | None, length: int) -> list[int]:
    del example
    ids = tokenizer(NOISE_HAYSTACK, add_special_tokens=False).input_ids
    if not ids:
        ids = tokenizer(NOISE_HAYSTACK, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError("empty RULER haystack")
    return (ids * ((length // len(ids)) + 2))[:length]


def gold_output_list(task: str, gold_text: str) -> list[str]:
    text = gold_text.strip()
    if task == "ruler_mkmq":
        return [part for part in text.split() if part]
    if task == "ruler_vt":
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text] if text else []


def pack_route_style_prompt(
    *,
    tokenizer,
    synthesizer,
    task: str,
    text_slots: int,
    sample_index: int,
    example: dict | None,
    block_length: int,
) -> tuple[list[int], int, list[str]]:
    """Pack like route eval/train: shrink haystack, keep needles + query + answer at the end."""

    task_id = TASK_SYNTH_ID[task]
    base = haystack_token_ids(tokenizer, example, text_slots)
    vocab = max(int(getattr(tokenizer, "vocab_size", 1) or 1), 1)
    base[0] = (int(base[0]) + int(sample_index) * 1315423911) % vocab
    clean, target_mask, _evidence = synthesizer.synthesize_with_evidence(
        torch.as_tensor(base, dtype=torch.long),
        task_id=task_id,
    )
    answer_idx = target_mask.nonzero(as_tuple=False).flatten()
    if answer_idx.numel() <= 0:
        raise RuntimeError("RULER synthesis produced an empty answer span")
    answer_start = int(answer_idx[0].item())
    prompt_ids = clean[:answer_start].tolist()
    gold_ids = clean[answer_start:].tolist()
    eos_id = tokenizer.eos_token_id
    gold_decode = list(gold_ids)
    if eos_id in gold_decode:
        gold_decode = gold_decode[: gold_decode.index(eos_id)]
    gold_text = tokenizer.decode(gold_decode, skip_special_tokens=True).strip()
    outputs = gold_output_list(task, gold_text)
    answer_tokens = padded_answer_tokens(len(gold_ids), block_length)
    if len(prompt_ids) + answer_tokens > text_slots:
        raise ValueError(
            f"packed prompt {len(prompt_ids)} + answer {answer_tokens} "
            f"exceeds {text_slots} text slots"
        )
    return prompt_ids, answer_tokens, outputs


def eval_one(args: argparse.Namespace) -> None:
    from prefilling_v2.fastdllm_v1 import DreamRouteFastDLLM
    from prefilling_v2.longbench_eval import (
        build_fastdllm_block_layouts,
        build_generation_layout,
        build_plain_fastdllm_block_layouts,
        build_plain_generation_layout,
        shard_indices,
    )
    from prefilling_v2.train_fulltext import (
        _build_model_and_tokenizer,
        _kernel_fallback_count,
        _set_seed,
        parse_args as parse_training_args,
        resolve_landmark_token_id,
    )
    from prefilling_v2.data import RulerDenoisingSynthesizer
    from scripts.prefilling_v2.eval_longbench_mfen import (
        configure_eval_trainables,
        decode_answer,
        load_trainables,
    )

    family = TASK_FAMILY[args.task]
    if args.task not in TASK_SYNTH_ID:
        raise ValueError(f"route-style packing only supports {list(TASK_SYNTH_ID)}")
    data_path = Path(args.data_dir) / f"len{args.max_seq_len}" / args.task / "validation.jsonl"
    rows = load_jsonl(data_path) if data_path.is_file() else [{} for _ in range(100)]
    if not rows:
        rows = [{} for _ in range(100)]
    if args.limit > 0:
        rows = rows[: args.limit]
    assigned = shard_indices(len(rows), args.rank, args.world_size)
    out_dir = Path(args.output_dir) / f"len{args.max_seq_len}" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"rank-{args.rank}.jsonl"
    done = {int(r["index"]) for r in load_jsonl(shard_path) if "index" in r}
    pending = [i for i in assigned if i not in done]
    if not pending:
        print(json.dumps({"rank": args.rank, "pending": 0, "task": args.task}), flush=True)
        return

    training_args = parse_training_args(
        ["--config", args.training_config, "--no_gradient_checkpointing"]
    )
    apply_yarn(training_args, args.max_seq_len)
    if args.eval_trainable_scope is not None:
        training_args.route_trainable_scope = args.eval_trainable_scope
    is_route = training_args.attention_mode == "route"
    _set_seed(args.seed + args.rank)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, tokenizer, plan = _build_model_and_tokenizer(training_args, device)
    configure_eval_trainables(model, training_args)
    skip_ckpt = str(args.checkpoint or "").strip().lower() in {"", "-", "none", "vanilla"}
    if skip_ckpt:
        if int(getattr(training_args, "lora_r", 0) or 0) > 0:
            raise SystemExit("vanilla/base eval needs lora_r=0 and no LoRA checkpoint")
        trainable_tensors = 0
        print(json.dumps({"vanilla_base": True, "trainable_tensors": 0}), flush=True)
    else:
        trainable_tensors = load_trainables(model, Path(args.checkpoint))
    if int(getattr(args, "layer_parallel_gpus", 1) or 1) >= 2:
        shard_dream_layers_two_gpus(model)
    model.eval()
    landmark_token_id = (
        int(resolve_landmark_token_id(training_args, tokenizer, model))
        if is_route
        else int(tokenizer.mask_token_id)
    )
    dense = str(training_args.attention_mode) == "dense"
    dense_cache = False
    wrap_idx = None
    if dense:
        sliding = list(getattr(plan, "sliding_window_layers", []) or [])
        dense_slots = list(getattr(plan, "dense_layers", []) or [])
        wrap_idx = dense_slots if sliding and dense_slots else None
        dense_cache = enable_dense_fastdllm_cache(
            model,
            local_window=int(args.max_seq_len),
            chunk_size=int(training_args.chunk_size),
            layer_indices=wrap_idx,
        )
    decoder = DreamRouteFastDLLM(
        model=model,
        mask_token_id=int(tokenizer.mask_token_id),
        threshold=args.threshold,
        bootstrap=args.bootstrap,
        use_cache=True,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    slots = text_slot_budget(
        args.max_seq_len,
        is_route=is_route,
        chunk_size=int(training_args.chunk_size),
    )
    synthesizer = RulerDenoisingSynthesizer(
        tokenizer, task_ids=(TASK_SYNTH_ID[args.task],)
    )
    print(
        json.dumps(
            {
                "rank": args.rank,
                "task": args.task,
                "max_seq_len": args.max_seq_len,
                "rope_type": (training_args.model_rope_scaling or {}).get(
                    "rope_type"
                ),
                "yarn_factor": (
                    (training_args.model_rope_scaling or {}).get("factor")
                ),
                "attention_mode": training_args.attention_mode,
                "lmk_token_mode": getattr(training_args, "lmk_token_mode", None),
                "lmk_token_id": landmark_token_id,
                "pending": len(pending),
                "trainable_tensors": trainable_tensors,
                "route_layers": getattr(plan, "route_layers", None),
                "sliding_window_layers": getattr(plan, "sliding_window_layers", None),
                "dense_layers": getattr(plan, "dense_layers", None),
                "dense_cache_wrap_layers": wrap_idx if dense else None,
                "text_slots": slots,
                "pack": "route_compose_goldspan",
                "answer_mask": "gold_tokens",
                "dense_fastdllm_cache": dense_cache,
                "device": str(device),
            }
        ),
        flush=True,
    )

    for ordinal, index in enumerate(pending, start=1):
        example = rows[index] if index < len(rows) else {}
        prompt_ids, answer_tokens, outputs = pack_route_style_prompt(
            tokenizer=tokenizer,
            synthesizer=synthesizer,
            task=args.task,
            text_slots=slots,
            sample_index=index,
            example=example,
            block_length=args.block_length,
        )
        logical_block_size = int(answer_tokens)
        if is_route:
            layout = build_generation_layout(
                prompt_ids=prompt_ids,
                answer_tokens=answer_tokens,
                physical_length=args.max_seq_len,
                chunk_size=int(training_args.chunk_size),
                mask_token_id=int(tokenizer.mask_token_id),
                pad_token_id=int(pad_token_id),
                landmark_token_id=landmark_token_id,
            )
            blocks = build_fastdllm_block_layouts(
                layout,
                logical_block_size=logical_block_size,
                chunk_size=int(training_args.chunk_size),
            )
            landmark_positions = layout.landmark_positions.to(device)
        else:
            layout = build_plain_generation_layout(
                prompt_ids=prompt_ids,
                answer_tokens=answer_tokens,
                physical_length=args.max_seq_len,
                mask_token_id=int(tokenizer.mask_token_id),
                pad_token_id=int(pad_token_id),
            )
            blocks = build_plain_fastdllm_block_layouts(
                layout, logical_block_size=logical_block_size
            )
            landmark_positions = None
        input_ids = layout.input_ids.unsqueeze(0).to(device)
        attention_mask = layout.attention_mask.unsqueeze(0).to(device)
        position_ids = layout.position_ids.unsqueeze(0).to(device)
        started = time.perf_counter()
        generated, generation_stats = decoder.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            blocks=blocks,
            landmark_positions=landmark_positions,
        )
        if int(getattr(args, "layer_parallel_gpus", 1) or 1) >= 2:
            for idx in range(torch.cuda.device_count()):
                torch.cuda.synchronize(idx)
        else:
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        fallback_count = _kernel_fallback_count(model)
        if fallback_count:
            raise RuntimeError(f"kernel fallback count became {fallback_count}")
        answer_ids = generated[0, layout.answer_positions.to(device)].tolist()
        prediction = decode_answer(tokenizer, answer_ids)
        record = {
            "index": index,
            "task": args.task,
            "max_seq_len": args.max_seq_len,
            "prediction": prediction,
            "outputs": outputs,
            "prompt_tokens": len(prompt_ids),
            "answer_tokens": answer_tokens,
            "pack": "route_compose_goldspan",
            "seconds": seconds,
            "full_prefills": generation_stats.full_prefills,
            "cached_forwards": generation_stats.cached_forwards,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        with shard_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(
            json.dumps(
                {
                    "rank": args.rank,
                    "progress": f"{ordinal}/{len(pending)}",
                    "index": index,
                    "seconds": seconds,
                    "prediction": prediction[:120],
                }
            ),
            flush=True,
        )


def merge_one(args: argparse.Namespace) -> None:
    family = TASK_FAMILY[args.task]
    metric = METRIC[family]
    out_dir = Path(args.output_dir) / f"len{args.max_seq_len}" / args.task
    records = []
    for shard in sorted(out_dir.glob("rank-*.jsonl")):
        records.extend(load_jsonl(shard))
    by_index = {int(r["index"]): r for r in records}
    n = int(args.limit) if int(args.limit or 0) > 0 else int(args.num_samples or 100)
    missing = [i for i in range(n) if i not in by_index]
    if missing:
        raise RuntimeError(f"{args.task} len{args.max_seq_len} missing {len(missing)}: {missing[:8]}")
    preds = [by_index[i]["prediction"] for i in range(n)]
    refs = [by_index[i]["outputs"] for i in range(n)]
    if any(not ref for ref in refs):
        raise RuntimeError(f"{args.task} len{args.max_seq_len} missing synthesized gold outputs")
    score = metric(preds, refs)
    payload = {
        "task": args.task,
        "max_seq_len": args.max_seq_len,
        "n": n,
        "score": score,
        "family": family,
        "pack": by_index[0].get("pack", "route_compose_goldspan") if n else "route_compose_goldspan",
        "metric": "string_match_all",
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


def merge_all(args: argparse.Namespace) -> None:
    summary = {"model": args.model_name, "lengths": {}}
    tasks = list(args.tasks) if args.tasks else list(PROBE_TASKS)
    for length in args.lengths:
        row = {}
        scores = []
        for task in tasks:
            metrics_path = (
                Path(args.output_dir) / f"len{length}" / task / "metrics.json"
            )
            if not metrics_path.is_file():
                row[task] = None
                continue
            score = json.loads(metrics_path.read_text())["score"]
            row[task] = score
            scores.append(score)
        row["avg"] = round(sum(scores) / len(scores), 2) if scores else None
        summary["lengths"][str(length)] = row
    out = Path(args.output_dir) / "ruler_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "eval", "merge", "summary"], required=True)
    parser.add_argument("--ruler_dir", default="third_party/RULER")
    parser.add_argument("--data_dir", default="data/ruler-probes")
    parser.add_argument("--model_path", default="models/Dream-v0-Base-7B")
    parser.add_argument("--training_config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output_dir")
    parser.add_argument("--model_name", default="baseline")
    parser.add_argument("--task", choices=ALL_TASKS)
    parser.add_argument("--max_seq_len", type=int, choices=list(LENGTHS))
    parser.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    parser.add_argument("--tasks", nargs="+", default=list(PROBE_TASKS), choices=ALL_TASKS)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument(
        "--layer_parallel_gpus",
        type=int,
        default=1,
        help="2 = split layers across two visible GPUs for one sequence",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--bootstrap", default="confidence")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval_trainable_scope")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare_split(args)
        return
    if args.mode == "summary":
        merge_all(args)
        return
    if args.task is None or args.max_seq_len is None:
        raise SystemExit("eval/merge require --task and --max_seq_len")
    if args.mode == "merge":
        merge_one(args)
        return
    if not args.training_config or not args.output_dir:
        raise SystemExit("eval requires --training_config --output_dir")
    if args.checkpoint is None:
        raise SystemExit("eval requires --checkpoint (use none for vanilla Dream-base)")
    eval_one(args)


if __name__ == "__main__":
    main()
