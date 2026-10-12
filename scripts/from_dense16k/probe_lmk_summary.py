#!/usr/bin/env python3
"""16k S-N probes: landmark shuffle / interior shuffle / needle surgery.

Three LMK-slot models (mask_type / vocab / eos). Dumps S-N scores and a
chunk-vs-landmark cosine heatmap from the first prefill. Do not launch
from the notebook.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def _load_ruler():
    path = Path(__file__).resolve().parent / "eval_ruler.py"
    spec = importlib.util.spec_from_file_location("_ruler_eval_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _iter_route_layers(model):
    from prefilling_v2.attention import KernelDreamFullRouteAttention

    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = core.model.layers if hasattr(core, "model") else core.layers
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, KernelDreamFullRouteAttention):
            yield attn


def _cosine_chunk_lmk(landmark_keys, k_chunked, chunk_key_valid) -> np.ndarray:
    content_valid = chunk_key_valid[:, :, :-1].float()
    keys = k_chunked[:, :, :-1].float()
    num = (keys * content_valid[..., None, None]).sum(dim=2)
    den = content_valid.sum(dim=2).clamp_min(1.0)[..., None, None]
    content = (num / den)[0]
    lmk = landmark_keys.float().mean(dim=3)[0]
    c = content.reshape(content.shape[0], -1)
    l = lmk.reshape(lmk.shape[0], -1)
    c = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    l = l / l.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (c @ l.T).detach().cpu().float().numpy()


def _bin_matrix(mat: np.ndarray, bins: int = 32) -> np.ndarray:
    n = int(mat.shape[0])
    if n <= bins:
        return mat
    edges = np.linspace(0, n, bins + 1).astype(int)
    out = np.zeros((bins, bins), dtype=np.float64)
    counts = np.zeros((bins, bins), dtype=np.float64)
    for i in range(bins):
        for j in range(bins):
            block = mat[edges[i] : edges[i + 1], edges[j] : edges[j + 1]]
            if block.size:
                out[i, j] = float(block.mean())
                counts[i, j] = 1.0
    return out


def _pack(ruler, tokenizer, synthesizer, text_slots, index, block_length):
    task = "ruler_sn"
    base = ruler.haystack_token_ids(tokenizer, {}, text_slots)
    vocab = max(int(getattr(tokenizer, "vocab_size", 1) or 1), 1)
    base[0] = (int(base[0]) + int(index) * 1315423911) % vocab
    clean, target_mask, evidence = synthesizer.synthesize_with_evidence(
        torch.as_tensor(base, dtype=torch.long),
        task_id=0,
    )
    answer_idx = target_mask.nonzero(as_tuple=False).flatten()
    answer_start = int(answer_idx[0].item())
    prompt_ids = clean[:answer_start].tolist()
    gold_ids = clean[answer_start:].tolist()
    eos_id = tokenizer.eos_token_id
    gold_decode = list(gold_ids)
    if eos_id in gold_decode:
        gold_decode = gold_decode[: gold_decode.index(eos_id)]
    gold_text = tokenizer.decode(gold_decode, skip_special_tokens=True).strip()
    outputs = ruler.gold_output_list(task, gold_text)
    answer_tokens = ruler.padded_answer_tokens(len(gold_ids), block_length)
    ev = [
        int(i)
        for i in evidence.nonzero(as_tuple=False).flatten().tolist()
        if int(i) < answer_start
    ]
    return prompt_ids, answer_tokens, outputs, ev


def _move_needle(prompt_ids: list[int], ev: list[int]) -> tuple[list[int], int, int]:
    if not ev:
        return list(prompt_ids), -1, -1
    lo, hi = min(ev), max(ev) + 1
    span = prompt_ids[lo:hi]
    n = len(span)
    dest = 8
    if dest + n > lo:
        dest = max(1, lo // 4)
    if dest + n > lo:
        return list(prompt_ids), lo, lo
    out = list(prompt_ids)
    other = out[dest : dest + n]
    out[dest : dest + n] = span
    out[lo:hi] = other
    return out, lo, dest


def _shuffle_interior(ids: torch.Tensor, answer_pos, landmark_pos, chunk_size: int, seed: int):
    out = ids.clone()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    banned = set(int(x) for x in answer_pos.tolist()) | set(int(x) for x in landmark_pos.tolist())
    length = int(out.numel())
    chunks = length // chunk_size
    for chunk in range(chunks):
        slots = [
            chunk * chunk_size + off
            for off in range(chunk_size - 1)
            if (chunk * chunk_size + off) not in banned
        ]
        if len(slots) < 2:
            continue
        perm = torch.randperm(len(slots), generator=gen)
        vals = out[slots].clone()
        out[slots] = vals[perm]
    return out


def _install_intervene(model, kind: str, n_chunks: int, seed: int):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 17)
    perm = torch.randperm(n_chunks, generator=gen)

    def shuffle_fn(landmark_keys, prior_bias):
        idx = perm.to(device=landmark_keys.device)
        return landmark_keys[:, idx], prior_bias[:, idx]

    def zero_fn(landmark_keys, prior_bias):
        return torch.zeros_like(landmark_keys), torch.zeros_like(prior_bias)

    fn = {"shuffle_lmk": shuffle_fn, "zero_lmk": zero_fn}.get(kind)
    for attn in _iter_route_layers(model):
        attn._lmk_keys_intervene = fn
    return perm.tolist() if kind == "shuffle_lmk" else None


def _clear_intervene(model):
    for attn in _iter_route_layers(model):
        attn._lmk_keys_intervene = None
        attn._lmk_summary_hook = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("mask", "vocab", "eos"))
    parser.add_argument(
        "--condition",
        required=True,
        choices=("clean", "shuffle_lmk", "shuffle_interior", "surgery"),
    )
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_len", type=int, default=16384)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--heat_bins", type=int, default=32)
    args = parser.parse_args()

    ruler = _load_ruler()
    from prefilling_v2.fastdllm_v1 import DreamRouteFastDLLM
    from dataclasses import replace

    from prefilling_v2.longbench_eval import (
        build_fastdllm_block_layouts,
        build_generation_layout,
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

    os.environ.pop("RULER_KEEP_TRAIN_YARN", None)
    training_args = parse_training_args(
        ["--config", args.training_config, "--no_gradient_checkpointing"]
    )
    ruler.apply_yarn(training_args, args.max_seq_len)
    if training_args.attention_mode != "route":
        raise SystemExit("this probe is route-only")
    _set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, tokenizer, plan = _build_model_and_tokenizer(training_args, device)
    configure_eval_trainables(model, training_args)
    n_train = load_trainables(model, Path(args.checkpoint))
    model.eval()
    lmk_id = resolve_landmark_token_id(training_args, tokenizer, model)
    decoder = DreamRouteFastDLLM(
        model=model,
        mask_token_id=int(tokenizer.mask_token_id),
        threshold=args.threshold,
        use_cache=True,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    chunk_size = int(training_args.chunk_size)
    slots = ruler.text_slot_budget(
        args.max_seq_len, is_route=True, chunk_size=chunk_size
    )
    n_chunks = args.max_seq_len // chunk_size
    synthesizer = RulerDenoisingSynthesizer(tokenizer, task_ids=(0,))
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "rank-0.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    heat_acc = None
    heat_n = 0
    preds = []
    refs = []
    print(
        json.dumps(
            {
                "arm": args.arm,
                "condition": args.condition,
                "lmk_token_mode": training_args.lmk_token_mode,
                "lmk_token_id": int(lmk_id),
                "trainable_tensors": int(n_train),
                "route_layers": getattr(plan, "route_layers", None),
                "n_chunks": n_chunks,
            }
        ),
        flush=True,
    )

    for index in range(int(args.num_samples)):
        prompt_ids, answer_tokens, outputs, ev = _pack(
            ruler, tokenizer, synthesizer, slots, index, args.block_length
        )
        src_lo = min(ev) if ev else -1
        dest_lo = src_lo
        if args.condition == "surgery":
            prompt_ids, src_lo, dest_lo = _move_needle(prompt_ids, ev)
        layout = build_generation_layout(
            prompt_ids=prompt_ids,
            answer_tokens=answer_tokens,
            physical_length=args.max_seq_len,
            chunk_size=chunk_size,
            mask_token_id=int(tokenizer.mask_token_id),
            pad_token_id=int(pad_token_id),
            landmark_token_id=int(lmk_id),
        )
        if args.condition == "shuffle_interior":
            layout = replace(
                layout,
                input_ids=_shuffle_interior(
                    layout.input_ids,
                    layout.answer_positions,
                    layout.landmark_positions,
                    chunk_size,
                    seed=args.seed + 1009 * index,
                ),
            )
        perm = None
        if args.condition == "shuffle_lmk":
            perm = _install_intervene(
                model, "shuffle_lmk", n_chunks, seed=args.seed + 17 * index
            )
        layer_mats: list[np.ndarray] = []

        def _hook(*, landmark_keys, k_chunked, chunk_key_valid):
            layer_mats.append(
                _cosine_chunk_lmk(landmark_keys, k_chunked, chunk_key_valid)
            )

        for attn in _iter_route_layers(model):
            attn._lmk_summary_hook = _hook if args.condition in {"clean", "shuffle_lmk"} else None
        blocks = build_fastdllm_block_layouts(
            layout,
            logical_block_size=int(answer_tokens),
            chunk_size=chunk_size,
        )
        input_ids = layout.input_ids.unsqueeze(0).to(device)
        attention_mask = layout.attention_mask.unsqueeze(0).to(device)
        position_ids = layout.position_ids.unsqueeze(0).to(device)
        landmark_positions = layout.landmark_positions.to(device)
        t0 = time.perf_counter()
        generated, stats = decoder.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            blocks=blocks,
            landmark_positions=landmark_positions,
        )
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - t0
        if _kernel_fallback_count(model):
            raise RuntimeError("kernel fallback during lmk summary probe")
        _clear_intervene(model)
        prediction = decode_answer(
            tokenizer, generated[0, layout.answer_positions.to(device)].tolist()
        )
        preds.append(prediction)
        refs.append(outputs)
        if layer_mats:
            mat = np.mean(np.stack(layer_mats, axis=0), axis=0)
            binned = _bin_matrix(mat, args.heat_bins)
            heat_acc = binned if heat_acc is None else heat_acc + binned
            heat_n += 1
        rec = {
            "index": index,
            "prediction": prediction,
            "outputs": outputs,
            "seconds": seconds,
            "src_lo": src_lo,
            "dest_lo": dest_lo,
            "perm_head": None if perm is None else perm[:8],
            "full_prefills": stats.full_prefills,
            "cached_forwards": stats.cached_forwards,
        }
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    "progress": f"{index + 1}/{args.num_samples}",
                    "seconds": seconds,
                    "prediction": prediction[:80],
                }
            ),
            flush=True,
        )

    score = ruler.string_match_all(preds, refs)
    metrics = {
        "arm": args.arm,
        "condition": args.condition,
        "task": "ruler_sn",
        "max_seq_len": args.max_seq_len,
        "n": int(args.num_samples),
        "score": score,
        "metric": "string_match_all",
        "lmk_token_mode": training_args.lmk_token_mode,
    }
    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if heat_acc is not None and heat_n:
        mean = heat_acc / float(heat_n)
        np.save(out_root / "heat_mean.npy", mean)
        metrics["heat_n"] = heat_n
        (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
