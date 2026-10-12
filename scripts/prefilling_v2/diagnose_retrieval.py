#!/usr/bin/env python3
"""Frozen-checkpoint routing probes and cache-based causal interventions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from prefilling_v2.diagnostic_helpers import RetrievalCase, corrupt_background, diagnostic_lengths, make_case, observer_error_limit, score_codes
from prefilling_v2.longbench_eval import append_jsonl_fsync, build_fastdllm_block_layouts, build_generation_layout


def sha256(path):
    with Path(path).open("rb") as stream:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(stream, "sha256").hexdigest()
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("prepare", "probe", "generate", "validate"), required=True)
    p.add_argument("--origin", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--training_config", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--variants", nargs="+", default=["baseline"])
    p.add_argument("--lengths", nargs="+", type=int)
    p.add_argument("--ratios", nargs="+", type=float, default=[0, 0.5, 0.9])
    p.add_argument("--bootstrap", choices=("first_token", "confidence"), default="first_token")
    p.add_argument("--qcal_scale", type=float, default=1.0)
    p.add_argument("--max_probe_queries", type=int, default=16)
    return p


def prepare(args):
    from transformers import AutoTokenizer
    from prefilling_v2.packed_corpus import DreamPackedCorpus
    config = json.loads(Path(args.training_config).read_text())
    corpus = Path(args.origin) / config["corpus_bin"]
    meta = Path(args.origin) / config["corpus_meta"]
    ds = DreamPackedCorpus(corpus, meta)
    rng = np.random.default_rng(int(config["seed"]) ^ 0x5EED5EED)
    validation = sorted(rng.choice(len(ds), size=config["validation_packs"], replace=False).tolist())
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    selected, skipped = [], []
    for pack_id in validation:
        pack = ds[pack_id]
        filler = pack["clean_ids"][pack["valid_tokens"]].tolist()
        i = len(selected)
        task = ("single", "multi", "chain")[i // 6]
        depth = (0.1, 0.5, 0.9)[i % 3]
        seed = 9202609 + (i // 6) * 101 + (i % 6) // 3
        try:
            make_case(tok, filler, max(args.lengths) // 64 * 63 - 32, task=task, seed=seed, depth=depth)
        except ValueError as exc:
            if str(exc) != "not enough filler or prompt capacity":
                raise
            skipped.append({"pack_id": pack_id, "valid_tokens": len(filler)})
            continue
        selected.append((pack_id, filler))
        if len(selected) == 18:
            break
    if len(selected) != 18:
        raise ValueError("held-out split lacks 18 sufficiently long valid packs")
    rows = []
    for length in args.lengths:
        for i in range(18):
            pack_id, filler = selected[i]
            task = ("single", "multi", "chain")[i // 6]
            depth = (0.1, 0.5, 0.9)[i % 3]
            seed = 9202609 + (i // 6) * 101 + (i % 6) // 3
            case = make_case(tok, filler, length // 64 * 63 - 32, task=task, seed=seed, depth=depth)
            row = asdict(case)
            row.update(case_id=f"L{length}-case{i:02}", physical_length=length, validation_pack_id=pack_id)
            rows.append(row)
    target = Path(args.manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    if target.exists() and target.read_text() != payload:
        raise ValueError("refusing to change an existing experiment manifest")
    target.write_text(payload)
    provenance = {"cases": len(rows), "corpus_sha256": sha256(corpus), "corpus_meta_sha256": sha256(meta),
                  "manifest_sha256": sha256(target), "validation_pack_ids": [p for p, _ in selected],
                  "excluded_short_validation_packs": skipped,
                  "sampling": "first 18 sorted eligible held-out IDs; filler capacity accounts for inserted records and question",
                  "held_out_scope": "32k adaptation split only; warm-start overlap is not established",
                  "synthetic": True, "benchmark": "RULER-style diagnostic, not official RULER"}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance), flush=True)


def make_layout(row, tokenizer, training, ratio):
    from prefilling_v2.train_fulltext import resolve_landmark_token_id
    fields = {name: row[name] for name in RetrievalCase.__dataclass_fields__}
    case = RetrievalCase(**fields)
    ids = corrupt_background(case, ratio, int(tokenizer.mask_token_id))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    layout = build_generation_layout(
        prompt_ids=ids, answer_tokens=32,
        physical_length=row["physical_length"], chunk_size=64, mask_token_id=int(tokenizer.mask_token_id),
        pad_token_id=int(pad), landmark_token_id=resolve_landmark_token_id(training, tokenizer),
        insert_landmarks=training.attention_mode not in {"dsa", "nsa"},
    )
    return case, layout


def compare(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("non-finite model output")
    return {"relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-8)),
            "cosine": float(torch.nn.functional.cosine_similarity(a[None], b[None]))}


@torch.inference_mode()
def run(args):
    from prefilling_v2.train_fulltext import (
        _build_model_and_tokenizer,
        _kernel_fallback_count,
        initialize_and_configure_trainables,
        parse_args,
    )
    from prefilling_v2.fastdllm_v1 import DreamRouteFastDLLM
    from prefilling_v2.diagnostic_probe import RoutingProbe
    from scripts.prefilling_v2.eval_longbench_mfen import decode_answer, load_trainables
    args.lengths = diagnostic_lengths(args.phase, args.lengths)
    torch.manual_seed(7)
    training = parse_args(["--config", args.training_config, "--model_path", args.model_path, "--no_gradient_checkpointing"])
    if training.chunk_size != 64 or training.attention_mode not in {"route", "dsa", "nsa"}:
        raise ValueError("diagnostics require the trained chunk-64 route/DSA/NSA setup")
    if training.attention_mode in {"dsa", "nsa"} and args.variants != ["baseline"]:
        raise ValueError("DSA/NSA have no route oracle interventions")
    allowed = {"baseline", "oracle_chunk", "oracle_token", "oracle_both", "flat_gate", "chunks31", "tokens768", "lmk_only_route"}
    if set(args.variants) - allowed:
        raise ValueError("unknown intervention")
    model, tok, plan = _build_model_and_tokenizer(training, torch.device("cuda"))
    initialize_and_configure_trainables(model, training)
    load_trainables(model, Path(args.checkpoint))
    from prefilling_v2.qcal import set_qcal_scale
    qcal_modules = set_qcal_scale(model, args.qcal_scale)
    model.eval()
    decoder = DreamRouteFastDLLM(model=model, mask_token_id=int(tok.mask_token_id), threshold=0.9, bootstrap=args.bootstrap)
    rows = [json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()]
    rows = [row for row in rows if row["physical_length"] in args.lengths]
    if args.limit:
        rows = rows[:args.limit]
    rows = rows[args.rank::args.world_size]
    if not rows:
        raise ValueError("empty diagnostic shard")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = {"manifest_sha256": sha256(args.manifest), "checkpoint_sha256": sha256(Path(args.checkpoint) / "trainable_state.pt"),
                "config_sha256": sha256(args.training_config), "phase": args.phase, "model": training.attention_mode,
                "bootstrap": args.bootstrap, "diagnostic_version": "bridge-v1",
                "max_probe_queries": args.max_probe_queries}
    if not math.isfinite(args.qcal_scale):
        raise ValueError("qcal_scale must be finite")
    if args.qcal_scale != 1.0 and qcal_modules == 0:
        raise ValueError("non-unit qcal scale requires Q-Cal modules")
    if args.qcal_scale != 1.0:
        identity["qcal_scale"] = args.qcal_scale
    existing = [json.loads(x) for x in output.read_text().splitlines() if x.strip()] if output.exists() and args.phase != "validate" else []
    if any(record["identity"] != identity for record in existing):
        raise ValueError("resume provenance mismatch")
    done = {record["key"] for record in existing}
    if args.phase == "validate":
        validations = []
        for row in rows:
            case, layout = make_layout(row, tok, training, 0)
            ids, valid, positions = [x[None].cuda() for x in (layout.input_ids, layout.attention_mask, layout.position_ids)]
            landmarks = layout.landmark_positions.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                first = decoder.prefill(ids, valid, positions, landmarks)
                expected = first.logits[:, layout.predictor_positions].clone()
                cache = first.cache
                del first
                partial = decoder.cached_forward(ids, build_fastdllm_block_layouts(layout, 32, 64)[0], cache, landmarks)
                query_rows = torch.searchsorted(partial.query_positions, layout.predictor_positions.cuda())
                partial_comparison = compare(partial.logits[:, query_rows], expected)
                del partial, cache
                native_outputs = [expected]
                native_comparisons = []
                for _ in range(2):
                    repeated = decoder.prefill(ids, valid, positions, landmarks)
                    repeated_logits = repeated.logits[:, layout.predictor_positions].clone()
                    del repeated
                    native_comparisons.extend(compare(repeated_logits, ref) for ref in native_outputs)
                    native_outputs.append(repeated_logits)
                with RoutingProbe(decoder, max_queries=args.max_probe_queries) as probe:
                    probe.begin(case, layout, {"case_id": row["case_id"]})
                    observed = decoder.prefill(ids, valid, positions, landmarks)
                    hook_comparisons = [compare(observed.logits[:, layout.predictor_positions], ref) for ref in native_outputs]
                    hook_comparison = {"relative_l2": max(x["relative_l2"] for x in hook_comparisons),
                                       "cosine": min(x["cosine"] for x in hook_comparisons)}
                    if len(probe.records) != 7:
                        raise AssertionError(f"expected seven sparse-layer observations, got {len(probe.records)}")
                    if training.attention_mode == "route" and any(
                        "score_decomposition" not in record
                        for record in probe.records if record["forward_index"] == 0
                    ):
                        raise AssertionError("missing initial-state route score decomposition")
                    del observed
            observer_limit = observer_error_limit([x["relative_l2"] for x in native_comparisons])
            validation = {"case_id": row["case_id"], "partial_same_input": partial_comparison,
                          "native_repeatability": native_comparisons, "observer_passthrough": hook_comparison,
                          "observer_relative_l2_limit": observer_limit, "observer_comparisons": hook_comparisons}
            if (partial_comparison["relative_l2"] > 0.05 or partial_comparison["cosine"] < 0.998
                or hook_comparison["relative_l2"] > observer_limit or hook_comparison["cosine"] < 0.998
                or min(x["cosine"] for x in native_comparisons) < 0.998):
                output.write_text(json.dumps({"identity": identity, "passed": False, "validations": [validation]}, indent=2) + "\n")
                raise AssertionError(f"model/cache/observer validation failed: {validation}")
            validations.append(validation)
            del native_outputs, expected, repeated_logits
        if _kernel_fallback_count(model):
            raise AssertionError("unexpected kernel fallback")
        result = {"identity": identity, "passed": True, "validations": validations, "fallback_count": 0}
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    with RoutingProbe(decoder, max_queries=args.max_probe_queries) as probe:
        for variant in args.variants:
            for layer in decoder._core().model.layers:
                attn = layer.self_attn
                if hasattr(attn, "token_budget"):
                    attn.topk = 31 if variant == "chunks31" else training.route_topk
                    attn.token_budget = 768 if variant == "tokens768" else training.route_token_budget
            for row in rows:
                for ratio in (args.ratios if args.phase == "probe" else [0]):
                    key = f"{row['case_id']}:{variant}:noise{ratio:g}"
                    if key in done:
                        continue
                    case, layout = make_layout(row, tok, training, ratio)
                    meta = {k: row[k] for k in ("case_id", "physical_length", "task", "seed", "depth", "validation_pack_id")}
                    meta["background_mask_ratio"] = ratio
                    probe.begin(case, layout, meta, variant)
                    ids, valid, positions = [x[None].cuda() for x in (layout.input_ids, layout.attention_mask, layout.position_ids)]
                    landmarks = layout.landmark_positions.cuda()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        if args.phase == "probe":
                            pref = decoder.prefill(ids, valid, positions, landmarks)
                            gold = tok(" " + " ".join(case.answers), add_special_tokens=False).input_ids
                            if not gold or len(gold) > layout.predictor_positions.numel():
                                raise ValueError("gold answer does not fit the diagnostic span")
                            gold_tensor = torch.tensor(gold, device="cuda")
                            logits = pref.logits[0, layout.predictor_positions[:len(gold)]].float()
                            extra = {"gold_answer_span_nll": float(torch.nn.functional.cross_entropy(logits, gold_tensor))}
                            del pref, logits
                        else:
                            generated, stats = decoder.generate(input_ids=ids, attention_mask=valid, position_ids=positions,
                                blocks=build_fastdllm_block_layouts(layout, 32, 64), landmark_positions=landmarks)
                            answer_ids = generated[0, layout.answer_positions.cuda()].tolist()
                            prediction = decode_answer(tok, answer_ids)
                            extra = dict(prediction=prediction, answers=case.answers,
                                         raw_answer_ids=answer_ids,
                                         raw_decoded=tok.decode(answer_ids, skip_special_tokens=False),
                                         eos_offset=answer_ids.index(tok.eos_token_id) if tok.eos_token_id in answer_ids else None,
                                         finished_in_prefill=stats.cached_forwards == 0,
                                         **score_codes(prediction, case.answers), **asdict(stats))
                            if stats.full_prefills != 1:
                                raise AssertionError("32-token diagnostic must prefill once")
                            del generated
                    torch.cuda.synchronize()
                    fallback = _kernel_fallback_count(model)
                    if fallback or not probe.records:
                        raise AssertionError(f"invalid diagnostics: fallback={fallback}, records={len(probe.records)}")
                    if training.attention_mode == "route" and any(
                        "score_decomposition" not in record
                        for record in probe.records if record["forward_index"] == 0
                    ):
                        raise AssertionError("missing initial-state route score decomposition")
                    result = dict(identity=identity, key=key, **meta, variant=variant, **extra,
                                  instrumented_seconds=time.perf_counter() - started, fallback_count=fallback,
                                  observations=probe.records, interventions=probe.interventions)
                    append_jsonl_fsync(output, result)
                    print(json.dumps({"key": key, "model": training.attention_mode, "fallback": fallback,
                                      "observations": len(probe.records), "seconds": result["instrumented_seconds"]}), flush=True)


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.lengths = diagnostic_lengths(arguments.phase, arguments.lengths)
    if (not 0 <= arguments.rank < arguments.world_size or arguments.limit < 0
            or arguments.max_probe_queries < 2):
        raise ValueError("invalid shard or limit")
    if arguments.phase == "prepare":
        prepare(arguments)
    else:
        run(arguments)
