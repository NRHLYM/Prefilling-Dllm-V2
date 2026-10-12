"""Resumable two-GPU LoRA training for Dream with kernelized route attention."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from prefilling_v2.attention import install_dream_sparse_attention

try:
    from prefilling_v2.attention import set_route_fusion_detach
except ImportError:  # ROUTE_FT_ROOT snapshot predates fusion detach

    def set_route_fusion_detach(model, detach: bool) -> int:
        enabled = bool(detach)
        updated = 0
        for module in model.modules():
            if hasattr(module, "detach_fusion_weights"):
                module.detach_fusion_weights = enabled
                updated += 1
        if hasattr(model, "config"):
            model.config.route_detach_fusion_weights = enabled
        return updated
from prefilling_v2.dsa_attention import (
    DSA_SELECTED_ATTENTION_SEMANTICS,
    collect_dsa_index_loss,
    install_dream_dsa_attention,
    prepare_dsa_index_losses,
    set_dsa_official_stage,
)
from prefilling_v2.nsa_attention import install_dream_nsa_attention
from prefilling_v2.checkpointing import (
    capture_rng_state,
    load_trainable_checkpoint,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from prefilling_v2.data import (
    FullTextComplementaryCollator,
    RULER_VIEW_ID_BASE,
    RulerDenoisingSynthesizer,
)
from prefilling_v2.distributed import (
    all_gather_object,
    all_reduce_detached_sum,
    barrier,
    destroy_distributed,
    global_target_count,
    init_distributed,
    scale_local_loss_for_ddp,
)
from prefilling_v2.packed_corpus import DreamPackedCorpus


DEFAULTS: dict[str, object] = {
    "attention_mode": "route",
    "non_route_attention": "sliding",
    "model_path": "models/Dream-v0-Base-7B",
    "model_max_position_embeddings": None,
    "model_rope_theta": None,
    "model_rope_scaling": None,
    "corpus_bin": "data/dolma3_longmino/dream-8k/train.bin",
    "corpus_meta": "data/dolma3_longmino/dream-8k/train.meta.json",
    "output_dir": "outputs/dream-route-dolma3-8k-pilot",
    "resume_from": None,
    "initialize_from": None,
    "max_length": 8192,
    "max_steps": 500,
    "stop_after_steps": None,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "lr_schedule": "cosine",
    "warmup_steps": 25,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "route_interleave": 4,
    "local_window": 512,
    "swa_local_window": None,
    "chunk_size": 64,
    "route_topk": 16,
    "route_token_budget": 0,
    "route_token_policy": "global_qk",
    "route_min_tokens_per_chunk": 1,
    "route_token_relaxation": "none",
    "route_token_gumbel_scale": 1.0,
    "route_relaxation": "none",
    "route_temperature": 1.0,
    "route_gumbel_scale": 1.0,
    "route_selection_mode": "post_softmax",
    "route_residual_weight": 0.0,
    "route_qcal_rank": 0,
    "route_freeze_qcal": False,
    "route_trainable_scope": "full",
    "route_chunk_summary": "attn",
    "route_entropy_prior": True,
    "route_detach_fusion_weights": False,
    "route_value_fusion_beta": 0.0,
    "route_value_fusion_rank": 32,
    "route_value_fusion_query_block": 128,
    "route_lmk_kl_ste": True,
    "route_lmk_ce_ste": False,
    "route_allchunk_st_queries": 0,
    "route_allchunk_st_temperature": 1.0,
    "route_allchunk_st_temperature_end": None,
    "route_allchunk_st_anneal": "cosine",
    "route_evidence_route_loss_weight": 0.0,
    "route_evidence_token_attn_weight": 0.0,
    "route_support_attn_kl_weight": 0.0,
    "route_support_attn_queries": 16,
    "route_support_attn_query_chunk": 16,
    "route_support_attn_temperature": 1.5,
    "route_support_attn_detach_gate": False,
    "route_asymmetric_gate_ce": False,
    "route_balanced_view_ce": False,
    "route_force_remote_unit": False,
    "route_force_remote_oracle_route": False,
    "route_gate_bce_weight": 0.0,
    "route_gate_ce_force": False,
    "route_qcal_max_grad_norm": 0.0,
    "route_lora_q_max_grad_norm": 0.0,
    "route_gate_offset_only": False,
    "route_gate_affine_only": False,
    "route_train_gate_offset": False,
    "route_lora_q_only": False,
    "route_lora_q_lr": 0.0,
    "route_qcal_lr": 0.0,
    "route_gate_offset_lr": 0.0,
    "route_gate_scale_lr": 0.0,
    "route_gate_offset_max_grad_norm": 0.0,
    "route_gate_scale_max_grad_norm": 0.0,
    "route_gate_offset_abs_max": 0.0,
    "route_gate_scale_min": 0.05,
    "route_gate_scale_max": 4.0,
    "route_gate_cal_clip_each": False,
    "route_gate_cal_init_from": None,
    "route_gate_train_offset_layers": None,
    "route_gate_train_scale_layers": None,
    "route_gate_offset_layer_lr": None,
    "route_gate_scale_layer_lr": None,
    "route_gate_offset_layer_clip": None,
    "route_gate_scale_layer_clip": None,
    "route_distant_span_min": 8,
    "route_distant_span_max": 16,
    "route_distant_min_gap": None,
    "route_distant_needles_min": 1,
    "route_distant_needles_max": 1,
    "route_distant_needle_sep": 64,
    "route_distant_cue_len": 0,
    "route_chunk_aux_queries": 16,
    "route_chunk_aux_loss_weight": 0.0,
    "route_dense_teacher_weight": 0.0,
    "route_dense_teacher_rule": "lse",
    "route_dense_teacher_queries": 16,
    "route_dense_teacher_temperature": 1.0,
    "route_dense_teacher_source": "online",
    "route_backend": "kernel_bidir",
    "no_kernel_fallback": True,
    "dsa_topk": 1024,
    "dsa_index_heads": 8,
    "dsa_index_head_dim": 128,
    "dsa_query_block_size": 64,
    "dsa_attention_query_block_size": 64,
    "dsa_backend": "torch",
    "dsa_aux_queries": 32,
    "dsa_aux_loss_weight": 0.01,
    "dsa_aux_loss_scope": "full",
    "dsa_aux_scale_by_targets": True,
    "dsa_lm_loss_weight": 1.0,
    "dsa_warmup_steps": 0,
    "dsa_warmup_indexer_lr": 0.001,
    "dsa_sparse_aux_scope": "selected",
    "dsa_sparse_aux_queries": 0,
    "nsa_block_count": 32,
    "nsa_compress_block": 32,
    "nsa_compress_stride": 16,
    "nsa_select_block": 64,
    "nsa_backend": "tilelang",
    "lmk_token_mode": "mask",
    "ruler_mix_ratio": 0.0,
    "ruler_answer_ce_weight": 1.0,
    "route_sync_ruler_ce": False,
    "sync_ruler_all_tasks": False,
    "ruler_joint_local_mask": False,
    "ruler_task_ids": [0, 1, 2],
    "ruler_val_batches": 0,
    "loss_chunk_size": 2048,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "gradient_checkpointing": True,
    "t_min": 0.2,
    "t_max": 0.8,
    "save_steps": 50,
    "log_steps": 10,
    "seed": 7,
    "num_workers": 0,
    "validation_packs": 0,
    "validation_batches": 0,
    "eval_steps": 0,
    "eval_only": False,
    "overfit_packs": 0,
}

_QCAL_SCOPES = frozenset({"qcal_only", "lora_qcal", "qcal_lmk", "lora_qcal_lmk"})
_LORA_SCOPES = frozenset({"lora_qcal", "lora_qcal_lmk", "lora_lmk"})
_LMK_SCOPES = frozenset({"qcal_lmk", "lora_qcal_lmk", "lora_lmk"})
_SCOPED_TRAINABLE = frozenset(_QCAL_SCOPES | {"lora_lmk"})
_FROZEN_LORA_SCOPES = frozenset({"qcal_only", "qcal_lmk"})


def _as_int_list(value) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value) if value.strip().startswith("[") else [
            item.strip() for item in value.split(",") if item.strip()
        ]
    return [int(item) for item in value]


def _as_float_map(value) -> dict[int, float]:
    if not value:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    return {int(key): float(item) for key, item in dict(value).items()}


def _gate_cal_params(model: torch.nn.Module, suffix: str) -> list[tuple[str, torch.nn.Parameter]]:
    unwrapped = model.module if hasattr(model, "module") else model
    return [
        (name, parameter)
        for name, parameter in unwrapped.named_parameters()
        if name.endswith(suffix)
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model_path")
    parser.add_argument("--model_max_position_embeddings", type=int)
    parser.add_argument("--model_rope_theta", type=float)
    parser.add_argument("--model_rope_scaling", type=json.loads)
    parser.add_argument("--corpus_bin")
    parser.add_argument("--corpus_meta")
    parser.add_argument("--output_dir")
    parser.add_argument("--resume_from")
    parser.add_argument("--initialize_from")
    parser.add_argument("--attention_mode", choices=("route", "dsa", "nsa", "dense"))
    parser.add_argument(
        "--non_route_attention", choices=("sliding", "dense")
    )
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--stop_after_steps", type=int)
    parser.add_argument("--micro_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument(
        "--lr_schedule",
        choices=("cosine", "constant"),
        help="cosine decays to 0 by max_steps; constant holds the peak after warmup.",
    )
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument(
        "--route_qcal_max_grad_norm",
        type=float,
        help="Clip Q-Cal grads independently before the global clip. 0 disables.",
    )
    parser.add_argument(
        "--route_lora_q_max_grad_norm",
        type=float,
        help="Clip LoRA-Q grads independently before the global clip. 0 disables.",
    )
    parser.add_argument("--route_lora_q_lr", type=float)
    parser.add_argument("--route_qcal_lr", type=float)
    parser.add_argument("--route_gate_offset_lr", type=float)
    parser.add_argument("--route_gate_scale_lr", type=float)
    parser.add_argument(
        "--route_gate_offset_max_grad_norm",
        type=float,
        help="Clip fusion_gate_offset grads independently. 0 disables.",
    )
    parser.add_argument(
        "--route_gate_scale_max_grad_norm",
        type=float,
        help="Clip fusion_gate_scale grads independently. 0 disables.",
    )
    parser.add_argument(
        "--route_gate_offset_abs_max",
        type=float,
        help="Clamp |fusion_gate_offset| after each step. 0 disables.",
    )
    parser.add_argument(
        "--route_gate_scale_min",
        type=float,
        help="Lower clamp for fusion_gate_scale after each step.",
    )
    parser.add_argument(
        "--route_gate_scale_max",
        type=float,
        help="Upper clamp for fusion_gate_scale after each step.",
    )
    parser.add_argument(
        "--route_gate_cal_clip_each",
        dest="route_gate_cal_clip_each",
        action="store_true",
        help="Clip each gate offset/scale scalar separately to avoid interlayer steal.",
    )
    parser.add_argument(
        "--no_route_gate_cal_clip_each",
        dest="route_gate_cal_clip_each",
        action="store_false",
    )
    parser.add_argument(
        "--route_gate_cal_init_from",
        help="Overlay fusion_gate_offset/scale from this checkpoint after initialize_from.",
    )
    parser.add_argument("--route_gate_train_offset_layers", type=json.loads)
    parser.add_argument("--route_gate_train_scale_layers", type=json.loads)
    parser.add_argument("--route_gate_offset_layer_lr", type=json.loads)
    parser.add_argument("--route_gate_scale_layer_lr", type=json.loads)
    parser.add_argument("--route_gate_offset_layer_clip", type=json.loads)
    parser.add_argument("--route_gate_scale_layer_clip", type=json.loads)
    parser.add_argument("--route_interleave", type=int)
    parser.add_argument("--local_window", type=int)
    parser.add_argument(
        "--swa_local_window",
        type=int,
        help="Window radius for the 21 non-sparse SWA layers. "
        "Defaults to local_window. Set 1280 to match the all-SWA baseline.",
    )
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--route_topk", type=int)
    parser.add_argument("--route_token_budget", type=int)
    parser.add_argument(
        "--route_token_policy",
        choices=("global_qk", "entropy_adaptive"),
    )
    parser.add_argument("--route_min_tokens_per_chunk", type=int)
    parser.add_argument(
        "--route_token_relaxation",
        choices=("none", "gumbel_topk"),
    )
    parser.add_argument("--route_token_gumbel_scale", type=float)
    parser.add_argument(
        "--route_relaxation",
        choices=("none", "gumbel_topk", "gumbel_softmax_topk", "gumbel_softmax_topk_st"),
    )
    parser.add_argument("--route_temperature", type=float)
    parser.add_argument("--route_gumbel_scale", type=float)
    parser.add_argument(
        "--route_selection_mode", choices=("post_softmax", "pre_softmax")
    )
    parser.add_argument("--route_residual_weight", type=float)
    parser.add_argument("--route_value_fusion_beta", type=float)
    parser.add_argument("--route_value_fusion_rank", type=int)
    parser.add_argument("--route_value_fusion_query_block", type=int)
    parser.add_argument("--route_qcal_rank", type=int)
    parser.add_argument(
        "--route_freeze_qcal",
        dest="route_freeze_qcal",
        action="store_true",
        help="Install Q-Cal at identity but do not train it (ablation B).",
    )
    parser.add_argument(
        "--no_route_freeze_qcal",
        dest="route_freeze_qcal",
        action="store_false",
    )
    parser.add_argument(
        "--route_trainable_scope",
        choices=("full", "qcal_only", "lora_qcal", "qcal_lmk", "lora_qcal_lmk", "lora_lmk"),
    )
    parser.add_argument(
        "--route_chunk_summary",
        choices=("attn", "mean"),
        help="Chunk key summary: attention-pool (mainline) or mean-pool ablation.",
    )
    parser.add_argument(
        "--route_entropy_prior",
        dest="route_entropy_prior",
        action="store_true",
        help="Add H(alpha_c) to chunk routing scores (mainline).",
    )
    parser.add_argument(
        "--no_route_entropy_prior",
        dest="route_entropy_prior",
        action="store_false",
    )
    parser.add_argument(
        "--route_detach_fusion_weights",
        dest="route_detach_fusion_weights",
        action="store_true",
    )
    parser.add_argument(
        "--no_route_detach_fusion_weights",
        dest="route_detach_fusion_weights",
        action="store_false",
    )
    parser.add_argument(
        "--route_lmk_kl_ste",
        dest="route_lmk_kl_ste",
        action="store_true",
        help="STE landmark type embed on the dense Z_c KL path (not CE).",
    )
    parser.add_argument(
        "--no_route_lmk_kl_ste",
        dest="route_lmk_kl_ste",
        action="store_false",
        help="Keep type embed off the KL graph so CE can train it.",
    )
    parser.add_argument(
        "--route_lmk_ce_ste",
        dest="route_lmk_ce_ste",
        action="store_true",
        help="STE landmark type embed into a soft-route CE bypass around hard top-k.",
    )
    parser.add_argument(
        "--no_route_lmk_ce_ste",
        dest="route_lmk_ce_ste",
        action="store_false",
    )
    parser.add_argument(
        "--route_allchunk_st_queries",
        type=int,
        help="S4: sampled supervised queries per route layer for all-chunk ST. 0 disables.",
    )
    parser.add_argument(
        "--route_allchunk_st_temperature",
        type=float,
        help="S4 Jacobian softmax τ at step 1. Does not change 33-way fusion τ.",
    )
    parser.add_argument(
        "--route_allchunk_st_temperature_end",
        type=float,
        help="S4 Jacobian softmax τ at max_steps. Omit to keep start temperature.",
    )
    parser.add_argument(
        "--route_allchunk_st_anneal",
        choices=("cosine", "linear", "none"),
        help="Anneal schedule for route_allchunk_st_temperature.",
    )
    parser.add_argument("--route_evidence_route_loss_weight", type=float)
    parser.add_argument("--route_evidence_token_attn_weight", type=float)
    parser.add_argument("--route_support_attn_kl_weight", type=float)
    parser.add_argument("--route_support_attn_queries", type=int)
    parser.add_argument("--route_support_attn_query_chunk", type=int)
    parser.add_argument("--route_support_attn_temperature", type=float)
    parser.add_argument(
        "--route_support_attn_detach_gate",
        dest="route_support_attn_detach_gate",
        action="store_true",
    )
    parser.add_argument(
        "--no_route_support_attn_detach_gate",
        dest="route_support_attn_detach_gate",
        action="store_false",
    )
    parser.add_argument(
        "--route_asymmetric_gate_ce",
        dest="route_asymmetric_gate_ce",
        action="store_true",
        help="Local Dolma CE detaches fusion; every-step distant infill CE trains the gate.",
    )
    parser.add_argument(
        "--no_route_asymmetric_gate_ce",
        dest="route_asymmetric_gate_ce",
        action="store_false",
    )
    parser.add_argument(
        "--route_balanced_view_ce",
        dest="route_balanced_view_ce",
        action="store_true",
        help="Stage-2: live fusion; 0.5 mean(local CE)+0.5 mean(distant CE).",
    )
    parser.add_argument(
        "--no_route_balanced_view_ce",
        dest="route_balanced_view_ce",
        action="store_false",
    )
    parser.add_argument(
        "--route_force_remote_unit",
        dest="route_force_remote_unit",
        action="store_true",
        help="1-token KEY/VALUE copy; force remote=1; freeze LoRA; Q-Cal/LMK only.",
    )
    parser.add_argument(
        "--no_route_force_remote_unit",
        dest="route_force_remote_unit",
        action="store_false",
    )
    parser.add_argument(
        "--route_force_remote_oracle_route",
        dest="route_force_remote_oracle_route",
        action="store_true",
        help="Force-remote unit: splice the evidence chunk into top-k on labeled queries.",
    )
    parser.add_argument(
        "--no_route_force_remote_oracle_route",
        dest="route_force_remote_oracle_route",
        action="store_false",
    )
    parser.add_argument(
        "--route_gate_bce_weight",
        type=float,
        help="BCE on native fusion gate_logit at labeled queries (target remote=1).",
    )
    parser.add_argument(
        "--route_gate_ce_force",
        dest="route_gate_ce_force",
        action="store_true",
        help="After native-gate BCE, apply force-on so train CE uses remote VALUE.",
    )
    parser.add_argument(
        "--no_route_gate_ce_force",
        dest="route_gate_ce_force",
        action="store_false",
    )
    parser.add_argument(
        "--route_gate_offset_only",
        dest="route_gate_offset_only",
        action="store_true",
        help="Train only per-layer fusion_gate_offset; freeze Q-Cal/LMK.",
    )
    parser.add_argument(
        "--no_route_gate_offset_only",
        dest="route_gate_offset_only",
        action="store_false",
    )
    parser.add_argument(
        "--route_gate_affine_only",
        dest="route_gate_affine_only",
        action="store_true",
        help="Train per-layer fusion_gate_scale and fusion_gate_offset; freeze Q-Cal/LMK.",
    )
    parser.add_argument(
        "--no_route_gate_affine_only",
        dest="route_gate_affine_only",
        action="store_false",
    )
    parser.add_argument(
        "--route_train_gate_offset",
        dest="route_train_gate_offset",
        action="store_true",
        help="Train fusion_gate_offset together with the current trainable scope.",
    )
    parser.add_argument(
        "--no_route_train_gate_offset",
        dest="route_train_gate_offset",
        action="store_false",
    )
    parser.add_argument(
        "--route_lora_q_only",
        dest="route_lora_q_only",
        action="store_true",
        help="When LoRA is unfrozen, train only q_proj LoRA (freeze K/V/O).",
    )
    parser.add_argument(
        "--no_route_lora_q_only",
        dest="route_lora_q_only",
        action="store_false",
    )
    parser.add_argument("--route_distant_span_min", type=int)
    parser.add_argument("--route_distant_span_max", type=int)
    parser.add_argument("--route_distant_min_gap", type=int)
    parser.add_argument("--route_distant_needles_min", type=int)
    parser.add_argument("--route_distant_needles_max", type=int)
    parser.add_argument("--route_distant_needle_sep", type=int)
    parser.add_argument("--route_distant_cue_len", type=int)
    parser.add_argument("--route_chunk_aux_queries", type=int)
    parser.add_argument("--route_chunk_aux_loss_weight", type=float)
    parser.add_argument("--route_dense_teacher_weight", type=float)
    parser.add_argument("--route_dense_teacher_rule", choices=("lse", "max", "top4_lse", "half_entropy", "weighted", "attn_mass"))
    parser.add_argument("--route_dense_teacher_queries", type=int)
    parser.add_argument("--route_dense_teacher_temperature", type=float)
    parser.add_argument(
        "--route_dense_teacher_source",
        choices=("online", "base", "same_forward", "dense"),
    )
    parser.add_argument(
        "--route_backend",
        choices=("torch_bidir", "chunk_kernel_bidir", "kernel_bidir"),
    )
    parser.add_argument(
        "--no_kernel_fallback", dest="no_kernel_fallback", action="store_true"
    )
    parser.add_argument(
        "--allow_kernel_fallback", dest="no_kernel_fallback", action="store_false"
    )
    parser.add_argument("--nsa_block_count", type=int)
    parser.add_argument("--nsa_compress_block", type=int)
    parser.add_argument("--nsa_compress_stride", type=int)
    parser.add_argument("--nsa_select_block", type=int)
    parser.add_argument("--nsa_backend", choices=("tilelang",))
    parser.add_argument("--dsa_topk", type=int)
    parser.add_argument("--dsa_index_heads", type=int)
    parser.add_argument("--dsa_index_head_dim", type=int)
    parser.add_argument("--dsa_query_block_size", type=int)
    parser.add_argument("--dsa_attention_query_block_size", type=int)
    parser.add_argument("--dsa_backend", choices=("torch", "tilelang"))
    parser.add_argument("--dsa_aux_queries", type=int)
    parser.add_argument("--dsa_aux_loss_weight", type=float)
    parser.add_argument(
        "--dsa_aux_loss_scope", choices=("full", "selected")
    )
    parser.add_argument("--dsa_aux_scale_by_targets", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dsa_lm_loss_weight", type=float)
    parser.add_argument("--dsa_warmup_steps", type=int)
    parser.add_argument("--dsa_warmup_indexer_lr", type=float)
    parser.add_argument(
        "--dsa_sparse_aux_scope", choices=("full", "selected")
    )
    parser.add_argument("--dsa_sparse_aux_queries", type=int)
    parser.add_argument(
        "--lmk_token_mode",
        choices=("mask", "external", "mask_type", "eos", "vocab"),
    )
    parser.add_argument(
        "--route_token",
        help="New tokenizer token for lmk_token_mode=vocab (default <|route|>).",
    )
    parser.add_argument("--ruler_mix_ratio", type=float)
    parser.add_argument(
        "--route_sync_ruler_ce",
        dest="route_sync_ruler_ce",
        action="store_true",
        help="Every step: complementary Dolma + RULER; 0.5 mean(Dolma CE)+0.5 mean(RULER CE).",
    )
    parser.add_argument(
        "--no_route_sync_ruler_ce",
        dest="route_sync_ruler_ce",
        action="store_false",
    )
    parser.add_argument(
        "--sync_ruler_all_tasks",
        dest="sync_ruler_all_tasks",
        action="store_true",
        help="Every step: Dolma + all RULER tasks; equal task-mean CE.",
    )
    parser.add_argument(
        "--no_sync_ruler_all_tasks",
        dest="sync_ruler_all_tasks",
        action="store_false",
    )
    parser.add_argument(
        "--ruler_answer_ce_weight",
        type=float,
        help="α in (Dolma CE + α RULER-answer CE) / (N_dolma + α N_ruler). 1 is unweighted.",
    )
    parser.add_argument(
        "--ruler_joint_local_mask",
        dest="ruler_joint_local_mask",
        action="store_true",
    )
    parser.add_argument(
        "--no_ruler_joint_local_mask",
        dest="ruler_joint_local_mask",
        action="store_false",
    )
    parser.add_argument("--ruler_task_ids", nargs="+", type=int)
    parser.add_argument("--ruler_val_batches", type=int)
    parser.add_argument("--loss_chunk_size", type=int)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        choices=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    parser.add_argument(
        "--gradient_checkpointing", dest="gradient_checkpointing", action="store_true"
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
    )
    parser.add_argument("--t_min", type=float)
    parser.add_argument("--t_max", type=float)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--log_steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--validation_packs", type=int)
    parser.add_argument("--validation_batches", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument(
        "--eval_only",
        dest="eval_only",
        action="store_true",
        help="Run the startup validation/ablations and exit.",
    )
    parser.add_argument("--overfit_packs", type=int)
    parser.set_defaults(**DEFAULTS)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    preliminary, _ = pre_parser.parse_known_args(argv)
    parser = _build_parser()
    if preliminary.config is not None:
        config = json.loads(preliminary.config.read_text(encoding="utf-8"))
        valid_keys = {action.dest for action in parser._actions}
        unknown = sorted(set(config) - valid_keys)
        if unknown:
            raise ValueError(f"Unknown config key(s): {unknown}")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    validate_training_config(vars(args))
    return args


def validate_training_config(config: dict[str, object]) -> None:
    rank = int(config.get("route_qcal_rank", 0))
    if rank < 0:
        raise ValueError("route_qcal_rank must be nonnegative")
    if rank and (config.get("attention_mode") != "route" or
                 config.get("route_backend") != "kernel_bidir" or
                 not config.get("no_kernel_fallback") or
                 float(config.get("route_chunk_aux_loss_weight", 0)) != 0):
        raise ValueError("Q-Cal experiment requires strict kernel_bidir route without chunk KL")
    trainable_scope = str(config.get("route_trainable_scope", "full"))
    if trainable_scope not in {"full"} | _SCOPED_TRAINABLE:
        raise ValueError("unsupported route_trainable_scope")
    if trainable_scope == "lora_lmk":
        if rank != 0 or config.get("attention_mode") != "route":
            raise ValueError(
                "lora_lmk is the no-Q-Cal ablation; requires route and route_qcal_rank=0"
            )
    elif trainable_scope in _QCAL_SCOPES and (
        rank <= 0 or config.get("attention_mode") != "route"
    ):
        raise ValueError("scoped routing training requires route and a positive Q-Cal rank")
    chunk_summary = str(config.get("route_chunk_summary", "attn"))
    if chunk_summary not in {"attn", "mean"}:
        raise ValueError(f"unsupported route_chunk_summary={chunk_summary}")
    freeze_qcal = bool(config.get("route_freeze_qcal", False))
    if freeze_qcal:
        if rank <= 0 or trainable_scope not in _QCAL_SCOPES:
            raise ValueError("route_freeze_qcal requires a positive Q-Cal rank and a Q-Cal scope")
        if trainable_scope == "qcal_only":
            raise ValueError("route_freeze_qcal leaves qcal_only with no trainables")
    if trainable_scope in _FROZEN_LORA_SCOPES and not (
        config.get("initialize_from") or config.get("resume_from")
    ):
        raise ValueError(
            "frozen-LoRA routing scopes require initialize_from or resume_from"
        )
    if trainable_scope in _LMK_SCOPES and str(config.get("lmk_token_mode", "mask")) not in {
        "mask_type",
        "external",
        "vocab",
    }:
        raise ValueError(
            "landmark Q-Cal scopes require lmk_token_mode mask_type, external, or vocab"
        )
    dense_weight = float(config.get("route_dense_teacher_weight", 0))
    if dense_weight < 0 or float(config.get("route_dense_teacher_temperature", 1)) <= 0:
        raise ValueError("invalid dense teacher weight/temperature")
    teacher_source = str(config.get("route_dense_teacher_source", "same_forward"))
    if teacher_source not in {"online", "base", "same_forward", "dense"}:
        raise ValueError("route_dense_teacher_source must be online, base, same_forward, or dense")
    if dense_weight > 0 and trainable_scope not in _QCAL_SCOPES:
        raise ValueError("dense teacher requires scoped Q-Cal training")
    if dense_weight > 0 and teacher_source == "dense" and not config.get("initialize_from"):
        raise ValueError("route_dense_teacher_source=dense requires initialize_from")
    if dense_weight > 0 and teacher_source == "base" and int(config.get("route_dense_teacher_queries", 16)) < 0:
        raise ValueError("dense teacher query count cannot be negative")
    if teacher_source == "base" and dense_weight <= 0:
        raise ValueError("base dense teacher requires a positive KL weight")
        support_kl_early = float(config.get("route_support_attn_kl_weight", 0.0) or 0.0)
        token_attn_early = float(
            config.get("route_evidence_token_attn_weight", 0.0) or 0.0
        )
        if token_attn_early > 0 and bool(config.get("route_detach_fusion_weights", False)):
            raise ValueError("evidence token attention keeps live fusion")
        if (
            trainable_scope in _QCAL_SCOPES
            and dense_weight <= 0
            and support_kl_early <= 0
            and bool(config.get("route_detach_fusion_weights", False))
            and not bool(config.get("route_asymmetric_gate_ce", False))
        ):
            if bool(config.get("route_force_remote_unit", False)):
                raise ValueError("force-remote unit keeps live remote mix weights")
            raise ValueError(
                "detached fusion with no dense KL leaves Q-Cal without a loss"
            )
    if (
        not bool(config.get("route_lmk_kl_ste", True))
        and trainable_scope not in _LMK_SCOPES
    ):
        raise ValueError("CE landmark type embed requires an LMK trainable scope")
    if bool(config.get("route_lmk_ce_ste", False)) and trainable_scope not in _LMK_SCOPES:
        raise ValueError("route_lmk_ce_ste requires an LMK trainable scope")
    if bool(config.get("route_lmk_kl_ste", True)) and bool(
        config.get("route_lmk_ce_ste", False)
    ):
        raise ValueError("do not STE type embed on both KL and CE")
    allchunk_queries = int(config.get("route_allchunk_st_queries", 0) or 0)
    if allchunk_queries < 0:
        raise ValueError("route_allchunk_st_queries must be nonnegative")
    allchunk_tau_raw = config.get("route_allchunk_st_temperature", 1.0)
    allchunk_tau = 1.0 if allchunk_tau_raw is None else float(allchunk_tau_raw)
    allchunk_tau_end = config.get("route_allchunk_st_temperature_end")
    if allchunk_tau_end is None:
        allchunk_tau_end = allchunk_tau
    else:
        allchunk_tau_end = float(allchunk_tau_end)
    allchunk_anneal = str(config.get("route_allchunk_st_anneal", "cosine") or "cosine")
    if allchunk_tau <= 0 or allchunk_tau_end <= 0:
        raise ValueError("route_allchunk_st_temperature must be positive")
    if allchunk_anneal not in {"cosine", "linear", "none"}:
        raise ValueError(f"unsupported route_allchunk_st_anneal={allchunk_anneal}")
    if allchunk_queries and bool(config.get("route_lmk_ce_ste", False)):
        raise ValueError("do not stack lmk_ce_ste (S3) with all-chunk ST (S4)")
    if allchunk_queries and (
        config.get("attention_mode") != "route"
        or trainable_scope not in _LMK_SCOPES
        or bool(config.get("route_detach_fusion_weights", False))
        or dense_weight != 0
        or int(config.get("route_token_budget", 0) or 0) != 0
        or float(config.get("route_residual_weight", 0) or 0) != 0
        or float(config.get("route_chunk_aux_loss_weight", 0) or 0) != 0
        or float(config.get("route_evidence_route_loss_weight", 0) or 0) != 0
        or float(config.get("route_evidence_token_attn_weight", 0) or 0) != 0
        or float(config.get("route_support_attn_kl_weight", 0) or 0) != 0
        or str(config.get("route_relaxation", "none")) != "none"
        or bool(config.get("route_asymmetric_gate_ce", False))
        or bool(config.get("route_balanced_view_ce", False))
        or bool(config.get("sync_ruler_all_tasks", False))
    ):
        raise ValueError(
            "all-chunk ST requires live fusion, no dense KL, hard top-k, no extra aux"
        )
    evidence_weight = float(
        config.get("route_evidence_route_loss_weight", 0.0)
    )
    if evidence_weight < 0:
        raise ValueError("route_evidence_route_loss_weight must be non-negative")
    if evidence_weight > 0 and (
        trainable_scope != "qcal_only"
        or float(config.get("ruler_mix_ratio", 0.0)) <= 0
    ):
        raise ValueError(
            "evidence route supervision requires qcal_only and RULER data"
        )
    token_attn_weight = float(
        config.get("route_evidence_token_attn_weight", 0.0) or 0.0
    )
    if token_attn_weight < 0:
        raise ValueError("route_evidence_token_attn_weight must be non-negative")
    if token_attn_weight > 0:
        if config.get("attention_mode") != "route":
            raise ValueError("evidence token attention requires attention_mode=route")
        if trainable_scope not in _LORA_SCOPES:
            raise ValueError("evidence token attention requires a LoRA Q-Cal scope")
        if float(config.get("ruler_mix_ratio", 0.0) or 0.0) <= 0:
            raise ValueError("evidence token attention requires RULER data")
        if bool(config.get("route_detach_fusion_weights", False)):
            raise ValueError("evidence token attention keeps live fusion")
        if dense_weight != 0:
            raise ValueError("do not stack evidence token attention with dense chunk KL")
        if allchunk_queries:
            raise ValueError("do not stack evidence token attention with all-chunk ST")
        if bool(config.get("route_lmk_ce_ste", False)):
            raise ValueError("do not stack evidence token attention with lmk_ce_ste")
        if evidence_weight > 0:
            raise ValueError("do not stack evidence token attention with chunk evidence CE")
        if str(config.get("route_relaxation", "none")) != "none":
            raise ValueError("evidence token attention uses hard top-k")
        if int(config.get("route_token_budget", 0) or 0) != 0:
            raise ValueError("evidence token attention requires the full selected chunk")
        if bool(config.get("route_force_remote_unit", False)):
            raise ValueError("do not stack evidence token attention with force-remote")
        if bool(config.get("route_asymmetric_gate_ce", False)):
            raise ValueError("do not stack evidence token attention with asymmetric gate CE")
        if bool(config.get("route_balanced_view_ce", False)):
            raise ValueError("do not stack evidence token attention with balanced view CE")
        if not config.get("initialize_from") and not config.get("resume_from"):
            raise ValueError("evidence token attention starts from a dense checkpoint")
    support_kl = float(config.get("route_support_attn_kl_weight", 0.0) or 0.0)
    if support_kl < 0:
        raise ValueError("route_support_attn_kl_weight must be non-negative")
    if support_kl > 0:
        if config.get("attention_mode") != "route":
            raise ValueError("support attn distill requires attention_mode=route")
        if trainable_scope not in _LORA_SCOPES:
            raise ValueError("support attn distill requires a LoRA Q-Cal scope")
        if dense_weight != 0:
            raise ValueError("do not stack support attn distill with dense chunk KL")
        if allchunk_queries:
            raise ValueError("do not stack support attn distill with all-chunk ST")
        if evidence_weight > 0:
            raise ValueError("do not stack support attn distill with evidence route CE")
        if token_attn_weight > 0:
            raise ValueError("do not stack support attn distill with evidence token attention")
        if not config.get("initialize_from"):
            raise ValueError("support attn distill requires initialize_from dense Q/K")
        if not bool(config.get("route_detach_fusion_weights", False)):
            raise ValueError("support attn distill keeps paper detach fusion")
        if str(config.get("route_dense_teacher_source", "")) != "dense":
            raise ValueError("support attn distill needs dense-500 Q/K snapshot")
        if float(config.get("route_support_attn_temperature", 1.5) or 0.0) < 1.0:
            raise ValueError("support attn temperature must be >= 1 (dense support is peaked)")
        if int(config.get("route_support_attn_queries", 16) or 0) < 0:
            raise ValueError("route_support_attn_queries must be nonnegative (0 = all MASK/answer spans)")
        if int(config.get("route_support_attn_query_chunk", 16) or 0) < 0:
            raise ValueError("route_support_attn_query_chunk must be nonnegative")
    asymmetric = bool(config.get("route_asymmetric_gate_ce", False))
    if asymmetric:
        if config.get("attention_mode") != "route":
            raise ValueError("asymmetric gate CE requires attention_mode=route")
        if trainable_scope not in _LORA_SCOPES:
            raise ValueError("asymmetric gate CE requires a LoRA Q-Cal scope")
        if bool(config.get("route_detach_fusion_weights", False)):
            raise ValueError(
                "asymmetric gate CE installs live fusion and detaches only on local CE"
            )
        if dense_weight != 0:
            raise ValueError("do not stack asymmetric gate CE with dense chunk KL")
        if allchunk_queries:
            raise ValueError("do not stack asymmetric gate CE with all-chunk ST")
        if evidence_weight > 0:
            raise ValueError("do not stack asymmetric gate CE with evidence route CE")
        if support_kl > 0:
            raise ValueError("do not stack asymmetric gate CE with support attn KL")
        if bool(config.get("route_lmk_ce_ste", False)):
            raise ValueError("do not stack asymmetric gate CE with lmk_ce_ste")
        if float(config.get("ruler_mix_ratio", 0.0) or 0.0) != 0.0:
            raise ValueError(
                "asymmetric gate CE uses an every-step distant view; keep ruler_mix_ratio=0"
            )
        span_min = int(config.get("route_distant_span_min", 8) or 8)
        span_max = int(config.get("route_distant_span_max", 16) or 16)
        if span_min < 1 or span_max < span_min:
            raise ValueError("invalid route_distant_span_min/max")
        gap = config.get("route_distant_min_gap")
        if gap is not None and int(gap) < int(config.get("local_window", 0) or 0):
            raise ValueError("route_distant_min_gap must be >= local_window")
    balanced = bool(config.get("route_balanced_view_ce", False))
    force_remote = bool(config.get("route_force_remote_unit", False))
    if force_remote and (asymmetric or balanced):
        raise ValueError("do not stack force-remote unit with copy-view CE")
    if balanced and asymmetric:
        raise ValueError("do not stack asymmetric gate CE with balanced view CE")
    if balanced:
        if config.get("attention_mode") != "route":
            raise ValueError("balanced view CE requires attention_mode=route")
        if trainable_scope not in _LORA_SCOPES:
            raise ValueError("balanced view CE requires a LoRA Q-Cal scope")
        if bool(config.get("route_detach_fusion_weights", False)):
            raise ValueError("balanced view CE requires live fusion on both views")
        if dense_weight != 0:
            raise ValueError("do not stack balanced view CE with dense chunk KL")
        if allchunk_queries:
            raise ValueError("do not stack balanced view CE with all-chunk ST")
        if evidence_weight > 0:
            raise ValueError("do not stack balanced view CE with evidence route CE")
        if support_kl > 0:
            raise ValueError("do not stack balanced view CE with support attn KL")
        if bool(config.get("route_lmk_ce_ste", False)):
            raise ValueError("do not stack balanced view CE with lmk_ce_ste")
        if float(config.get("ruler_mix_ratio", 0.0) or 0.0) != 0.0:
            raise ValueError("balanced view CE keeps ruler_mix_ratio=0")
        span_min = int(config.get("route_distant_span_min", 8) or 8)
        span_max = int(config.get("route_distant_span_max", 16) or 16)
        if span_min < 1 or span_max < span_min:
            raise ValueError("invalid route_distant_span_min/max")
        gap = config.get("route_distant_min_gap")
        if gap is not None and int(gap) < int(config.get("local_window", 0) or 0):
            raise ValueError("route_distant_min_gap must be >= local_window")
    if asymmetric or balanced:
        needles_min = int(config.get("route_distant_needles_min", 1) or 1)
        needles_max = int(config.get("route_distant_needles_max", 1) or 1)
        if needles_min < 1 or needles_max < needles_min:
            raise ValueError("invalid route_distant_needles_min/max")
        if int(config.get("route_distant_needle_sep", 64) or 0) < 0:
            raise ValueError("route_distant_needle_sep must be nonnegative")
        if int(config.get("route_distant_cue_len", 0) or 0) < 0:
            raise ValueError("route_distant_cue_len must be nonnegative")
        cue_len = int(config.get("route_distant_cue_len", 0) or 0)
        if cue_len and cue_len >= int(config.get("route_distant_span_min", 8) or 8):
            raise ValueError("route_distant_cue_len must leave a MASK remainder")
    if force_remote:
        if config.get("attention_mode") != "route":
            raise ValueError("force-remote unit requires attention_mode=route")
        if trainable_scope not in {"qcal_lmk", "lora_qcal_lmk"}:
            raise ValueError("force-remote unit uses qcal_lmk or lora_qcal_lmk")
        if bool(config.get("route_detach_fusion_weights", False)):
            raise ValueError("force-remote unit keeps live remote mix weights")
        if dense_weight != 0:
            raise ValueError("do not stack force-remote unit with dense chunk KL")
        if allchunk_queries:
            raise ValueError("do not stack force-remote unit with all-chunk ST")
        if evidence_weight > 0:
            raise ValueError("do not stack force-remote unit with evidence route CE")
        if support_kl > 0:
            raise ValueError("do not stack force-remote unit with support attn KL")
        if bool(config.get("route_lmk_ce_ste", False)):
            raise ValueError("do not stack force-remote unit with lmk_ce_ste")
        if float(config.get("ruler_mix_ratio", 0.0) or 0.0) != 0.0:
            raise ValueError("force-remote unit keeps ruler_mix_ratio=0")
        if int(config.get("route_token_budget", 0) or 0) != 0:
            raise ValueError("force-remote unit requires full selected chunks")
        if str(config.get("route_relaxation", "none")) != "none":
            raise ValueError("force-remote unit uses hard top-k")
        if int(config.get("route_distant_span_min", 0) or 0) != 2:
            raise ValueError("force-remote unit MASK is one VALUE token (span_min=2)")
        if int(config.get("route_distant_span_max", 0) or 0) != 2:
            raise ValueError("force-remote unit MASK is one VALUE token (span_max=2)")
        if int(config.get("route_distant_cue_len", 0) or 0) != 1:
            raise ValueError("force-remote unit keeps a 1-token KEY cue")
        if int(config.get("route_distant_needles_min", 1) or 1) != 1:
            raise ValueError("force-remote unit is single-needle")
        if int(config.get("route_distant_needles_max", 1) or 1) != 1:
            raise ValueError("force-remote unit is single-needle")
        gap = config.get("route_distant_min_gap")
        blind = int(config.get("local_window", 0) or 0)
        swa = config.get("swa_local_window")
        if swa is not None:
            blind = max(blind, int(swa))
        if gap is None or int(gap) < blind:
            raise ValueError(
                "route_distant_min_gap must be >= route local_window and SWA window"
            )
    if bool(config.get("route_force_remote_oracle_route", False)) and not force_remote:
        raise ValueError("oracle route requires route_force_remote_unit")
    gate_bce_weight = float(config.get("route_gate_bce_weight", 0.0) or 0.0)
    if gate_bce_weight < 0:
        raise ValueError("route_gate_bce_weight must be non-negative")
    if gate_bce_weight > 0:
        if not force_remote:
            raise ValueError("gate BCE requires route_force_remote_unit")
        if not bool(config.get("route_force_remote_oracle_route", False)):
            raise ValueError("gate BCE requires oracle route so remote contains the needle")
        if bool(config.get("route_detach_fusion_weights", False)):
            raise ValueError("gate BCE keeps live fusion")
    if bool(config.get("route_gate_ce_force", False)) and gate_bce_weight <= 0:
        raise ValueError("route_gate_ce_force requires route_gate_bce_weight > 0")
    if bool(config.get("route_gate_offset_only", False)) and bool(
        config.get("route_gate_affine_only", False)
    ):
        raise ValueError("route_gate_offset_only and route_gate_affine_only are mutually exclusive")
    for cal_flag in ("route_gate_offset_only", "route_gate_affine_only"):
        if bool(config.get(cal_flag, False)):
            if gate_bce_weight <= 0:
                raise ValueError(f"{cal_flag} requires route_gate_bce_weight > 0")
            if not force_remote:
                raise ValueError(f"{cal_flag} requires route_force_remote_unit")
    scale_min = float(config.get("route_gate_scale_min", 0.05) or 0.05)
    scale_max = float(config.get("route_gate_scale_max", 4.0) or 4.0)
    if scale_min <= 0 or scale_max < scale_min:
        raise ValueError("route_gate_scale_min/max must satisfy 0 < min <= max")
    offset_layers = _as_int_list(config.get("route_gate_train_offset_layers"))
    scale_layers = _as_int_list(config.get("route_gate_train_scale_layers"))
    for name, layers in (
        ("route_gate_train_offset_layers", offset_layers),
        ("route_gate_train_scale_layers", scale_layers),
    ):
        if layers is not None and any(index < 0 for index in layers):
            raise ValueError(f"{name} must be non-negative")
    if (offset_layers is not None or scale_layers is not None) and not bool(
        config.get("route_gate_affine_only", False)
    ):
        raise ValueError("per-layer gate-cal freeze requires route_gate_affine_only")
    if bool(config.get("route_gate_affine_only", False)):
        if offset_layers == [] and scale_layers == []:
            raise ValueError("affine-only layer lists freeze every gate-cal parameter")
    if bool(config.get("route_lora_q_only", False)):
        if trainable_scope not in _LORA_SCOPES:
            raise ValueError("route_lora_q_only requires a LoRA Q-Cal scope")
    schedule = str(config.get("lr_schedule", "cosine") or "cosine")
    if schedule not in {"cosine", "constant"}:
        raise ValueError("lr_schedule must be cosine or constant")
    for key in (
        "route_lora_q_lr",
        "route_qcal_lr",
        "route_gate_offset_lr",
        "route_gate_scale_lr",
        "route_lora_q_max_grad_norm",
        "route_qcal_max_grad_norm",
        "route_gate_offset_max_grad_norm",
        "route_gate_scale_max_grad_norm",
        "route_gate_offset_abs_max",
    ):
        if float(config.get(key, 0.0) or 0.0) < 0:
            raise ValueError(f"{key} must be non-negative")
    if float(config.get("route_chunk_aux_loss_weight", 0)) < 0:
        raise ValueError("route_chunk_aux_loss_weight must be non-negative")
    if float(config.get("route_chunk_aux_loss_weight", 0)) > 0:
        if config.get("attention_mode") != "route" or int(config.get("route_chunk_aux_queries", 16)) <= 0:
            raise ValueError("chunk distillation requires route and positive query count")
    if config.get("resume_from") and config.get("initialize_from"):
        # Scoped Q-Cal configs still name initialize_from. Resume then overwrites
        # trainables, optimizer, scheduler, and RNG from the checkpoint.
        pass
    attention_mode = str(config.get("attention_mode", "route"))
    if attention_mode not in {"route", "dsa", "nsa", "dense"}:
        raise ValueError(f"unsupported attention_mode={attention_mode}")
    non_route_attention = str(config.get("non_route_attention", "sliding"))
    if non_route_attention not in {"sliding", "dense"}:
        raise ValueError(
            f"unsupported non_route_attention={non_route_attention}"
        )
    max_length = int(config["max_length"])
    chunk_size = int(config["chunk_size"])
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if attention_mode != "dsa" and max_length % chunk_size != 0:
        raise ValueError("max_length must be divisible by chunk_size")
    swa_window = config.get("swa_local_window")
    if swa_window is not None:
        swa_window = int(swa_window)
        if swa_window <= 0 or swa_window >= max_length:
            raise ValueError(
                "swa_local_window must be positive and smaller than max_length"
            )
    if attention_mode == "route":
        if chunk_size < 16:
            raise ValueError("kernel_bidir requires chunk_size >= 16")
        if config["route_backend"] != "kernel_bidir":
            raise ValueError(
                "production route training requires route_backend=kernel_bidir"
            )
        if not bool(config["no_kernel_fallback"]):
            raise ValueError(
                "production route training requires kernel fallback disabled"
            )
        topk = int(config.get("route_topk", 16))
        if topk <= 0 or topk > 32:
            raise ValueError("kernel_bidir requires 1 <= route_topk <= 32")
        token_budget = int(config.get("route_token_budget", 0))
        if token_budget < 0 or token_budget > topk * chunk_size:
            raise ValueError(
                "route_token_budget must be between 0 and "
                "route_topk * chunk_size"
            )
        token_policy = str(
            config.get("route_token_policy", "global_qk")
        )
        if token_policy not in {"global_qk", "entropy_adaptive"}:
            raise ValueError(
                f"unsupported route_token_policy={token_policy}"
            )
        min_tokens = int(
            config.get("route_min_tokens_per_chunk", 1)
        )
        if min_tokens < 0:
            raise ValueError(
                "route_min_tokens_per_chunk must be non-negative"
            )
        if token_budget > 0 and token_budget < topk * min_tokens:
            raise ValueError(
                "route_token_budget must cover every routed chunk minimum"
            )
        token_relaxation = str(
            config.get("route_token_relaxation", "none")
        )
        if token_relaxation not in {"none", "gumbel_topk"}:
            raise ValueError(
                f"unsupported route_token_relaxation={token_relaxation}"
            )
        token_gumbel_scale = float(
            config.get("route_token_gumbel_scale", 1.0)
        )
        if token_gumbel_scale < 0:
            raise ValueError("route_token_gumbel_scale must be non-negative")
        route_relaxation = str(
            config.get("route_relaxation", "none")
        )
        if route_relaxation not in {
            "none",
            "gumbel_topk",
            "gumbel_softmax_topk",
            "gumbel_softmax_topk_st",
        }:
            raise ValueError(
                f"unsupported route_relaxation={route_relaxation}"
            )
        if route_relaxation == "gumbel_softmax_topk_st" and bool(
            config.get("route_detach_fusion_weights", False)
        ):
            raise ValueError(
                "gumbel_softmax_topk_st needs live fusion weights"
            )
        route_temperature = float(
            config.get("route_temperature", 1.0)
        )
        if route_temperature <= 0:
            raise ValueError("route_temperature must be positive")
        route_gumbel_scale = float(
            config.get("route_gumbel_scale", 1.0)
        )
        if route_gumbel_scale < 0:
            raise ValueError("route_gumbel_scale must be non-negative")
        value_fusion_beta = float(config.get("route_value_fusion_beta", 0.0) or 0.0)
        if value_fusion_beta < 0:
            raise ValueError("route_value_fusion_beta must be non-negative")
        if value_fusion_beta > 0 and int(config.get("route_value_fusion_rank", 32) or 0) <= 0:
            raise ValueError("route_value_fusion_rank must be positive when beta > 0")
        if bool(config.get("route_detach_fusion_weights", False)) and value_fusion_beta > 0:
            raise ValueError("value-aware fusion needs live fusion weights")
    elif attention_mode == "dsa":
        if str(config.get("dsa_backend", "torch")) not in {
            "torch",
            "tilelang",
        }:
            raise ValueError("unsupported dsa_backend")
        for key in (
            "dsa_topk",
            "dsa_index_heads",
            "dsa_index_head_dim",
            "dsa_query_block_size",
            "dsa_attention_query_block_size",
        ):
            if int(config.get(key, DEFAULTS[key])) <= 0:
                raise ValueError(f"{key} must be positive")
        if int(config.get("dsa_topk", 1024)) > max_length:
            raise ValueError("dsa_topk cannot exceed max_length")
        if int(config.get("dsa_aux_queries", 32)) < 0:
            raise ValueError("dsa_aux_queries must be non-negative")
        if int(config.get("dsa_sparse_aux_queries", 0) or 0) < 0:
            raise ValueError("dsa_sparse_aux_queries must be non-negative")
        if float(config.get("dsa_aux_loss_weight", 0.01)) < 0:
            raise ValueError("dsa_aux_loss_weight must be non-negative")
        if str(config.get("dsa_aux_loss_scope", "full")) not in {
            "full",
            "selected",
        }:
            raise ValueError("unsupported dsa_aux_loss_scope")
        if float(config.get("dsa_lm_loss_weight", 1.0)) < 0:
            raise ValueError("dsa_lm_loss_weight must be non-negative")
        warmup_steps = int(config.get("dsa_warmup_steps", 0) or 0)
        if warmup_steps < 0:
            raise ValueError("dsa_warmup_steps must be non-negative")
        if warmup_steps > 0 and warmup_steps >= int(config.get("max_steps", 0) or 0):
            raise ValueError("dsa_warmup_steps must be < max_steps")
        if (
            float(config.get("dsa_lm_loss_weight", 1.0)) == 0
            and float(config.get("dsa_aux_loss_weight", 0.01)) == 0
            and warmup_steps <= 0
        ):
            raise ValueError(
                "DSA training requires a non-zero LM or auxiliary loss weight"
            )
    elif attention_mode == "nsa":
        if str(config.get("nsa_backend", "tilelang")) != "tilelang":
            raise ValueError("Dream NSA production backend requires TileLang")
        block_count = int(config.get("nsa_block_count", 32))
        compress_block = int(config.get("nsa_compress_block", 32))
        compress_stride = int(config.get("nsa_compress_stride", 16))
        select_block = int(config.get("nsa_select_block", chunk_size))
        if block_count <= 0 or block_count > max_length // select_block:
            raise ValueError("nsa_block_count must fit the selected-block sequence")
        if compress_block <= 0 or compress_stride <= 0 or compress_stride > compress_block:
            raise ValueError("NSA compression block/stride are invalid")
        if select_block <= 0 or max_length % select_block:
            raise ValueError("nsa_select_block must tile max_length")
        if (max_length - compress_block) % compress_stride:
            raise ValueError("NSA compression windows must tile max_length")
        if chunk_size < 2:
            raise ValueError("Dream NSA requires chunk_size >= 2")
    elif chunk_size < 2:
        raise ValueError("dense control still requires chunk_size >= 2")
    if int(config["gradient_accumulation_steps"]) <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if int(config["micro_batch_size"]) <= 0:
        raise ValueError("micro_batch_size must be positive")
    model_max_position_embeddings = config.get("model_max_position_embeddings")
    if model_max_position_embeddings is not None:
        if int(model_max_position_embeddings) < max_length:
            raise ValueError(
                "model_max_position_embeddings must be >= max_length"
            )
    model_rope_scaling = config.get("model_rope_scaling")
    if model_rope_scaling is not None:
        if not isinstance(model_rope_scaling, dict):
            raise ValueError("model_rope_scaling must be a JSON object")
        rope_type = model_rope_scaling.get(
            "rope_type",
            model_rope_scaling.get("type"),
        )
        if rope_type == "yarn":
            factor = model_rope_scaling.get("factor")
            if not isinstance(factor, (int, float)) or float(factor) < 1.0:
                raise ValueError("model_rope_scaling.factor must be >= 1.0")
        elif rope_type == "hope":
            orig = int(
                model_rope_scaling.get("original_max_position_embeddings") or 0
            )
            if orig <= 0:
                raise ValueError(
                    "HoPE requires original_max_position_embeddings > 0 "
                    "(Dream-v0 pretrain length, usually 2048)"
                )
        else:
            raise ValueError(
                "model_rope_scaling supports rope_type=yarn or hope"
            )
    target_modules = list(
        config.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
    )
    allowed_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
    if not target_modules or not set(target_modules) <= allowed_targets:
        raise ValueError(
            "lora_target_modules must be a non-empty subset of "
            f"{sorted(allowed_targets)}"
        )
    if len(target_modules) != len(set(target_modules)):
        raise ValueError("lora_target_modules must not contain duplicates")
    lmk_token_mode = str(config.get("lmk_token_mode", "mask"))
    if lmk_token_mode not in {"mask", "external", "mask_type", "eos", "vocab"}:
        raise ValueError(f"unsupported lmk_token_mode={lmk_token_mode}")
    if lmk_token_mode == "vocab" and trainable_scope not in _LMK_SCOPES:
        raise ValueError("lmk_token_mode=vocab requires an LMK trainable scope")
    if attention_mode == "dsa":
        lmk_token_mode = "mask"
    ruler_mix_ratio = float(config.get("ruler_mix_ratio", 0.0))
    if not 0.0 <= ruler_mix_ratio <= 1.0:
        raise ValueError(
            f"ruler_mix_ratio must be between 0 and 1, got {ruler_mix_ratio}"
        )
    answer_ce_weight = float(config.get("ruler_answer_ce_weight", 1.0) or 1.0)
    if not math.isfinite(answer_ce_weight) or answer_ce_weight <= 0:
        raise ValueError("ruler_answer_ce_weight must be positive and finite")
    if answer_ce_weight > 256:
        raise ValueError(
            "ruler_answer_ce_weight > 256 is refused; start mild (e.g. 32), not 16k/8"
        )
    sync_ruler = bool(config.get("route_sync_ruler_ce", False))
    all_ruler_tasks = bool(config.get("sync_ruler_all_tasks", False))
    if sync_ruler and all_ruler_tasks:
        raise ValueError("route_sync_ruler_ce and sync_ruler_all_tasks are mutually exclusive")
    if (
        abs(answer_ce_weight - 1.0) > 1e-12
        and ruler_mix_ratio <= 0
        and not sync_ruler
        and not all_ruler_tasks
    ):
        raise ValueError("ruler_answer_ce_weight != 1 requires ruler_mix_ratio > 0")
    if (sync_ruler or all_ruler_tasks) and abs(answer_ce_weight - 1.0) > 1e-12:
        raise ValueError("sync RULER CE uses equal task means; keep ruler_answer_ce_weight=1")
    ruler_task_ids = [
        int(task_id) for task_id in config.get("ruler_task_ids", [0, 1, 2])
    ]
    if not ruler_task_ids:
        raise ValueError("ruler_task_ids must not be empty")
    unsupported_ruler = sorted(set(ruler_task_ids) - {0, 1, 2})
    if unsupported_ruler:
        raise ValueError(f"unsupported ruler_task_ids={unsupported_ruler}")
    if all_ruler_tasks and set(ruler_task_ids) != {0, 1, 2}:
        raise ValueError("sync_ruler_all_tasks requires ruler_task_ids=[0, 1, 2]")
    if bool(config.get("ruler_joint_local_mask", False)):
        if ruler_mix_ratio <= 0:
            raise ValueError("ruler_joint_local_mask requires ruler_mix_ratio > 0")
        if bool(config.get("route_balanced_view_ce", False)) or bool(
            config.get("route_asymmetric_gate_ce", False)
        ) or sync_ruler or all_ruler_tasks:
            raise ValueError("ruler_joint_local_mask cannot stack with dual-view CE")
    if sync_ruler or all_ruler_tasks:
        if config.get("attention_mode") not in {"route", "dense", "swa", "dsa", "nsa"}:
            raise ValueError("sync RULER CE supports attention_mode=route, dense, swa, dsa, or nsa")
        if config.get("attention_mode") == "route":
            if trainable_scope not in _LORA_SCOPES:
                raise ValueError("route sync RULER CE requires a LoRA Q-Cal scope")
            if bool(config.get("route_detach_fusion_weights", False)):
                raise ValueError("route sync RULER CE requires live fusion on both tasks")
        if dense_weight != 0:
            raise ValueError("do not stack sync RULER CE with dense chunk KL")
        if evidence_weight > 0:
            raise ValueError("do not stack sync RULER CE with evidence route CE")
        if float(config.get("route_support_attn_kl_weight", 0) or 0) > 0:
            raise ValueError("do not stack sync RULER CE with support attn KL")
        if bool(config.get("route_lmk_ce_ste", False)):
            raise ValueError("do not stack sync RULER CE with lmk_ce_ste")
        if ruler_mix_ratio != 0.0:
            raise ValueError("sync RULER CE keeps ruler_mix_ratio=0; RULER is a fixed view")
        if bool(config.get("route_asymmetric_gate_ce", False)) or bool(
            config.get("route_balanced_view_ce", False)
        ) or bool(config.get("route_force_remote_unit", False)):
            raise ValueError("sync RULER CE cannot stack with copy-view or force-remote CE")
    if int(config.get("loss_chunk_size", 2048)) <= 0:
        raise ValueError("loss_chunk_size must be positive")
    stop_after_steps = config.get("stop_after_steps")
    if stop_after_steps is not None and int(stop_after_steps) <= 0:
        raise ValueError("stop_after_steps must be positive when provided")
    if int(config.get("num_workers", 0)) != 0:
        raise ValueError("num_workers must be 0 for exact deterministic collator resume")
    for key in (
        "validation_packs",
        "validation_batches",
        "eval_steps",
        "overfit_packs",
        "ruler_val_batches",
    ):
        if int(config.get(key, 0)) < 0:
            raise ValueError(f"{key} must be non-negative")


@dataclass(frozen=True)
class TrainingAttentionPlan:
    mode: str
    route_layers: list[int]
    dsa_layers: list[int]
    sliding_window_layers: list[int]
    dense_layers: list[int]


def _training_resume_contract(args: argparse.Namespace) -> dict[str, object]:
    if args.attention_mode == "route":
        return {
            "attention_mode": "route",
            "chunk_distillation_version": "remote-per-query-head-kl-v1",
            "route_qcal_rank": int(getattr(args, "route_qcal_rank", 0)),
            "route_freeze_qcal": bool(getattr(args, "route_freeze_qcal", False)),
            "route_qcal_version": "residual-random-lowrank-rmsnorm-v1",
            "route_trainable_scope": str(
                getattr(args, "route_trainable_scope", "full")
            ),
            "route_chunk_summary": str(getattr(args, "route_chunk_summary", "attn")),
            "route_entropy_prior": bool(getattr(args, "route_entropy_prior", True)),
            "lmk_token_mode": str(getattr(args, "lmk_token_mode", "mask")),
            "route_evidence_route_loss_weight": float(
                getattr(args, "route_evidence_route_loss_weight", 0.0)
            ),
            "route_evidence_token_attn_weight": float(
                getattr(args, "route_evidence_token_attn_weight", 0.0)
            ),
            "route_evidence_token_attn_version": "selected-remote-needle-token-ce-v1",
            "route_support_attn_kl_weight": float(
                getattr(args, "route_support_attn_kl_weight", 0.0)
            ),
            "route_support_attn_queries": int(
                getattr(args, "route_support_attn_queries", 16)
            ),
            "route_support_attn_temperature": float(
                getattr(args, "route_support_attn_temperature", 1.5)
            ),
            "route_support_attn_query_chunk": int(
                getattr(args, "route_support_attn_query_chunk", 16)
            ),
            "route_support_attn_detach_gate": bool(
                getattr(args, "route_support_attn_detach_gate", False)
            ),
            "route_support_attn_version": "fused-support-mask-span-kl-v2",
            "route_evidence_route_loss_version": "fixed-evidence-qcal-only-v1",
            "route_chunk_aux_queries": int(args.route_chunk_aux_queries),
            "route_chunk_aux_loss_weight": float(args.route_chunk_aux_loss_weight),
            "route_dense_teacher_weight": float(args.route_dense_teacher_weight),
            "route_allchunk_st_queries": int(
                getattr(args, "route_allchunk_st_queries", 0)
            ),
            "route_allchunk_st_version": "hard-forward-exact-chunk-output-gate-jacobian-v1",
            "route_dense_teacher_rule": args.route_dense_teacher_rule,
            "route_dense_teacher_queries": args.route_dense_teacher_queries,
            "route_dense_teacher_temperature": args.route_dense_teacher_temperature,
            "route_dense_teacher_source": str(
                getattr(args, "route_dense_teacher_source", "same_forward")
            ),
            "route_asymmetric_gate_ce": bool(
                getattr(args, "route_asymmetric_gate_ce", False)
            ),
            "route_balanced_view_ce": bool(
                getattr(args, "route_balanced_view_ce", False)
            ),
            "route_sync_ruler_ce": bool(
                getattr(args, "route_sync_ruler_ce", False)
            ),
            "sync_ruler_all_tasks": bool(
                getattr(args, "sync_ruler_all_tasks", False)
            ),
            "route_force_remote_unit": bool(
                getattr(args, "route_force_remote_unit", False)
            ),
            "route_force_remote_oracle_route": bool(
                getattr(args, "route_force_remote_oracle_route", False)
            ),
            "route_gate_bce_weight": float(
                getattr(args, "route_gate_bce_weight", 0.0) or 0.0
            ),
            "route_gate_ce_force": bool(getattr(args, "route_gate_ce_force", False)),
            "route_qcal_max_grad_norm": float(
                getattr(args, "route_qcal_max_grad_norm", 0.0) or 0.0
            ),
            "full_dense_teacher_version": "same-forward-qk-qcal-only-kl-v1",
            "model_rope_scaling": args.model_rope_scaling,
        }
    if args.attention_mode == "nsa":
        return {
            "attention_mode": "nsa",
            "nsa_block_count": int(args.nsa_block_count),
            "nsa_compress_block": int(getattr(args, "nsa_compress_block", 32)),
            "nsa_compress_stride": int(getattr(args, "nsa_compress_stride", 16)),
            "nsa_select_block": int(getattr(args, "nsa_select_block", args.chunk_size)),
            "nsa_block_size": int(getattr(args, "nsa_select_block", args.chunk_size)),
            "nsa_backend": "tilelang",
        }
    if args.attention_mode != "dsa":
        if bool(getattr(args, "route_sync_ruler_ce", False)) or bool(
            getattr(args, "sync_ruler_all_tasks", False)
        ):
            return {
                "attention_mode": str(args.attention_mode),
                "route_sync_ruler_ce": bool(
                    getattr(args, "route_sync_ruler_ce", False)
                ),
                "sync_ruler_all_tasks": bool(
                    getattr(args, "sync_ruler_all_tasks", False)
                ),
            }
        return {}
    return {
        "attention_mode": "dsa",
        "dsa_backend": str(args.dsa_backend),
        "dsa_selected_attention_semantics": DSA_SELECTED_ATTENTION_SEMANTICS,
    }


class DenseDreamAttentionAdapter(torch.nn.Module):
    """Keep native Dream attention math while accepting decoder extras."""

    def __init__(self, source_attn: torch.nn.Module) -> None:
        super().__init__()
        self.source_attn = source_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        sparse_keep_indices=None,
        sparse_prompt_len=None,
        **kwargs,
    ):
        del sparse_keep_indices, sparse_prompt_len, kwargs
        return self.source_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )


def resolved_swa_local_window(args: argparse.Namespace) -> int:
    explicit = getattr(args, "swa_local_window", None)
    if explicit is None:
        return int(args.local_window)
    return int(explicit)


def configure_training_attention(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> TrainingAttentionPlan:
    if args.attention_mode == "dense":
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise TypeError("expected a DreamModel-like object with model.layers")
        dense_layers = list(range(len(model.model.layers)))
        for layer in model.model.layers:
            layer.self_attn = DenseDreamAttentionAdapter(layer.self_attn)
        model.config.training_attention_mode = "dense"
        return TrainingAttentionPlan(
            mode="dense",
            route_layers=[],
            dsa_layers=[],
            sliding_window_layers=[],
            dense_layers=dense_layers,
        )

    if args.attention_mode == "dsa":
        dsa_plan = install_dream_dsa_attention(
            model,
            interleave=args.route_interleave,
            local_window=resolved_swa_local_window(args),
            chunk_size=args.chunk_size,
            topk=args.dsa_topk,
            num_index_heads=args.dsa_index_heads,
            index_head_dim=args.dsa_index_head_dim,
            query_block_size=args.dsa_query_block_size,
            attention_query_block_size=args.dsa_attention_query_block_size,
            aux_queries=args.dsa_aux_queries,
            non_dsa_attention=args.non_route_attention,
            aux_loss_scope=getattr(args, "dsa_aux_loss_scope", "full"),
            backend=getattr(args, "dsa_backend", "torch"),
        )
        for layer_idx in dsa_plan.dense_layers:
            layer = model.model.layers[layer_idx]
            layer.self_attn = DenseDreamAttentionAdapter(layer.self_attn)
        model.config.training_attention_mode = "dsa"
        return TrainingAttentionPlan(
            mode="dsa",
            route_layers=[],
            dsa_layers=list(dsa_plan.dsa_layers),
            sliding_window_layers=list(dsa_plan.sliding_window_layers),
            dense_layers=list(dsa_plan.dense_layers),
        )

    if args.attention_mode == "nsa":
        nsa_plan = install_dream_nsa_attention(
            model,
            interleave=args.route_interleave,
            local_window=args.local_window,
            swa_local_window=resolved_swa_local_window(args),
            chunk_size=args.chunk_size,
            block_count=args.nsa_block_count,
            non_nsa_attention=args.non_route_attention,
            compress_block=getattr(args, "nsa_compress_block", 32),
            compress_stride=getattr(args, "nsa_compress_stride", 16),
            select_block=getattr(args, "nsa_select_block", args.chunk_size),
            skip_inert_slots=False,
        )
        for layer_idx in nsa_plan.dense_layers:
            layer = model.model.layers[layer_idx]
            layer.self_attn = DenseDreamAttentionAdapter(layer.self_attn)
        model.config.training_attention_mode = "nsa"
        return TrainingAttentionPlan(
            mode="nsa",
            route_layers=[],
            dsa_layers=[],
            sliding_window_layers=list(nsa_plan.sliding_window_layers),
            dense_layers=list(nsa_plan.dense_layers),
        )

    sparse_plan = install_dream_sparse_attention(
        model,
        interleave=args.route_interleave,
        local_window=args.local_window,
        swa_local_window=resolved_swa_local_window(args),
        chunk_size=args.chunk_size,
        topk=args.route_topk,
        token_budget=getattr(args, "route_token_budget", 0),
        token_policy=getattr(args, "route_token_policy", "global_qk"),
        min_tokens_per_chunk=getattr(
            args, "route_min_tokens_per_chunk", 1
        ),
        token_relaxation=getattr(args, "route_token_relaxation", "none"),
        token_gumbel_scale=getattr(args, "route_token_gumbel_scale", 1.0),
        route_relaxation=getattr(args, "route_relaxation", "none"),
        route_temperature=getattr(args, "route_temperature", 1.0),
        route_gumbel_scale=getattr(args, "route_gumbel_scale", 1.0),
        route_selection_mode=getattr(args, "route_selection_mode", "post_softmax"),
        route_residual_weight=getattr(args, "route_residual_weight", 0.0),
        detach_fusion_weights=bool(getattr(args, "route_detach_fusion_weights", False)),
        value_fusion_beta=float(getattr(args, "route_value_fusion_beta", 0.0) or 0.0),
        value_fusion_rank=int(getattr(args, "route_value_fusion_rank", 32) or 32),
        value_fusion_query_block=int(
            getattr(args, "route_value_fusion_query_block", 128) or 128
        ),
        backend=args.route_backend,
        allow_kernel_fallback=not args.no_kernel_fallback,
        non_route_attention=args.non_route_attention,
        chunk_summary=str(getattr(args, "route_chunk_summary", "attn")),
        entropy_prior=bool(getattr(args, "route_entropy_prior", True)),
    )
    for layer_idx in sparse_plan.dense_layers:
        layer = model.model.layers[layer_idx]
        layer.self_attn = DenseDreamAttentionAdapter(layer.self_attn)
    from prefilling_v2.qcal import install_qcal
    install_qcal(model, int(getattr(args, "route_qcal_rank", 0)))
    teacher_source = str(getattr(args, "route_dense_teacher_source", "same_forward"))
    for layer in model.model.layers:
        if layer.self_attn.__class__.__name__ == "KernelDreamFullRouteAttention":
            layer.self_attn.full_teacher_source = teacher_source
            layer.self_attn.lmk_kl_ste = bool(getattr(args, "route_lmk_kl_ste", True))
            layer.self_attn.lmk_ce_ste = bool(getattr(args, "route_lmk_ce_ste", False))
    model.config.training_attention_mode = "route"
    return TrainingAttentionPlan(
        mode="route",
        route_layers=list(sparse_plan.route_layers),
        dsa_layers=[],
        sliding_window_layers=list(sparse_plan.sliding_window_layers),
        dense_layers=list(sparse_plan.dense_layers),
    )


class ResumableDataIterator:
    def __init__(
        self,
        loader: DataLoader,
        sampler: DistributedSampler,
        collator,
        *,
        epoch: int = 0,
        batches_in_epoch: int = 0,
    ) -> None:
        self.loader = loader
        self.sampler = sampler
        self.collator = collator
        self.epoch = int(epoch)
        self.batches_in_epoch = 0
        self._reset_iterator()
        for _ in range(int(batches_in_epoch)):
            try:
                next(self._iterator)
                self.batches_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self._reset_iterator()

    def _reset_iterator(self) -> None:
        self.sampler.set_epoch(self.epoch)
        if hasattr(self.collator, "set_epoch"):
            self.collator.set_epoch(self.epoch)
        self._iterator = iter(self.loader)
        self.batches_in_epoch = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self._reset_iterator()
            batch = next(self._iterator)
        self.batches_in_epoch += 1
        return batch


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


@contextlib.contextmanager
def _preserve_rng_state(enabled: bool):
    state = capture_rng_state() if enabled else None
    try:
        yield
    finally:
        if state is not None:
            restore_rng_state(state)


def apply_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if int(getattr(args, "lora_r", 0) or 0) <= 0:
        return model
    from peft import LoraConfig, TaskType, get_peft_model

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=tuple(args.lora_target_modules),
        ),
    )
    entropy_parameters = 0
    dsa_indexer_parameters = 0
    for name, parameter in model.named_parameters():
        if _is_qcal_param(name):
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
        if "entropy_bias_scale" in name:
            parameter.requires_grad_(True)
            entropy_parameters += parameter.numel()
        if ".dsa_indexer." in name:
            parameter.requires_grad_(True)
            dsa_indexer_parameters += parameter.numel()
    if args.attention_mode == "route" and entropy_parameters == 0:
        raise RuntimeError("route installation produced no entropy_bias_scale parameters")
    if args.attention_mode == "dense" and entropy_parameters:
        raise RuntimeError("dense control unexpectedly contains route entropy parameters")
    if args.attention_mode == "dsa" and dsa_indexer_parameters == 0:
        raise RuntimeError("Dream DSA installation produced no indexer parameters")
    if args.attention_mode != "dsa" and dsa_indexer_parameters:
        raise RuntimeError("non-DSA model unexpectedly contains DSA indexer parameters")
    return model


def resolve_landmark_token_id(
    args: argparse.Namespace,
    tokenizer,
    model: torch.nn.Module | None = None,
) -> int:
    mode = str(getattr(args, "lmk_token_mode", "mask"))
    if mode in {"mask", "mask_type"}:
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("Dream tokenizer has no mask_token_id")
        return int(mask_token_id)
    if mode == "eos":
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("Dream tokenizer has no eos_token_id")
        return int(eos_token_id)
    if mode == "vocab":
        if hasattr(args, "vocab_lmk_token_id"):
            return int(args.vocab_lmk_token_id)
        token = str(getattr(args, "route_token", None) or "<|route|>")
        token_id = tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if token_id is None or int(token_id) < 0 or (
            unk_id is not None and int(token_id) == int(unk_id) and token not in tokenizer.get_vocab()
        ):
            raise ValueError(
                f"vocab route token {token!r} is not in the tokenizer; "
                "install_vocab_lmk_embedding must run first"
            )
        return int(token_id)
    if mode != "external":
        raise ValueError(f"unsupported lmk_token_mode={mode}")
    if hasattr(args, "external_lmk_token_id"):
        return int(args.external_lmk_token_id)
    if model is not None:
        return int(model.get_input_embeddings().num_embeddings)
    return int(getattr(tokenizer, "vocab_size"))


def install_external_lmk_embedding(
    model: torch.nn.Module,
    args: argparse.Namespace,
    tokenizer,
) -> int:
    """Add a standalone trainable embedding for input-only LMK slots."""

    lmk_token_id = resolve_landmark_token_id(args, tokenizer, model)
    args.external_lmk_token_id = int(lmk_token_id)
    model.config.dream_route_lmk_token_mode = "external"
    model.config.dream_route_lmk_token_id = int(lmk_token_id)

    hidden_size = int(getattr(model.config, "hidden_size"))
    std = float(getattr(model.config, "initializer_range", 0.02))
    lmk_embed = torch.nn.Parameter(torch.empty(hidden_size, dtype=torch.float32))
    torch.nn.init.normal_(lmk_embed, mean=0.0, std=std)
    model.register_parameter("dream_route_lmk_embed", lmk_embed)

    original_forward = model.forward

    def forward_with_external_lmk(
        input_ids=None,
        *args_forward,
        inputs_embeds=None,
        **kwargs,
    ):
        if (
            input_ids is not None
            and inputs_embeds is None
            and torch.is_tensor(input_ids)
        ):
            lmk_mask = input_ids.eq(int(lmk_token_id))
            if bool(lmk_mask.any()):
                safe_ids = input_ids.masked_fill(lmk_mask, 0)
                token_embeds = model.get_input_embeddings()(safe_ids)
                replacement = model.dream_route_lmk_embed.to(token_embeds.dtype)
                inputs_embeds = torch.where(
                    lmk_mask.unsqueeze(-1),
                    replacement.view(1, 1, -1),
                    token_embeds,
                )
                input_ids = None
        return original_forward(
            input_ids,
            *args_forward,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    model.forward = forward_with_external_lmk
    return int(lmk_token_id)


def _patch_embedding_trainable_vocab_row(
    embed: torch.nn.Module,
    token_id: int,
    row: torch.nn.Parameter,
) -> None:
    """Look up `token_id` from a trainable vocab row without optimizing the full table."""

    tid = int(token_id)

    def forward(input_ids):
        weight = embed.weight
        if tid < 0 or tid >= int(weight.shape[0]):
            raise RuntimeError(
                f"vocab route id {tid} is outside embedding rows {int(weight.shape[0])}"
            )
        row_vec = row.to(device=weight.device, dtype=weight.dtype)
        mixed = torch.cat(
            (
                weight[:tid].detach(),
                row_vec.unsqueeze(0),
                weight[tid + 1 :].detach(),
            ),
            dim=0,
        )
        with torch.no_grad():
            weight[tid].copy_(row_vec)
        return torch.nn.functional.embedding(
            input_ids,
            mixed,
            padding_idx=embed.padding_idx,
            max_norm=embed.max_norm,
            norm_type=embed.norm_type,
            scale_grad_by_freq=embed.scale_grad_by_freq,
            sparse=embed.sparse,
        )

    embed.forward = forward


def install_vocab_lmk_embedding(
    model: torch.nn.Module,
    args: argparse.Namespace,
    tokenizer,
) -> int:
    """Add `<|route|>` to the tokenizer/embedding table and train that row."""

    token = str(getattr(args, "route_token", None) or "<|route|>")
    vocab = tokenizer.get_vocab()
    if token not in vocab:
        added = tokenizer.add_tokens([token], special_tokens=True)
        if int(added) != 1:
            raise RuntimeError(
                f"failed to add vocab route token {token!r} (added={added})"
            )
    token_id = int(tokenizer.convert_tokens_to_ids(token))
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if token_id < 0 or (unk_id is not None and token_id == int(unk_id) and token not in tokenizer.get_vocab()):
        raise RuntimeError(f"vocab route token {token!r} resolved to an invalid id {token_id}")

    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    embed = core.get_input_embeddings()
    needed = max(int(len(tokenizer)), token_id + 1)
    if int(embed.num_embeddings) < needed:
        resizer = None
        if hasattr(model, "resize_token_embeddings"):
            resizer = model.resize_token_embeddings
        elif hasattr(core, "resize_token_embeddings"):
            resizer = core.resize_token_embeddings
        if resizer is None:
            raise RuntimeError("model cannot resize_token_embeddings for vocab route token")
        resizer(needed)
        embed = core.get_input_embeddings()
    if int(embed.num_embeddings) <= token_id:
        raise RuntimeError(
            f"embedding rows {int(embed.num_embeddings)} do not cover route id {token_id}"
        )

    std = float(getattr(getattr(core, "config", None), "initializer_range", 0.02) or 0.02)
    row = torch.nn.Parameter(embed.weight.data[token_id].detach().float().clone())
    if not bool(torch.isfinite(row).all()) or float(row.detach().abs().sum()) == 0.0:
        torch.nn.init.normal_(row, mean=0.0, std=std)
    with torch.no_grad():
        embed.weight[token_id].copy_(row.to(dtype=embed.weight.dtype))
    model.register_parameter("dream_route_lmk_embed", row)
    _patch_embedding_trainable_vocab_row(embed, token_id, row)

    args.vocab_lmk_token_id = int(token_id)
    model.config.dream_route_lmk_token_mode = "vocab"
    model.config.dream_route_lmk_token_id = int(token_id)
    model.config.dream_route_token = token
    if hasattr(model.config, "vocab_size"):
        model.config.vocab_size = int(embed.num_embeddings)
    return int(token_id)


def install_mask_lmk_type_embedding(
    model: torch.nn.Module,
    args: argparse.Namespace,
    tokenizer,
) -> int:
    """Add a trainable role embedding to native MASK landmark slots."""

    lmk_token_id = resolve_landmark_token_id(args, tokenizer, model)
    model.config.dream_route_lmk_token_mode = "mask_type"
    model.config.dream_route_lmk_token_id = int(lmk_token_id)

    hidden_size = int(getattr(model.config, "hidden_size"))
    std = float(getattr(model.config, "initializer_range", 0.02))
    type_embed = torch.nn.Parameter(torch.empty(hidden_size, dtype=torch.float32))
    torch.nn.init.normal_(type_embed, mean=0.0, std=std)
    model.register_parameter("dream_route_lmk_type_embed", type_embed)
    _bind_lmk_type_embed(model, type_embed)
    return int(lmk_token_id)


def _bind_lmk_type_embed(model: torch.nn.Module, type_embed: torch.nn.Parameter) -> None:
    """Point route layers at the shared type embed so dense KL can STE it."""
    from prefilling_v2.attention import KernelDreamFullRouteAttention

    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = core.model.layers if hasattr(core, "model") else core.layers
    bound = 0
    for layer in layers:
        attn = layer.self_attn
        if isinstance(attn, KernelDreamFullRouteAttention):
            attn.lmk_type_embed = type_embed
            bound += 1
    if bound == 0:
        raise RuntimeError("mask_type LMK requires route layers to bind type embed")


def configure_landmark_embedding(
    model: torch.nn.Module,
    args: argparse.Namespace,
    tokenizer,
) -> int:
    mode = str(getattr(args, "lmk_token_mode", "mask"))
    if mode in {"mask", "eos"}:
        lmk_token_id = resolve_landmark_token_id(args, tokenizer, model)
        model.config.dream_route_lmk_token_mode = mode
        model.config.dream_route_lmk_token_id = int(lmk_token_id)
        return int(lmk_token_id)
    if mode == "mask_type":
        return install_mask_lmk_type_embedding(model, args, tokenizer)
    if mode == "external":
        return install_external_lmk_embedding(model, args, tokenizer)
    if mode == "vocab":
        return install_vocab_lmk_embedding(model, args, tokenizer)
    raise ValueError(f"unsupported lmk_token_mode={mode}")


def _zero_join_unused_qcal(model: torch.nn.Module, loss_sum: torch.Tensor) -> torch.Tensor:
    """Keep detached Q-Cal in the DDP reduction with a numerically zero join."""

    extra = loss_sum.new_zeros(())
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and _is_qcal_param(name):
            extra = extra + parameter.float().sum() * 0.0
    return extra


def _zero_join_lmk_type_embed(
    model: torch.nn.Module, loss_sum: torch.Tensor
) -> torch.Tensor:
    """Keep detached type-embed in the DDP graph with a numerically zero join."""

    unwrapped = model.module if hasattr(model, "module") else model
    parameter = getattr(unwrapped, "dream_route_lmk_type_embed", None)
    if parameter is None or not parameter.requires_grad:
        return loss_sum
    return loss_sum + parameter.float().sum() * 0.0


def audit_frozen_gradients(model: torch.nn.Module) -> None:
    unwrapped = model.module if hasattr(model, "module") else model
    offenders = [
        name
        for name, parameter in unwrapped.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if offenders:
        raise RuntimeError(f"frozen parameters received gradients: {offenders[:8]}")


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


_LMK_PARAM_NAMES = {"dream_route_lmk_type_embed", "dream_route_lmk_embed"}


def _is_qcal_param(name: str) -> bool:
    return ".qcal." in name or ".qcal_norm." in name


def _is_lmk_param(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in _LMK_PARAM_NAMES


def initialize_and_configure_trainables(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> list[str]:
    """Load the common source state, then apply the requested freeze scope."""

    scope = str(getattr(args, "route_trainable_scope", "full"))
    loaded: list[str] = []
    if args.initialize_from:
        loaded = load_trainable_checkpoint(
            checkpoint_dir=args.initialize_from,
            model=model,
            allow_partial=True,
        )
    if scope == "full":
        return loaded
    if scope not in _SCOPED_TRAINABLE:
        raise ValueError(f"unsupported trainable scope: {scope}")
    for name, parameter in model.named_parameters():
        keep = _is_qcal_param(name)
        if scope in _LORA_SCOPES:
            keep = keep or "lora_" in name
        if scope in _LMK_SCOPES:
            keep = keep or _is_lmk_param(name)
        if float(getattr(args, "route_value_fusion_beta", 0.0) or 0.0) > 0:
            keep = keep or "value_fusion" in name
        if name.endswith("fusion_gate_offset"):
            keep = bool(getattr(args, "route_gate_offset_only", False)) or bool(
                getattr(args, "route_gate_affine_only", False)
            ) or bool(getattr(args, "route_train_gate_offset", False))
        if name.endswith("fusion_gate_scale"):
            keep = bool(getattr(args, "route_gate_affine_only", False))
        parameter.requires_grad_(keep)
    if bool(getattr(args, "route_freeze_qcal", False)):
        for name, parameter in model.named_parameters():
            if _is_qcal_param(name):
                parameter.requires_grad_(False)
    if bool(getattr(args, "route_lora_q_only", False)):
        for name, parameter in model.named_parameters():
            if "lora_" in name and ".q_proj." not in name:
                parameter.requires_grad_(False)
    if bool(getattr(args, "route_gate_offset_only", False)):
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.endswith("fusion_gate_offset"))
    if bool(getattr(args, "route_gate_affine_only", False)):
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.endswith("fusion_gate_offset") or name.endswith("fusion_gate_scale")
            )
    trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if bool(getattr(args, "route_gate_offset_only", False)):
        if not trainable or any(
            not name.endswith("fusion_gate_offset") for name in trainable
        ):
            raise RuntimeError(f"offset-only expected fusion_gate_offset, got {trainable[:8]}")
        return loaded
    if bool(getattr(args, "route_gate_affine_only", False)):
        if not trainable or any(
            not (
                name.endswith("fusion_gate_offset") or name.endswith("fusion_gate_scale")
            )
            for name in trainable
        ):
            raise RuntimeError(f"affine-only expected gate scale/offset, got {trainable[:8]}")
        if not any(name.endswith("fusion_gate_scale") for name in trainable):
            raise RuntimeError("affine-only missing fusion_gate_scale")
        cal_init = getattr(args, "route_gate_cal_init_from", None)
        if cal_init:
            from prefilling_v2.checkpointing import overlay_checkpoint_parameters

            overlaid = overlay_checkpoint_parameters(
                checkpoint_dir=cal_init,
                model=model,
                name_predicate=lambda name: name.endswith("fusion_gate_offset")
                or name.endswith("fusion_gate_scale"),
            )
            loaded.extend(overlaid)
        offset_layers = _as_int_list(getattr(args, "route_gate_train_offset_layers", None))
        scale_layers = _as_int_list(getattr(args, "route_gate_train_scale_layers", None))
        if offset_layers is not None or scale_layers is not None:
            offset_params = _gate_cal_params(model, "fusion_gate_offset")
            scale_params = _gate_cal_params(model, "fusion_gate_scale")
            if offset_layers is None:
                offset_layers = list(range(len(offset_params)))
            if scale_layers is None:
                scale_layers = list(range(len(scale_params)))
            offset_set = set(offset_layers)
            scale_set = set(scale_layers)
            for index, (_, parameter) in enumerate(offset_params):
                parameter.requires_grad_(index in offset_set)
            for index, (_, parameter) in enumerate(scale_params):
                parameter.requires_grad_(index in scale_set)
            trainable = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]
            if not trainable:
                raise RuntimeError("layer-wise affine freeze left no trainables")
        return loaded
    if (
        scope in _QCAL_SCOPES
        and not bool(getattr(args, "route_freeze_qcal", False))
        and not any(_is_qcal_param(name) for name in trainable)
    ):
        raise RuntimeError(f"missing Q-Cal trainables: {trainable[:8]}")
    if bool(getattr(args, "route_freeze_qcal", False)):
        if any(_is_qcal_param(name) for name in trainable):
            raise RuntimeError(f"freeze_qcal still has Q-Cal trainables: {trainable[:8]}")
        if not trainable:
            raise RuntimeError("freeze_qcal left no trainables")
    if scope in _LMK_SCOPES and not any(_is_lmk_param(name) for name in trainable):
        raise RuntimeError(f"missing landmark embedding trainables: {trainable[:8]}")
    allow_value_fusion = float(getattr(args, "route_value_fusion_beta", 0.0) or 0.0) > 0
    illegal = [
        name
        for name in trainable
        if not _is_qcal_param(name)
        and not (scope in _LORA_SCOPES and "lora_" in name)
        and not (scope in _LMK_SCOPES and _is_lmk_param(name))
        and not name.endswith("fusion_gate_offset")
        and not name.endswith("fusion_gate_scale")
        and not (allow_value_fusion and "value_fusion" in name)
    ]
    if bool(getattr(args, "route_lora_q_only", False)):
        illegal.extend(
            name
            for name in trainable
            if "lora_" in name and ".q_proj." not in name
        )
    if illegal:
        raise RuntimeError(f"invalid {scope} trainables: {illegal[:8]}")
    return loaded


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return {
        name: value.to(device=device, non_blocking=True)
        for name, value in batch.items()
    }


def _kernel_fallback_count(model: torch.nn.Module) -> int:
    unwrapped = model.module if hasattr(model, "module") else model
    return sum(
        int(bool(getattr(module, "_warned_kernel_fallback", False)))
        for module in unwrapped.modules()
    )


class _DenoisingLossWrapper(torch.nn.Module):
    """Make the memory-bounded Dream loss the actual DDP forward."""

    def __init__(self, model: torch.nn.Module, args: argparse.Namespace) -> None:
        super().__init__()
        self.model = model
        self.args = args

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _denoising_loss_sum_for_unwrapped_model(
            self.model,
            self.args,
            batch,
        )


def _unwrap_parallel_training_model(model: torch.nn.Module) -> torch.nn.Module:
    unwrapped = model.module if hasattr(model, "module") else model
    if isinstance(unwrapped, _DenoisingLossWrapper):
        return unwrapped.model
    return unwrapped


def _unwrap_training_model(model: torch.nn.Module) -> torch.nn.Module:
    unwrapped = _unwrap_parallel_training_model(model)
    if hasattr(unwrapped, "get_base_model"):
        return unwrapped.get_base_model()
    return unwrapped


def _landmark_inputs_embeds(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    landmark_mask: torch.Tensor | None = None,
    *,
    detach_lmk_type: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    unwrapped = model.module if hasattr(model, "module") else model
    config = getattr(unwrapped, "config", None)
    mode = getattr(config, "dream_route_lmk_token_mode", "mask")
    if mode not in {"external", "mask_type"}:
        return input_ids, None
    lmk_token_id = int(getattr(config, "dream_route_lmk_token_id"))
    if mode == "external":
        lmk_mask = input_ids.eq(lmk_token_id)
    else:
        if landmark_mask is None:
            raise RuntimeError("mask_type LMK mode requires landmark_mask in the batch")
        if landmark_mask.shape != input_ids.shape:
            raise ValueError("landmark_mask must match input_ids shape")
        lmk_mask = landmark_mask.bool()
        if torch.any(lmk_mask & input_ids.ne(lmk_token_id)):
            raise ValueError("landmark_mask marks positions that are not LMK tokens")
    if not bool(lmk_mask.any()):
        return input_ids, None
    if mode == "external":
        if not hasattr(unwrapped, "dream_route_lmk_embed"):
            raise RuntimeError("external LMK mode is missing dream_route_lmk_embed")
        safe_ids = input_ids.masked_fill(lmk_mask, 0)
        token_embeds = unwrapped.get_input_embeddings()(safe_ids)
        replacement = unwrapped.dream_route_lmk_embed.to(token_embeds.dtype)
        inputs_embeds = torch.where(
            lmk_mask.unsqueeze(-1),
            replacement.view(1, 1, -1),
            token_embeds,
        )
    else:
        if not hasattr(unwrapped, "dream_route_lmk_type_embed"):
            raise RuntimeError("mask_type LMK mode is missing dream_route_lmk_type_embed")
        token_embeds = unwrapped.get_input_embeddings()(input_ids)
        offset = unwrapped.dream_route_lmk_type_embed
        # Force-remote dumps the labeled query onto remote chunks. The shared
        # type offset is added to every landmark (~256 slots) and overflowed
        # in bf16 on the first backward. Detach it for this unit test.
        if bool(detach_lmk_type):
            offset = offset.detach()
        offset = offset.to(token_embeds.dtype)
        inputs_embeds = token_embeds + (
            lmk_mask.unsqueeze(-1).to(token_embeds.dtype)
            * offset.view(1, 1, -1)
        )
    return None, inputs_embeds


def _ruler_view_mask(view_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
    ids = view_ids.reshape(-1)
    if ids.numel() != batch_size:
        raise ValueError("view_ids must have one entry per batch row")
    return ids >= int(RULER_VIEW_ID_BASE)


def _supervised_token_weights(
    batch: dict[str, torch.Tensor],
    selected: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-supervised-token CE weights and a boolean RULER mask.

    α=1 or missing view_ids keeps every token at weight 1.
    """

    n_selected = int(selected.sum().item())
    device = selected.device
    ones = torch.ones((n_selected,), device=device, dtype=torch.float32)
    none_ruler = torch.zeros((n_selected,), device=device, dtype=torch.bool)
    if n_selected == 0 or "view_ids" not in batch:
        return ones, none_ruler
    labels = batch["labels"]
    ruler_rows = _ruler_view_mask(batch["view_ids"], labels.shape[0])
    token_ruler = ruler_rows.view(-1, *([1] * (labels.ndim - 1))).expand_as(labels)
    selected_ruler = token_ruler[selected]
    if abs(float(alpha) - 1.0) <= 1e-12:
        return ones, selected_ruler
    weights = torch.where(
        selected_ruler,
        torch.full((), float(alpha), device=device, dtype=torch.float32),
        ones,
    )
    return weights, selected_ruler


def weighted_supervised_count(
    batch: dict[str, torch.Tensor],
    alpha: float,
) -> float:
    labels = batch["labels"]
    selected = labels.ne(-100)
    raw = float(selected.sum().item())
    if abs(float(alpha) - 1.0) <= 1e-12 or "view_ids" not in batch:
        return raw
    ruler_rows = _ruler_view_mask(batch["view_ids"], labels.shape[0])
    token_ruler = ruler_rows.view(-1, *([1] * (labels.ndim - 1))).expand_as(labels)
    n_ruler = float(token_ruler[selected].sum().item())
    return (raw - n_ruler) + float(alpha) * n_ruler


def _denoising_loss_sum_for_unwrapped_model(
    model: torch.nn.Module,
    args: argparse.Namespace,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch["labels"]
    selected = labels.ne(-100)
    target_count = selected.sum(dtype=torch.long)
    if int(target_count.item()) <= 0:
        raise RuntimeError("batch contains no supervised tokens")

    base_model = _unwrap_training_model(model)
    train_root = model.module if hasattr(model, "module") else model
    if base_model.training and int(getattr(args, "route_allchunk_st_queries", 0)) > 0:
        from prefilling_v2.allchunk_gumbel import prepare_allchunk_queries
        prepare_allchunk_queries(base_model, batch, args.route_allchunk_st_queries)
    dense_layers = None
    if base_model.training and float(getattr(args, "route_dense_teacher_weight", 0)) > 0:
        from prefilling_v2.full_dense_teacher import prepare_dense_teacher
        dense_layers = prepare_dense_teacher(train_root, batch, args)
    input_ids, inputs_embeds = _landmark_inputs_embeds(
        model,
        batch["input_ids"],
        batch.get("landmark_mask"),
        detach_lmk_type=bool(getattr(args, "route_force_remote_unit", False)),
    )
    collect_dsa_loss = (
        args.attention_mode == "dsa"
        and base_model.training
        and float(args.dsa_aux_loss_weight) > 0
    )
    if collect_dsa_loss:
        prepared = prepare_dsa_index_losses(base_model)
        if prepared <= 0:
            raise RuntimeError("Dream DSA mode has no DSA attention layers")
    chunk_layers = None
    if base_model.training and float(getattr(args, "route_chunk_aux_loss_weight", 0)) > 0:
        from prefilling_v2.chunk_distillation import prepare_chunk_losses
        chunk_layers = prepare_chunk_losses(base_model, args.route_chunk_aux_queries)
    evidence_route_layers = None
    evidence_route_weight = float(
        getattr(args, "route_evidence_route_loss_weight", 0.0)
    )
    if base_model.training and evidence_route_weight > 0:
        from prefilling_v2.evidence_routing import (
            prepare_evidence_route_losses,
        )

        if "route_evidence_chunks" not in batch:
            raise RuntimeError("batch is missing fixed evidence chunk labels")
        evidence_route_layers = prepare_evidence_route_losses(
            base_model,
            selected,
            batch["route_evidence_chunks"],
        )
    evidence_token_layers = None
    evidence_token_weight = float(
        getattr(args, "route_evidence_token_attn_weight", 0.0) or 0.0
    )
    if base_model.training and evidence_token_weight > 0:
        from prefilling_v2.evidence_token_attn import prepare_evidence_token_attn

        if "route_evidence_tokens" not in batch:
            raise RuntimeError("batch is missing evidence token labels")
        has_evidence = batch["route_evidence_tokens"].any(dim=-1, keepdim=True)
        evidence_query_mask = batch["labels"].ne(-100) & has_evidence
        evidence_token_layers = prepare_evidence_token_attn(
            base_model,
            evidence_query_mask,
            batch["route_evidence_tokens"],
        )
    support_attn_layers = None
    support_attn_weight = float(getattr(args, "route_support_attn_kl_weight", 0.0) or 0.0)
    if base_model.training and support_attn_weight > 0:
        from prefilling_v2.support_attn_distill import prepare_support_attn_distill

        support_attn_layers = prepare_support_attn_distill(train_root, batch, args)
    if bool(getattr(args, "route_force_remote_unit", False)):
        from prefilling_v2.force_remote import prepare_force_remote

        prepare_force_remote(
            base_model,
            batch["labels"],
            ablation=str(getattr(base_model.config, "route_remote_ablation", "none")),
            seed=int(getattr(base_model.config, "route_remote_ablation_seed", 0) or 0),
            evidence_chunks=batch.get("route_evidence_chunks"),
            oracle_route=bool(
                getattr(args, "route_force_remote_oracle_route", False)
            ),
            supervise_gate=float(getattr(args, "route_gate_bce_weight", 0.0) or 0.0)
            > 0,
            gate_ce_force=bool(getattr(args, "route_gate_ce_force", False)),
        )
    outputs = base_model.model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=_attention_mask_for_model(
            args,
            batch["attention_mask"],
            batch["segment_ids"],
        ),
        position_ids=batch["position_ids"],
        use_cache=False,
        return_dict=True,
    )
    selected_hidden = outputs.last_hidden_state[selected]
    selected_labels = labels[selected]
    answer_alpha = float(getattr(args, "ruler_answer_ce_weight", 1.0) or 1.0)
    token_weights, selected_ruler = _supervised_token_weights(
        batch, selected, answer_alpha
    )
    raw_count = selected.sum(dtype=torch.long)
    weighted_count = token_weights.sum().to(dtype=torch.float64)
    lm_loss_weight = (
        float(getattr(args, "dsa_lm_loss_weight", 1.0))
        if args.attention_mode == "dsa"
        else 1.0
    )
    dolma_ce_sum = selected_hidden.new_zeros((), dtype=torch.float32)
    ruler_ce_sum = selected_hidden.new_zeros((), dtype=torch.float32)
    if lm_loss_weight > 0:
        loss_sum = selected_hidden.new_zeros((), dtype=torch.float32)
        chunk_size = int(getattr(args, "loss_chunk_size", 2048))
        weight_cursor = 0
        for start in range(0, selected_labels.numel(), chunk_size):
            end = min(start + chunk_size, selected_labels.numel())
            logits = base_model.lm_head(selected_hidden[start:end])
            token_ce = F.cross_entropy(
                logits,
                selected_labels[start:end],
                reduction="none",
            ).float()
            chunk_w = token_weights[start:end].to(device=token_ce.device)
            chunk_ruler = selected_ruler[start:end]
            loss_sum = loss_sum + (token_ce * chunk_w).sum()
            if bool(chunk_ruler.any()):
                ruler_ce_sum = ruler_ce_sum + token_ce[chunk_ruler].sum()
            if bool((~chunk_ruler).any()):
                dolma_ce_sum = dolma_ce_sum + token_ce[~chunk_ruler].sum()
            weight_cursor = end
        if weight_cursor != token_weights.numel():
            raise RuntimeError("supervised CE weights drifted from selected tokens")
        loss_sum = loss_sum * lm_loss_weight
    else:
        # Keep every LoRA path in the DDP graph while freezing it numerically.
        loss_sum = selected_hidden.float().sum() * 0.0
    base_model._last_dolma_ce_sum = float(dolma_ce_sum.detach().item())
    base_model._last_ruler_ce_sum = float(ruler_ce_sum.detach().item())
    base_model._last_dolma_targets = int((~selected_ruler).sum().item())
    base_model._last_ruler_targets = int(selected_ruler.sum().item())
    if "view_ids" in batch:
        base_model._last_ruler_views = int(
            _ruler_view_mask(batch["view_ids"], labels.shape[0]).sum().item()
        )
    else:
        base_model._last_ruler_views = 0
    if collect_dsa_loss:
        index_loss = collect_dsa_index_loss(base_model)
        if not torch.isfinite(index_loss):
            raise FloatingPointError(f"non-finite DSA index loss: {index_loss}")
        base_model._last_dsa_index_loss = float(index_loss.detach().item())
        if bool(getattr(args, "dsa_aux_scale_by_targets", True)):
            index_term = index_loss * raw_count.to(dtype=index_loss.dtype)
        else:
            index_term = index_loss
        loss_sum = loss_sum + (
            float(args.dsa_aux_loss_weight) * index_term
        )
    if chunk_layers is not None:
        from prefilling_v2.chunk_distillation import collect_chunk_loss
        chunk_loss = collect_chunk_loss(chunk_layers)
        if not torch.isfinite(chunk_loss):
            raise FloatingPointError(f"non-finite chunk KL: {chunk_loss}")
        base_model._last_route_chunk_loss = float(chunk_loss.detach())
        loss_sum = loss_sum + args.route_chunk_aux_loss_weight * chunk_loss * raw_count.to(chunk_loss.dtype)
    if evidence_route_layers is not None:
        from prefilling_v2.evidence_routing import collect_evidence_route_loss

        route_loss = collect_evidence_route_loss(evidence_route_layers)
        if not torch.isfinite(route_loss):
            raise FloatingPointError(
                f"non-finite evidence route loss: {route_loss}"
            )
        base_model._last_route_evidence_route_loss = float(route_loss.detach())
        loss_sum = loss_sum + (
            evidence_route_weight
            * route_loss
            * raw_count.to(route_loss.dtype)
        )
    if evidence_token_layers is not None:
        from prefilling_v2.evidence_token_attn import (
            collect_evidence_token_attn_loss,
        )

        token_attn_loss, token_attn_stats = collect_evidence_token_attn_loss(
            evidence_token_layers
        )
        if not torch.isfinite(token_attn_loss):
            raise FloatingPointError(
                f"non-finite evidence token attention loss: {token_attn_loss}"
            )
        base_model._last_route_evidence_token_attn_loss = float(
            token_attn_loss.detach()
        )
        base_model._last_remote_needle_qk_mass = float(
            token_attn_stats["remote_needle_qk_mass"]
        )
        base_model._last_evidence_local_weight = float(
            token_attn_stats["local_weight"]
        )
        loss_sum = loss_sum + (
            evidence_token_weight
            * token_attn_loss
            * raw_count.to(token_attn_loss.dtype)
        )
    if dense_layers is not None:
        from prefilling_v2.full_dense_teacher import collect_dense_teacher_loss
        dense_loss = collect_dense_teacher_loss(dense_layers)
        base_model._last_full_dense_teacher_kl = float(dense_loss.detach())
        loss_sum = loss_sum + args.route_dense_teacher_weight * dense_loss * raw_count.to(dense_loss.dtype)
    if support_attn_layers is not None:
        from prefilling_v2.support_attn_distill import collect_support_attn_kl

        support_loss = collect_support_attn_kl(support_attn_layers)
        base_model._last_route_support_attn_kl = float(support_loss.detach())
        loss_sum = loss_sum + (
            support_attn_weight
            * support_loss
            * raw_count.to(support_loss.dtype)
        )
        # Q-Cal is trainable but detached from this KL; pin it in the DDP graph.
        loss_sum = loss_sum + _zero_join_unused_qcal(train_root, loss_sum)
    if bool(getattr(args, "route_force_remote_unit", False)):
        loss_sum = _zero_join_lmk_type_embed(train_root, loss_sum)
        gate_bce_weight = float(getattr(args, "route_gate_bce_weight", 0.0) or 0.0)
        if base_model.training and gate_bce_weight > 0:
            from prefilling_v2.force_remote import (
                collect_fusion_gate_bce,
                collect_labeled_w_remote,
            )

            gate_loss = collect_fusion_gate_bce(base_model)
            if not torch.isfinite(gate_loss):
                raise FloatingPointError(f"non-finite fusion gate BCE: {gate_loss}")
            base_model._last_route_gate_bce = float(gate_loss.detach())
            base_model._last_route_w_remote = collect_labeled_w_remote(base_model)
            # Mean BCE, not BCE * n_targets. The previous scale drowned CE
            # (~9) under ~60 nats of gate loss and clipped every Q-Cal step.
            loss_sum = loss_sum + gate_bce_weight * gate_loss
    return loss_sum, weighted_count.to(device=loss_sum.device)


def _index_batch(
    batch: dict[str, torch.Tensor], keep: torch.Tensor
) -> dict[str, torch.Tensor]:
    if keep.ndim != 1:
        raise ValueError("batch keep mask must be 1D")
    indexed: dict[str, torch.Tensor] = {}
    for name, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == keep.shape:
            indexed[name] = value[keep]
        else:
            indexed[name] = value
    return indexed


def _distant_infill_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "route_asymmetric_gate_ce", False)) or bool(
        getattr(args, "route_balanced_view_ce", False)
    ) or bool(getattr(args, "route_force_remote_unit", False))


def _split_gate_views(
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if "gate_live" not in batch:
        raise RuntimeError("distant-infill batch is missing gate_live")
    live = batch["gate_live"].bool().reshape(-1)
    if int(live.sum().item()) <= 0 or int((~live).sum().item()) <= 0:
        raise RuntimeError("distant infill requires both local and distant views")
    return _index_batch(batch, ~live), _index_batch(batch, live)


def _split_dolma_ruler_views(
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if "view_ids" not in batch:
        raise RuntimeError("sync ruler CE batch is missing view_ids")
    ruler = _ruler_view_mask(batch["view_ids"], int(batch["view_ids"].shape[0]))
    if int(ruler.sum().item()) <= 0 or int((~ruler).sum().item()) <= 0:
        raise RuntimeError("sync ruler CE requires Dolma and RULER views in the same batch")
    return _index_batch(batch, ~ruler), _index_batch(batch, ruler)


def _index_ruler_task(
    batch: dict[str, torch.Tensor],
    task_id: int,
) -> dict[str, torch.Tensor]:
    ids = batch["view_ids"].reshape(-1)
    selected = ids == (int(RULER_VIEW_ID_BASE) + int(task_id))
    if int(selected.sum().item()) <= 0:
        raise RuntimeError(f"sync ruler CE missing RULER task {task_id}")
    return _index_batch(batch, selected)


def _iter_sync_task_batches(
    batch: dict[str, torch.Tensor],
    *,
    all_tasks: bool,
    task_ids: Sequence[int],
):
    dolma_batch, ruler_batch = _split_dolma_ruler_views(batch)
    yield "dolma", dolma_batch
    if all_tasks:
        for task_id in task_ids:
            yield f"ruler{int(task_id)}", _index_ruler_task(ruler_batch, int(task_id))
        return
    yield "ruler", ruler_batch


def _sync_named_window_target_counts(
    cpu_window: list[dict[str, torch.Tensor]],
    *,
    all_tasks: bool,
    task_ids: Sequence[int],
    device: torch.device,
    all_reduce_sum=None,
) -> dict[str, torch.Tensor]:
    names = ["dolma"] + (
        [f"ruler{int(task_id)}" for task_id in task_ids] if all_tasks else ["ruler"]
    )
    local_counts = {name: [] for name in names}
    for cpu_batch in cpu_window:
        for name, task_batch in _iter_sync_task_batches(
            cpu_batch, all_tasks=all_tasks, task_ids=task_ids
        ):
            local_counts[name].append(int(task_batch["target_count"].sum().item()))
    global_counts: dict[str, torch.Tensor] = {}
    for name in names:
        global_counts[name] = global_target_count(
            local_counts[name], device=device, all_reduce_sum=all_reduce_sum
        )
        if float(global_counts[name].item()) <= 0:
            raise RuntimeError(f"sync ruler CE window has no {name} targets")
    return global_counts


def _sync_ruler_window_target_counts(
    cpu_window: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    all_reduce_sum=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-rank Dolma/RULER target counts over the whole accumulation window."""

    counts = _sync_named_window_target_counts(
        cpu_window,
        all_tasks=False,
        task_ids=(),
        device=device,
        all_reduce_sum=all_reduce_sum,
    )
    return counts["dolma"], counts["ruler"]


def scale_sync_task_loss(
    local_loss_sum: torch.Tensor,
    *,
    global_count: torch.Tensor,
    world_size: int,
    task_weight: float = 0.5,
) -> torch.Tensor:
    """task_weight * world_size * local CE sum / global task targets (DDP-mean cancelled)."""

    return float(task_weight) * scale_local_loss_for_ddp(
        local_loss_sum, global_count=global_count, world_size=world_size
    )


def _set_fusion_detach_for_forward(model: torch.nn.Module, detach: bool) -> None:
    root = _unwrap_parallel_training_model(model)
    updated = set_route_fusion_detach(root, detach)
    if updated <= 0:
        raise RuntimeError("asymmetric gate CE found no route fusion layers")


def denoising_loss_sum_for_batch(
    model: torch.nn.Module,
    args: argparse.Namespace,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    unwrapped = model.module if hasattr(model, "module") else model
    if isinstance(unwrapped, _DenoisingLossWrapper):
        # Calling the outer object is essential: DDP must observe its own
        # forward before autograd starts so gradient buckets are reduced.
        return model(batch)
    return _denoising_loss_sum_for_unwrapped_model(model, args, batch)


def _write_run_config(args: argparse.Namespace, rank: int, world_size: int) -> None:
    if rank != 0:
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    payload["config"] = str(payload["config"]) if payload.get("config") else None
    payload["world_size"] = world_size
    payload["resume_contract"] = _training_resume_contract(args)
    (output_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _install_legacy_dream_rope_compat() -> None:
    """Restore the default RoPE factory expected by Dream remote code.

    Transformers 5 removed ``default`` from ``ROPE_INIT_FUNCTIONS`` while the
    released Dream-v0 module still indexes that public registry directly.
    Registering the original formula keeps the model checkpoint untouched and
    is a no-op on compatible Transformers 4.x environments.
    """

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def default_rope(config, device, seq_len=None, **rope_kwargs):
        del seq_len
        dim = int(
            rope_kwargs.get(
                "dim",
                getattr(config, "head_dim", None)
                or (config.hidden_size // config.num_attention_heads),
            )
        )
        base = float(
            rope_kwargs.get("base", getattr(config, "rope_theta", 10000.0))
        )
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64, device=device).float()
                / dim
            )
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = default_rope


def _build_model_and_tokenizer(args: argparse.Namespace, device: torch.device):
    _install_legacy_dream_rope_compat()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    model_config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    if args.model_max_position_embeddings is not None:
        model_config.max_position_embeddings = int(
            args.model_max_position_embeddings
        )
    if args.model_rope_theta is not None:
        model_config.rope_theta = float(args.model_rope_theta)
    hope_scaling = None
    if args.model_rope_scaling is not None:
        scaling = dict(args.model_rope_scaling)
        rope_type = scaling.get("rope_type", scaling.get("type"))
        if rope_type == "hope":
            hope_scaling = scaling
            model_config.rope_scaling = None
        else:
            model_config.rope_scaling = scaling
    model = AutoModel.from_pretrained(
        args.model_path,
        config=model_config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    plan = configure_training_attention(model, args)
    if hope_scaling is not None:
        from prefilling_v2.hope import apply_hope_inrange

        patched = apply_hope_inrange(
            model,
            context_length=int(
                hope_scaling.get("original_max_position_embeddings") or 2048
            ),
            period_multiplier=float(hope_scaling.get("period_multiplier") or 1.0),
        )
        print(
            json.dumps(
                {
                    "hope_inrange": True,
                    "context_length": int(
                        hope_scaling.get("original_max_position_embeddings") or 2048
                    ),
                    "rotary_modules": patched,
                }
            ),
            flush=True,
        )
    if args.model_rope_scaling and args.model_rope_scaling.get("original_max_position_embeddings") == 2048 and args.model_rope_scaling.get("factor") == 16:
        from prefilling_v2.chunk_distillation import verify_yarn
        verify_yarn(model, args)
    model = apply_lora(model, args)
    configure_landmark_embedding(model, args, tokenizer)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.to(device)
    return model, tokenizer, plan


def _attention_mask_for_model(
    args: argparse.Namespace,
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor | None = None,
) -> torch.Tensor | str:
    if segment_ids is not None:
        if segment_ids.shape != attention_mask.shape:
            raise ValueError("segment_ids must match the 2D attention mask")
        valid = attention_mask.bool()
        segments = segment_ids.to(dtype=torch.long)
        if torch.any(valid & (segments <= 0)):
            raise ValueError("valid tokens must have positive segment ids")
        if torch.any((~valid) & (segments != 0)):
            raise ValueError("invalid tokens must have segment id zero")
        same_segment = segments[:, :, None] == segments[:, None, :]
        allowed = valid[:, :, None] & valid[:, None, :] & same_segment
        return allowed[:, None, :, :]
    if args.attention_mode in {"route", "dsa"}:
        return attention_mask
    if not bool(attention_mask.bool().all()):
        raise ValueError(
            "dense control requires fully valid fixed-length packed sequences"
        )
    return "full"


def split_packed_indices(
    length: int,
    validation_packs: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    if length <= 0:
        raise ValueError("packed corpus must not be empty")
    if validation_packs < 0 or validation_packs >= length:
        raise ValueError(
            "validation_packs must be non-negative and leave training data"
        )
    if validation_packs == 0:
        return list(range(length)), []
    generator = np.random.default_rng(int(seed) ^ 0x5EED5EED)
    validation = sorted(
        int(index)
        for index in generator.choice(
            length, size=validation_packs, replace=False
        ).tolist()
    )
    validation_set = set(validation)
    training = [
        index for index in range(length) if index not in validation_set
    ]
    return training, validation


def _build_data(args: argparse.Namespace, tokenizer, rank: int, world_size: int):
    metadata = json.loads(Path(args.corpus_meta).read_text(encoding="utf-8"))
    expected_text_slots = (
        None
        if args.attention_mode == "dsa"
        else (args.max_length // args.chunk_size) * (args.chunk_size - 1)
    )
    full_dataset = DreamPackedCorpus(
        args.corpus_bin,
        args.corpus_meta,
        expected_real_slots=expected_text_slots,
        expected_tokenizer_fingerprint=metadata["tokenizer_fingerprint"],
    )
    if args.attention_mode == "dsa" and full_dataset.real_slots_per_pack > args.max_length:
        raise ValueError(
            "DSA packed corpus is longer than max_length: "
            f"{full_dataset.real_slots_per_pack} > {args.max_length}"
        )
    train_indices, validation_indices = split_packed_indices(
        len(full_dataset), int(args.validation_packs), args.seed
    )
    if args.overfit_packs > 0:
        if args.overfit_packs < world_size:
            raise ValueError("overfit_packs must be at least world_size")
        train_indices = train_indices[: args.overfit_packs]
    dataset = Subset(full_dataset, train_indices)
    validation_dataset = Subset(full_dataset, validation_indices)
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Dream tokenizer has no mask_token_id")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    lmk_token_id = resolve_landmark_token_id(args, tokenizer)
    ruler_synthesizer = (
        RulerDenoisingSynthesizer(
            tokenizer,
            task_ids=tuple(int(task_id) for task_id in args.ruler_task_ids),
        )
        if float(args.ruler_mix_ratio) > 0
        or bool(getattr(args, "route_sync_ruler_ce", False))
        or bool(getattr(args, "sync_ruler_all_tasks", False))
        else None
    )
    ruler_val_batches = int(getattr(args, "ruler_val_batches", 0) or 0)
    ruler_val_synthesizer = (
        RulerDenoisingSynthesizer(tokenizer, task_ids=(0,))
        if ruler_val_batches > 0
        else None
    )
    collator = FullTextComplementaryCollator(
        mask_token_id=int(mask_token_id),
        pad_token_id=int(pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        lmk_token_id=int(lmk_token_id),
        chunk_size=args.chunk_size,
        t_min=args.t_min,
        t_max=args.t_max,
        seed=args.seed,
        ruler_mix_ratio=float(args.ruler_mix_ratio),
        ruler_synthesizer=ruler_synthesizer,
        ruler_every_step=(
            bool(getattr(args, "route_sync_ruler_ce", False))
            or bool(getattr(args, "sync_ruler_all_tasks", False))
        ),
        ruler_all_tasks=bool(getattr(args, "sync_ruler_all_tasks", False)),
        ruler_joint_local_mask=bool(getattr(args, "ruler_joint_local_mask", False)),
        insert_landmarks=args.attention_mode != "dsa",
        pad_to=args.max_length if args.attention_mode == "dsa" else None,
        distant_infill=_distant_infill_enabled(args),
        distant_min_gap=int(
            getattr(args, "route_distant_min_gap", None) or args.local_window
        ),
        distant_span_min=int(getattr(args, "route_distant_span_min", 8) or 8),
        distant_span_max=int(getattr(args, "route_distant_span_max", 16) or 16),
        distant_needles_min=int(getattr(args, "route_distant_needles_min", 1) or 1),
        distant_needles_max=int(getattr(args, "route_distant_needles_max", 1) or 1),
        distant_needle_sep=int(getattr(args, "route_distant_needle_sep", 64) or 0),
        distant_cue_len=int(getattr(args, "route_distant_cue_len", 0) or 0),
        distant_only=bool(getattr(args, "route_force_remote_unit", False)),
        tokenizer=tokenizer if bool(getattr(args, "route_force_remote_unit", False)) else None,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        generator=loader_generator,
    )
    validation_collator = FullTextComplementaryCollator(
        mask_token_id=int(mask_token_id),
        pad_token_id=int(pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        lmk_token_id=int(lmk_token_id),
        chunk_size=args.chunk_size,
        t_min=args.t_min,
        t_max=args.t_max,
        seed=args.seed,
        ruler_mix_ratio=float(args.ruler_mix_ratio),
        ruler_synthesizer=ruler_synthesizer,
        insert_landmarks=args.attention_mode != "dsa",
        pad_to=args.max_length if args.attention_mode == "dsa" else None,
        distant_infill=bool(getattr(args, "route_force_remote_unit", False)),
        distant_min_gap=int(
            getattr(args, "route_distant_min_gap", None) or args.local_window
        ),
        distant_span_min=int(getattr(args, "route_distant_span_min", 8) or 8),
        distant_span_max=int(getattr(args, "route_distant_span_max", 16) or 16),
        distant_needles_min=int(getattr(args, "route_distant_needles_min", 1) or 1),
        distant_needles_max=int(getattr(args, "route_distant_needles_max", 1) or 1),
        distant_needle_sep=int(getattr(args, "route_distant_needle_sep", 64) or 0),
        distant_cue_len=int(getattr(args, "route_distant_cue_len", 0) or 0),
        distant_only=bool(getattr(args, "route_force_remote_unit", False)),
        tokenizer=tokenizer if bool(getattr(args, "route_force_remote_unit", False)) else None,
    )
    validation_sampler = DistributedSampler(
        validation_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.micro_batch_size,
        sampler=validation_sampler,
        collate_fn=validation_collator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    ruler_val_loader = None
    if ruler_val_batches > 0:
        if ruler_val_synthesizer is None:
            raise RuntimeError("ruler_val_batches requires a RULER synthesizer")
        if len(validation_dataset) == 0:
            raise ValueError("ruler_val_batches requires validation_packs > 0")
        ruler_val_collator = FullTextComplementaryCollator(
            mask_token_id=int(mask_token_id),
            pad_token_id=int(pad_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
            lmk_token_id=int(lmk_token_id),
            chunk_size=args.chunk_size,
            t_min=args.t_min,
            t_max=args.t_max,
            seed=args.seed + 91,
            ruler_mix_ratio=1.0,
            ruler_synthesizer=ruler_val_synthesizer,
            insert_landmarks=args.attention_mode != "dsa",
            pad_to=args.max_length if args.attention_mode == "dsa" else None,
        )
        ruler_val_loader = DataLoader(
            validation_dataset,
            batch_size=args.micro_batch_size,
            sampler=validation_sampler,
            collate_fn=ruler_val_collator,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
    return (
        dataset,
        validation_dataset,
        metadata,
        sampler,
        collator,
        loader,
        validation_loader,
        ruler_val_loader,
    )


@torch.no_grad()
def evaluate_validation_loss(
    model: torch.nn.Module,
    args: argparse.Namespace,
    loader: DataLoader,
    device: torch.device,
    *,
    collect_gate: bool = False,
) -> tuple[float, int, dict[str, float]]:
    was_training = model.training
    model.eval()
    local_loss = torch.zeros((), device=device, dtype=torch.float64)
    local_count = torch.zeros((), device=device, dtype=torch.float64)
    gate_logits: list[torch.Tensor] = []
    gate_masses: list[torch.Tensor] = []
    layer_logits: dict[int, list[torch.Tensor]] = {}
    layer_masses: dict[int, list[torch.Tensor]] = {}
    layer_controls: dict[int, list[torch.Tensor]] = {}
    core = _unwrap_training_model(model)
    for batch_idx, cpu_batch in enumerate(loader):
        if batch_idx >= args.validation_batches:
            break
        batch = _move_batch(cpu_batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss_sum, target_count = denoising_loss_sum_for_batch(
                model,
                args,
                batch,
            )
        local_loss += loss_sum.double()
        local_count += target_count
        if collect_gate:
            from prefilling_v2.force_remote import collect_labeled_gate_vectors

            packed = collect_labeled_gate_vectors(core)
            if packed is not None:
                gate_logits.append(packed["logit"])
                gate_masses.append(packed["mass"])
                for index, layer in enumerate(packed["layers"]):
                    layer_logits.setdefault(index, []).append(layer["logit"])
                    layer_masses.setdefault(index, []).append(layer["mass"])
                    if "control_logit" in layer:
                        layer_controls.setdefault(index, []).append(
                            layer["control_logit"]
                        )
    global_loss = all_reduce_detached_sum(local_loss)
    global_count = all_reduce_detached_sum(local_count)
    if global_count.item() <= 0:
        raise RuntimeError("validation produced no supervised tokens")
    extras: dict[str, float] = {}
    if collect_gate:
        from prefilling_v2.force_remote import (
            collect_fusion_gate_offsets,
            layer_gate_metrics,
            summarize_gate_vectors,
        )

        local_logit = (
            torch.cat(gate_logits) if gate_logits else torch.zeros((0,))
        )
        local_mass = (
            torch.cat(gate_masses) if gate_masses else torch.zeros((0,))
        )
        local_layers = []
        for index in sorted(layer_logits):
            control = layer_controls.get(index, [])
            local_layers.append(
                {
                    "logit": torch.cat(layer_logits[index]),
                    "mass": torch.cat(layer_masses[index]),
                    "control_logit": (
                        torch.cat(control) if control else torch.zeros((0,))
                    ),
                }
            )
        gathered = all_gather_object(
            {
                "logit": local_logit.cpu(),
                "mass": local_mass.cpu(),
                "layers": local_layers,
            }
        )
        logit = torch.cat([item["logit"].reshape(-1) for item in gathered])
        mass = torch.cat([item["mass"].reshape(-1) for item in gathered])
        extras.update(summarize_gate_vectors(logit, mass))
        merged_layers = []
        layer_count = max((len(item["layers"]) for item in gathered), default=0)
        for index in range(layer_count):
            merged_layers.append(
                {
                    "logit": torch.cat(
                        [
                            item["layers"][index]["logit"].reshape(-1)
                            for item in gathered
                            if index < len(item["layers"])
                        ]
                    ),
                    "mass": torch.cat(
                        [
                            item["layers"][index]["mass"].reshape(-1)
                            for item in gathered
                            if index < len(item["layers"])
                        ]
                    ),
                    "control_logit": torch.cat(
                        [
                            item["layers"][index].get(
                                "control_logit", torch.zeros((0,))
                            ).reshape(-1)
                            for item in gathered
                            if index < len(item["layers"])
                        ]
                    )
                    if any(
                        "control_logit" in item["layers"][index]
                        for item in gathered
                        if index < len(item["layers"])
                    )
                    else torch.zeros((0,)),
                }
            )
        extras.update(layer_gate_metrics(merged_layers))
        from prefilling_v2.force_remote import collect_fusion_gate_scales

        offsets = collect_fusion_gate_offsets(core)
        extras["fusion_gate_offset"] = (
            float(sum(offsets) / len(offsets)) if offsets else float("nan")
        )
        for index, value in enumerate(offsets):
            extras[f"fusion_gate_offset_l{index}"] = float(value)
        scales = collect_fusion_gate_scales(core)
        extras["fusion_gate_scale"] = (
            float(sum(scales) / len(scales)) if scales else float("nan")
        )
        for index, value in enumerate(scales):
            extras[f"fusion_gate_scale_l{index}"] = float(value)
    if was_training:
        model.train()
    return float((global_loss / global_count).item()), int(global_count.item()), extras


@torch.no_grad()
def evaluate_force_remote_ablations(
    model: torch.nn.Module,
    args: argparse.Namespace,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compare forced-remote CE against remote-off and shuffled blocks."""

    core = _unwrap_training_model(model)
    config = getattr(core, "config", None)
    if config is None:
        raise RuntimeError("force-remote ablation requires a model config")
    metrics: dict[str, float] = {}
    previous = (
        str(getattr(config, "route_remote_ablation", "none")),
        int(getattr(config, "route_remote_ablation_seed", 0) or 0),
    )
    try:
        for name, ablation, seed in (
            ("live_gate", "learned", 0),
            ("force_on", "none", 0),
            ("remote_off", "off", 0),
            ("shuffle_block", "shuffle", 7),
        ):
            config.route_remote_ablation = ablation
            config.route_remote_ablation_seed = seed
            loss, count, extras = evaluate_validation_loss(
                model,
                args,
                loader,
                device,
                collect_gate=(name == "live_gate"),
            )
            metrics[f"{name}_ce"] = loss
            metrics[f"{name}_targets"] = float(count)
            if name == "live_gate":
                metrics.update(extras)
    finally:
        config.route_remote_ablation = previous[0]
        config.route_remote_ablation_seed = previous[1]
    return metrics


def ruler_sn_span_counts(
    pred: torch.Tensor,
    gold: torch.Tensor,
    *,
    eos_id: int | None,
) -> dict[str, torch.Tensor]:
    """Split a teacher-forced SN answer span into digit vs EOS hits.

    SN gold is ``8 digits + EOS``. Token accuracy includes EOS, so a model that
    only learns to emit EOS can look like ~11% token acc with 0 digit acc.
    """
    if pred.shape != gold.shape:
        raise ValueError(
            f"pred/gold shape mismatch {tuple(pred.shape)} vs {tuple(gold.shape)}"
        )
    if pred.ndim != 1:
        raise ValueError("pred and gold must be 1D answer spans")
    match = pred.eq(gold)
    eos_mask = (
        gold.eq(int(eos_id))
        if eos_id is not None
        else torch.zeros_like(gold, dtype=torch.bool)
    )
    digit_mask = ~eos_mask
    digit_n = digit_mask.sum().to(dtype=torch.long)
    eos_n = eos_mask.sum().to(dtype=torch.long)
    digit_ok = (match & digit_mask).sum().to(dtype=torch.long)
    eos_ok = (match & eos_mask).sum().to(dtype=torch.long)
    return {
        "token_ok": match.sum().to(dtype=torch.long),
        "token_n": gold.new_tensor(int(gold.numel()), dtype=torch.long),
        "digit_ok": digit_ok,
        "digit_n": digit_n,
        "eos_ok": eos_ok,
        "eos_n": eos_n,
        "digit_em": ((digit_n > 0) & digit_ok.eq(digit_n)).to(dtype=torch.long),
        "seq_em": match.all().to(dtype=torch.long),
        "seq_n": gold.new_ones((), dtype=torch.long),
    }


@torch.no_grad()
def evaluate_ruler_sn(
    model: torch.nn.Module,
    args: argparse.Namespace,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Teacher-forced single-needle token EM on packed RULER val (task 0)."""
    was_training = model.training
    model.eval()
    base_model = _unwrap_training_model(model)
    eos_id = getattr(getattr(base_model, "config", None), "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(args, "eos_token_id", None)
    token_ok = torch.zeros((), device=device, dtype=torch.long)
    token_n = torch.zeros((), device=device, dtype=torch.long)
    digit_ok = torch.zeros((), device=device, dtype=torch.long)
    digit_n = torch.zeros((), device=device, dtype=torch.long)
    eos_ok = torch.zeros((), device=device, dtype=torch.long)
    eos_n = torch.zeros((), device=device, dtype=torch.long)
    digit_em = torch.zeros((), device=device, dtype=torch.long)
    seq_ok = torch.zeros((), device=device, dtype=torch.long)
    seq_n = torch.zeros((), device=device, dtype=torch.long)
    limit = int(getattr(args, "ruler_val_batches", 0) or 0)
    for batch_idx, cpu_batch in enumerate(loader):
        if batch_idx >= limit:
            break
        batch = _move_batch(cpu_batch, device)
        labels = batch["labels"]
        selected = labels.ne(-100)
        input_ids, inputs_embeds = _landmark_inputs_embeds(
            model,
            batch["input_ids"],
            batch.get("landmark_mask"),
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = base_model.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=_attention_mask_for_model(
                    args,
                    batch["attention_mask"],
                    batch["segment_ids"],
                ),
                position_ids=batch["position_ids"],
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            for row in range(hidden.size(0)):
                mask = selected[row]
                if not bool(mask.any()):
                    continue
                logits = base_model.lm_head(hidden[row][mask])
                pred = logits.argmax(dim=-1)
                gold = labels[row][mask]
                counts = ruler_sn_span_counts(pred, gold, eos_id=eos_id)
                token_ok += counts["token_ok"]
                token_n += counts["token_n"]
                digit_ok += counts["digit_ok"]
                digit_n += counts["digit_n"]
                eos_ok += counts["eos_ok"]
                eos_n += counts["eos_n"]
                digit_em += counts["digit_em"]
                seq_ok += counts["seq_em"]
                seq_n += counts["seq_n"]
    token_ok = all_reduce_detached_sum(token_ok)
    token_n = all_reduce_detached_sum(token_n)
    digit_ok = all_reduce_detached_sum(digit_ok)
    digit_n = all_reduce_detached_sum(digit_n)
    eos_ok = all_reduce_detached_sum(eos_ok)
    eos_n = all_reduce_detached_sum(eos_n)
    digit_em = all_reduce_detached_sum(digit_em)
    seq_ok = all_reduce_detached_sum(seq_ok)
    seq_n = all_reduce_detached_sum(seq_n)
    if was_training:
        model.train()
    n_seq = int(seq_n.item())
    n_tok = int(token_n.item())
    n_digit = int(digit_n.item())
    n_eos = int(eos_n.item())
    return {
        "ruler_val_sn_em": float(seq_ok.item()) / n_seq if n_seq else 0.0,
        "ruler_val_sn_digit_em": float(digit_em.item()) / n_seq if n_seq else 0.0,
        "ruler_val_sn_token_acc": float(token_ok.item()) / n_tok if n_tok else 0.0,
        "ruler_val_sn_digit_acc": float(digit_ok.item()) / n_digit if n_digit else 0.0,
        "ruler_val_sn_eos_acc": float(eos_ok.item()) / n_eos if n_eos else 0.0,
        "ruler_val_sn_n": float(n_seq),
        "ruler_val_sn_tokens": float(n_tok),
        "ruler_val_sn_digits": float(n_digit),
        "ruler_val_sn_eos_tokens": float(n_eos),
    }


def _print_ruler_val(
    rank: int,
    step: int,
    metrics: dict[str, float],
    output_dir: str,
) -> None:
    _print(
        rank,
        f"step={step} ruler_val_sn_em={metrics['ruler_val_sn_em']:.4f} "
        f"ruler_val_sn_digit_em={metrics.get('ruler_val_sn_digit_em', float('nan')):.4f} "
        f"ruler_val_sn_token_acc={metrics['ruler_val_sn_token_acc']:.4f} "
        f"ruler_val_sn_digit_acc={metrics.get('ruler_val_sn_digit_acc', float('nan')):.4f} "
        f"ruler_val_sn_eos_acc={metrics.get('ruler_val_sn_eos_acc', float('nan')):.4f} "
        f"ruler_val_sn_n={int(metrics['ruler_val_sn_n'])} "
        f"ruler_val_sn_tokens={int(metrics['ruler_val_sn_tokens'])} "
        f"ruler_val_sn_digits={int(metrics.get('ruler_val_sn_digits', 0))} "
        f"ruler_val_sn_eos_tokens={int(metrics.get('ruler_val_sn_eos_tokens', 0))}",
    )
    if rank != 0:
        return
    path = Path(output_dir) / "ruler_val_sn.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"step": int(step), **metrics}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _format_named_vector(metrics: dict[str, float], prefix: str) -> str:
    values = []
    index = 0
    while f"{prefix}_l{index}" in metrics:
        values.append(f"{float(metrics[f'{prefix}_l{index}']):.4f}")
        index += 1
    return ",".join(values) if values else "na"


def _format_offset_vector(metrics: dict[str, float]) -> str:
    return _format_named_vector(metrics, "fusion_gate_offset")


def _format_layer_field(metrics: dict[str, float], field: str) -> str:
    values = []
    index = 0
    while f"l{index}_{field}" in metrics:
        values.append(f"{float(metrics[f'l{index}_{field}']):.4f}")
        index += 1
    return ",".join(values) if values else "na"


def _print_force_remote_ablations(
    rank: int,
    step: int,
    metrics: dict[str, float],
    output_dir: str,
) -> None:
    _print(
        rank,
        f"step={step} live_gate_ce={metrics.get('live_gate_ce', float('nan')):.6f} "
        f"live_gate_bce={metrics.get('live_gate_bce', float('nan')):.6f} "
        f"live_w_remote={metrics.get('live_w_remote', float('nan')):.6f} "
        f"live_w_remote_p50={metrics.get('live_w_remote_p50', float('nan')):.6f} "
        f"live_frac_w_gt_0.8={metrics.get('live_frac_w_gt_0.8', float('nan')):.6f} "
        f"gate_logit_mean={metrics.get('gate_logit_mean', float('nan')):.6f} "
        f"gate_logit_std={metrics.get('gate_logit_std', float('nan')):.6f} "
        f"gate_logit_p50={metrics.get('gate_logit_p50', float('nan')):.6f} "
        f"gate_logit_p90={metrics.get('gate_logit_p90', float('nan')):.6f} "
        f"fusion_gate_offset={metrics.get('fusion_gate_offset', float('nan')):.6f} "
        f"fusion_gate_offsets={_format_offset_vector(metrics)} "
        f"fusion_gate_scale={metrics.get('fusion_gate_scale', float('nan')):.6f} "
        f"fusion_gate_scales={_format_named_vector(metrics, 'fusion_gate_scale')} "
        f"layer_w_remote={_format_layer_field(metrics, 'live_w_remote')} "
        f"layer_frac_w_gt_0.8={_format_layer_field(metrics, 'live_frac_w_gt_0.8')} "
        f"layer_gate_logit_mean={_format_layer_field(metrics, 'gate_logit_mean')} "
        f"layer_gate_logit_std={_format_layer_field(metrics, 'gate_logit_std')} "
        f"layer_gate_logit_p90={_format_layer_field(metrics, 'gate_logit_p90')} "
        f"layer_n_pos={_format_layer_field(metrics, 'n_pos')} "
        f"layer_n_neg={_format_layer_field(metrics, 'n_neg')} "
        f"layer_pos_mean={_format_layer_field(metrics, 'pos_mean')} "
        f"layer_neg_mean={_format_layer_field(metrics, 'neg_mean')} "
        f"layer_delta={_format_layer_field(metrics, 'delta_pos_minus_neg')} "
        f"layer_auc={_format_layer_field(metrics, 'auc')} "
        f"layer_bce_grad_y1={_format_layer_field(metrics, 'bce_grad_y1')} "
        f"layer_sign_ok={_format_layer_field(metrics, 'sign_ok')} "
        f"force_on_ce={metrics['force_on_ce']:.6f} "
        f"remote_off_ce={metrics['remote_off_ce']:.6f} "
        f"shuffle_block_ce={metrics['shuffle_block_ce']:.6f} "
        f"force_on_targets={int(metrics['force_on_targets'])}",
    )
    if rank != 0:
        return
    path = Path(output_dir) / "force_remote_ablate.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": int(step), **metrics}) + "\n")
    trace_path = Path(output_dir) / "gate_layer_trace.jsonl"
    with trace_path.open("a", encoding="utf-8") as handle:
        offsets = []
        index = 0
        while f"fusion_gate_offset_l{index}" in metrics:
            offsets.append(float(metrics[f"fusion_gate_offset_l{index}"]))
            index += 1
        scales = []
        index = 0
        while f"fusion_gate_scale_l{index}" in metrics:
            scales.append(float(metrics[f"fusion_gate_scale_l{index}"]))
            index += 1
        handle.write(
            json.dumps(
                {
                    "step": int(step),
                    "split": "val",
                    "offsets": offsets,
                    "scales": scales,
                    **metrics,
                }
            )
            + "\n"
        )


def _gradient_audit(
    model: torch.nn.Module,
    *,
    require_lora_nonzero: bool = True,
) -> dict[str, int]:
    unwrapped = model.module if hasattr(model, "module") else model
    stats = {
        "lora_nonzero": 0,
        "entropy_nonzero": 0,
        "entropy_tensors": 0,
        "dsa_indexer_nonzero": 0,
        "dsa_indexer_tensors": 0,
        "qcal_nonzero": 0,
        "qcal_tensors": 0,
    }
    for name, parameter in unwrapped.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is not None and not torch.isfinite(gradient).all():
            raise FloatingPointError(f"non-finite trainable gradient: {name}")
        nonzero = gradient is not None and bool(torch.count_nonzero(gradient))
        if _is_qcal_param(name):
            stats["qcal_tensors"] += 1
            stats["qcal_nonzero"] += int(nonzero)
        if "lora_" in name and nonzero:
            stats["lora_nonzero"] += 1
        if "entropy_bias_scale" in name:
            stats["entropy_tensors"] += 1
            if parameter.dtype != torch.float32:
                raise TypeError(f"entropy parameter is not FP32: {name}")
            if nonzero:
                stats["entropy_nonzero"] += 1
        if ".dsa_indexer." in name:
            stats["dsa_indexer_tensors"] += 1
            if nonzero:
                stats["dsa_indexer_nonzero"] += 1
    if require_lora_nonzero and stats["lora_nonzero"] == 0:
        raise RuntimeError("all LoRA gradients are zero or missing")
    return stats


def _optimizer_param_groups(
    model: torch.nn.Module, args: argparse.Namespace
) -> list[dict]:
    """Split AdamW groups so gate offset / Q-Cal / LoRA-Q can use different LRs."""

    default_lr = float(args.learning_rate)
    lora_q_lr = float(getattr(args, "route_lora_q_lr", 0.0) or 0.0) or default_lr
    qcal_lr = float(getattr(args, "route_qcal_lr", 0.0) or 0.0) or default_lr
    offset_lr = float(getattr(args, "route_gate_offset_lr", 0.0) or 0.0) or default_lr
    scale_lr = float(getattr(args, "route_gate_scale_lr", 0.0) or 0.0) or offset_lr
    offset_layer_lr = _as_float_map(getattr(args, "route_gate_offset_layer_lr", None))
    scale_layer_lr = _as_float_map(getattr(args, "route_gate_scale_layer_lr", None))
    buckets: dict[str, tuple[list, float]] = {
        "lora_q": ([], lora_q_lr),
        "qcal": ([], qcal_lr),
        "dsa_indexer": (
            [],
            float(getattr(args, "dsa_warmup_indexer_lr", 0.0) or 0.0) or default_lr
            if int(getattr(args, "dsa_warmup_steps", 0) or 0) > 0
            else default_lr,
        ),
        "rest": ([], default_lr),
    }
    unwrapped = model.module if hasattr(model, "module") else model
    offset_params = [
        (name, parameter)
        for name, parameter in unwrapped.named_parameters()
        if name.endswith("fusion_gate_offset")
    ]
    scale_params = [
        (name, parameter)
        for name, parameter in unwrapped.named_parameters()
        if name.endswith("fusion_gate_scale")
    ]
    for index, (name, parameter) in enumerate(offset_params):
        if not parameter.requires_grad:
            continue
        buckets[f"offset_l{index}"] = (
            [parameter],
            float(offset_layer_lr.get(index, offset_lr)),
        )
    for index, (name, parameter) in enumerate(scale_params):
        if not parameter.requires_grad:
            continue
        buckets[f"scale_l{index}"] = (
            [parameter],
            float(scale_layer_lr.get(index, scale_lr)),
        )
    for name, parameter in unwrapped.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("fusion_gate_offset") or name.endswith("fusion_gate_scale"):
            continue
        if _is_qcal_param(name):
            buckets["qcal"][0].append(parameter)
        elif ".dsa_indexer." in name:
            buckets["dsa_indexer"][0].append(parameter)
        elif "lora_" in name and ".q_proj." in name:
            buckets["lora_q"][0].append(parameter)
        else:
            buckets["rest"][0].append(parameter)
    groups = []
    for name, (params, lr) in buckets.items():
        if params:
            groups.append({"params": params, "lr": lr, "group_name": name})
    if not groups:
        raise RuntimeError("no trainable parameter groups")
    return groups


def _configure_dsa_official_step(
    model: torch.nn.Module, args: argparse.Namespace, step: int
) -> str:
    warmup_n = int(getattr(args, "dsa_warmup_steps", 0) or 0)
    if args.attention_mode != "dsa" or warmup_n <= 0:
        return "sparse"
    warmup = step <= warmup_n
    if warmup:
        aux_queries = int(getattr(args, "dsa_aux_queries", 0) or 0)
    else:
        sparse_queries = getattr(args, "dsa_sparse_aux_queries", None)
        aux_queries = (
            int(sparse_queries)
            if sparse_queries is not None
            else int(getattr(args, "dsa_aux_queries", 0) or 0)
        )
    set_dsa_official_stage(
        _unwrap_training_model(model),
        warmup=warmup,
        aux_queries=aux_queries,
        sparse_scope=str(getattr(args, "dsa_sparse_aux_scope", "selected")),
    )
    args.dsa_lm_loss_weight = 0.0 if warmup else 1.0
    args._dsa_stage_now = "warmup" if warmup else "sparse"
    return args._dsa_stage_now


def _apply_dsa_official_lrs(
    optimizer: torch.optim.Optimizer, args: argparse.Namespace, step: int
) -> None:
    warmup_n = int(getattr(args, "dsa_warmup_steps", 0) or 0)
    if args.attention_mode != "dsa" or warmup_n <= 0:
        return
    import math

    indexer_warm = float(getattr(args, "dsa_warmup_indexer_lr", 1e-3) or 1e-3)
    base_lr = float(args.learning_rate)
    sparse_steps = max(int(args.max_steps) - warmup_n, 1)
    cosine_warmup = int(args.warmup_steps)

    def cosine_lr(step_in_phase: int, n_steps: int, warmup: int, lr: float) -> float:
        if step_in_phase < warmup:
            return lr * float(step_in_phase + 1) / max(warmup, 1)
        t = (step_in_phase - warmup) / max(n_steps - warmup, 1)
        return lr * 0.5 * (1.0 + math.cos(math.pi * min(max(t, 0.0), 1.0)))

    for group in optimizer.param_groups:
        name = str(group.get("group_name", "rest"))
        if step <= warmup_n:
            group["lr"] = indexer_warm if name == "dsa_indexer" else 0.0
        else:
            group["lr"] = cosine_lr(
                step - warmup_n - 1, sparse_steps, cosine_warmup, base_lr
            )


def _clip_gate_cal_params(
    model: torch.nn.Module,
    *,
    suffix: str,
    default_max_norm: float,
    layer_clips: dict[int, float],
) -> float:
    if default_max_norm <= 0 and not layer_clips:
        return 0.0
    preclips = []
    for index, (_, parameter) in enumerate(_gate_cal_params(model, suffix)):
        if not parameter.requires_grad or parameter.grad is None:
            continue
        max_norm = float(layer_clips.get(index, default_max_norm) or 0.0)
        if max_norm <= 0:
            continue
        preclips.append(float(parameter.grad.detach().float().norm().item()))
        torch.nn.utils.clip_grad_norm_([parameter], max_norm)
    return float(max(preclips)) if preclips else 0.0


def _clip_matching_grad_norm(
    model: torch.nn.Module, max_norm: float, predicate, *, each: bool = False
) -> float:
    if max_norm <= 0:
        return 0.0
    unwrapped = model.module if hasattr(model, "module") else model
    parameters = [
        parameter
        for name, parameter in unwrapped.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and predicate(name)
    ]
    if not parameters:
        return 0.0
    if not each:
        return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm))
    preclips = []
    for parameter in parameters:
        preclips.append(float(parameter.grad.detach().float().norm().item()))
        torch.nn.utils.clip_grad_norm_([parameter], max_norm)
    return float(max(preclips))


def _mean_fusion_gate_offset(model: torch.nn.Module) -> float:
    values = [
        float(parameter.detach().float().cpu().item())
        for name, parameter in (
            model.module if hasattr(model, "module") else model
        ).named_parameters()
        if name.endswith("fusion_gate_offset")
    ]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _gradient_group_norms(model: torch.nn.Module) -> dict[str, float]:
    """Measure unclipped trainable gradient norms by parameter role."""

    unwrapped = model.module if hasattr(model, "module") else model
    squared = {
        "lora": torch.zeros((), device=next(model.parameters()).device),
        "lora_q": torch.zeros((), device=next(model.parameters()).device),
        "lora_k": torch.zeros((), device=next(model.parameters()).device),
        "lora_v": torch.zeros((), device=next(model.parameters()).device),
        "lora_o": torch.zeros((), device=next(model.parameters()).device),
        "lora_other": torch.zeros(
            (), device=next(model.parameters()).device
        ),
        "entropy": torch.zeros((), device=next(model.parameters()).device),
        "qcal": torch.zeros((), device=next(model.parameters()).device),
        "external_lmk": torch.zeros((), device=next(model.parameters()).device),
        "lmk_type": torch.zeros((), device=next(model.parameters()).device),
        "dsa_indexer": torch.zeros(
            (), device=next(model.parameters()).device
        ),
        "other": torch.zeros((), device=next(model.parameters()).device),
    }
    for name, parameter in unwrapped.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if "lora_" in name:
            groups = ["lora"]
            projection_group = next(
                (
                    f"lora_{projection}"
                    for projection in ("q", "k", "v", "o")
                    if f".{projection}_proj." in name
                ),
                "lora_other",
            )
            groups.append(projection_group)
        elif _is_qcal_param(name):
            groups = ["qcal"]
        elif "entropy_bias_scale" in name:
            groups = ["entropy"]
        elif "dream_route_lmk_embed" in name:
            groups = ["external_lmk"]
        elif "dream_route_lmk_type_embed" in name:
            groups = ["lmk_type"]
        elif ".dsa_indexer." in name:
            groups = ["dsa_indexer"]
        else:
            groups = ["other"]
        gradient_square = (
            parameter.grad.detach().float().square().sum()
        )
        for group in groups:
            squared[group] += gradient_square
    return {
        name: float(value.sqrt().item())
        for name, value in squared.items()
    }


def _token_allocation_stats(
    model: torch.nn.Module,
) -> dict[str, float] | None:
    """Summarize the latest entropy-adaptive allocations across route layers."""

    unwrapped = model.module if hasattr(model, "module") else model
    allocations = [
        allocation
        for module in unwrapped.modules()
        if (
            allocation := getattr(
                module, "_last_token_allocation", None
            )
        )
        is not None
    ]
    if not allocations:
        return None
    total = torch.stack(
        [allocation.float().sum() for allocation in allocations]
    ).sum()
    count = sum(allocation.numel() for allocation in allocations)
    minimum = torch.stack(
        [allocation.min() for allocation in allocations]
    ).min()
    maximum = torch.stack(
        [allocation.max() for allocation in allocations]
    ).max()
    return {
        "min": float(minimum.item()),
        "mean": float((total / count).item()),
        "max": float(maximum.item()),
    }


def _print(rank: int, message: str) -> None:
    if int(rank) != 0:
        return
    print(f"rank={rank} {message}", flush=True)


def _append_metrics_jsonl(output_dir: str, rank: int, record: dict[str, object]) -> None:
    if int(rank) != 0:
        return
    path = Path(output_dir) / "train_metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def train(args: argparse.Namespace) -> None:
    if int(getattr(args, "lora_r", 0) or 0) <= 0:
        raise ValueError("production trainer requires lora_r > 0")
    context = init_distributed()
    try:
        _set_seed(args.seed + context.rank)
        _write_run_config(args, context.rank, context.world_size)
        model, tokenizer, plan = _build_model_and_tokenizer(args, context.device)
        loaded_names = initialize_and_configure_trainables(model, args)
        if (
            str(getattr(args, "route_dense_teacher_source", "")) == "dense"
            and float(getattr(args, "route_dense_teacher_weight", 0)) > 0
        ) or float(getattr(args, "route_support_attn_kl_weight", 0) or 0) > 0:
            from prefilling_v2.full_dense_teacher import snapshot_frozen_dense_qk
            n_teacher = snapshot_frozen_dense_qk(model)
            _print(context.rank, f"frozen_dense_teacher_layers={n_teacher}")
        if args.initialize_from:
            _print(
                context.rank,
                f"initialized_trainables={len(loaded_names)} "
                f"initialize_from={args.initialize_from}",
            )
        if getattr(args, "route_gate_cal_init_from", None):
            _print(
                context.rank,
                "gate_cal_init_from="
                f"{args.route_gate_cal_init_from} "
                "gate_cal_trainables="
                + ",".join(
                    name
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                ),
            )
        _print(
            context.rank,
            "dense_teacher_source="
            f"{getattr(args, 'route_dense_teacher_source', 'online')} "
            f"dense_teacher_weight={float(getattr(args, 'route_dense_teacher_weight', 0)):.6g} "
            f"dense_teacher_queries={getattr(args, 'route_dense_teacher_queries', 16)} "
            f"dense_teacher_rule={getattr(args, 'route_dense_teacher_rule', None)} "
            f"detach_fusion_weights={bool(getattr(args, 'route_detach_fusion_weights', False))} "
            f"lmk_kl_ste={bool(getattr(args, 'route_lmk_kl_ste', True))} "
            f"lmk_ce_ste={bool(getattr(args, 'route_lmk_ce_ste', False))} "
            f"support_attn_kl_weight={float(getattr(args, 'route_support_attn_kl_weight', 0) or 0):.6g} "
            f"support_attn_queries={getattr(args, 'route_support_attn_queries', 0)} "
            f"support_attn_temperature={float(getattr(args, 'route_support_attn_temperature', 1.5) or 0):.4g} "
            f"support_attn_detach_gate={bool(getattr(args, 'route_support_attn_detach_gate', False))} "
            f"asymmetric_gate_ce={bool(getattr(args, 'route_asymmetric_gate_ce', False))} "
            f"balanced_view_ce={bool(getattr(args, 'route_balanced_view_ce', False))} "
            f"force_remote_unit={bool(getattr(args, 'route_force_remote_unit', False))} "
            f"force_remote_oracle_route={bool(getattr(args, 'route_force_remote_oracle_route', False))} "
            f"gate_bce_weight={float(getattr(args, 'route_gate_bce_weight', 0) or 0):.6g} "
            f"gate_ce_force={bool(getattr(args, 'route_gate_ce_force', False))} "
            f"qcal_max_grad_norm={float(getattr(args, 'route_qcal_max_grad_norm', 0) or 0):.6g} "
            f"allchunk_st_queries={int(getattr(args, 'route_allchunk_st_queries', 0))} "
            f"allchunk_st_temperature={float(getattr(args, 'route_allchunk_st_temperature', 1.0) or 1.0):.4g} "
            f"allchunk_st_temperature_end={getattr(args, 'route_allchunk_st_temperature_end', None)} "
            f"allchunk_st_anneal={getattr(args, 'route_allchunk_st_anneal', 'cosine')}",
        )
        (
            dataset,
            validation_dataset,
            corpus_metadata,
            sampler,
            collator,
            loader,
            validation_loader,
            ruler_val_loader,
        ) = _build_data(
            args, tokenizer, context.rank, context.world_size
        )
        trainable_parameters = _trainable_parameters(model)
        optimizer = torch.optim.AdamW(
            _optimizer_param_groups(model, args),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        if str(getattr(args, "lr_schedule", "cosine")) == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
            )
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=args.max_steps,
            )

        start_step = 0
        sampler_epoch = 0
        batches_in_epoch = 0
        resume_state = None
        if args.resume_from:
            resume_state = load_training_checkpoint(
                checkpoint_dir=args.resume_from,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                rank=context.rank,
                restore_rng=False,
                expected_resume_contract=_training_resume_contract(args),
            )
            start_step = int(resume_state["step"])
            sampler_epoch = int(resume_state["sampler_epoch"])
            batches_in_epoch = int(resume_state["batches_in_epoch"])

        iterator = ResumableDataIterator(
            loader,
            sampler,
            collator,
            epoch=sampler_epoch,
            batches_in_epoch=batches_in_epoch,
        )
        model.train()
        if context.enabled:
            model = DistributedDataParallel(
                _DenoisingLossWrapper(model, args),
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=args.attention_mode == "dsa",
                gradient_as_bucket_view=True,
            )
        if resume_state is not None:
            rng_states = resume_state.get("rng_states") or []
            if context.rank < len(rng_states):
                restore_rng_state(rng_states[context.rank])

        trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
        total_count = sum(parameter.numel() for parameter in model.parameters())
        _print(
            context.rank,
            f"world_size={context.world_size} dataset_packs={len(dataset)} "
            f"validation_packs={len(validation_dataset)} "
            f"route_layers={plan.route_layers} local_layers={len(plan.sliding_window_layers)} "
            f"dsa_layers={plan.dsa_layers} "
            f"dense_layers={len(plan.dense_layers)} attention_mode={plan.mode} "
            f"backend={args.dsa_backend if plan.mode == 'dsa' else args.route_backend} "
            f"dsa_semantics={DSA_SELECTED_ATTENTION_SEMANTICS if plan.mode == 'dsa' else 'n/a'} "
            f"fallback_allowed={not args.no_kernel_fallback} "
            f"chunk_topk={args.route_topk} "
            f"token_budget={getattr(args, 'route_token_budget', 0)} "
            f"token_policy={getattr(args, 'route_token_policy', 'global_qk')} "
            f"token_relaxation={getattr(args, 'route_token_relaxation', 'none')} "
            f"token_gumbel_scale={getattr(args, 'route_token_gumbel_scale', 1.0)} "
            f"route_relaxation={getattr(args, 'route_relaxation', 'none')} "
            f"route_temperature={getattr(args, 'route_temperature', 1.0)} "
            f"route_gumbel_scale={getattr(args, 'route_gumbel_scale', 1.0)} "
            f"value_fusion_beta={getattr(args, 'route_value_fusion_beta', 0.0)} "
            f"value_fusion_rank={getattr(args, 'route_value_fusion_rank', 32)} "
            f"dsa_topk={getattr(args, 'dsa_topk', 0)} "
            f"dsa_index_heads={getattr(args, 'dsa_index_heads', 0)} "
            f"dsa_aux_queries={getattr(args, 'dsa_aux_queries', 0)} "
            f"dsa_aux_loss_weight={getattr(args, 'dsa_aux_loss_weight', 0.0)} "
            f"dsa_aux_loss_scope={getattr(args, 'dsa_aux_loss_scope', 'full')} "
            f"dsa_warmup_steps={getattr(args, 'dsa_warmup_steps', 0)} "
            f"dsa_lm_loss_weight={getattr(args, 'dsa_lm_loss_weight', 1.0)} "
            f"lmk_token_mode={getattr(args, 'lmk_token_mode', 'mask')} "
            f"lmk_token_id={resolve_landmark_token_id(args, tokenizer)} "
            f"route_token={getattr(args, 'route_token', None)} "
            f"tokenizer_len={len(tokenizer)} "
            f"ruler_mix_ratio={getattr(args, 'ruler_mix_ratio', 0.0)} "
            f"route_sync_ruler_ce={bool(getattr(args, 'route_sync_ruler_ce', False))} "
            f"sync_ruler_all_tasks={bool(getattr(args, 'sync_ruler_all_tasks', False))} "
            f"ruler_answer_ce_weight={float(getattr(args, 'ruler_answer_ce_weight', 1.0) or 1.0):.4f} "
            f"ruler_val_batches={getattr(args, 'ruler_val_batches', 0)} "
            f"ruler_task_ids={','.join(str(item) for item in args.ruler_task_ids)} "
            f"trainable_scope={getattr(args, 'route_trainable_scope', 'full')} "
            f"freeze_qcal={bool(getattr(args, 'route_freeze_qcal', False))} "
            f"evidence_route_loss_weight={getattr(args, 'route_evidence_route_loss_weight', 0.0)} "
            f"evidence_token_attn_weight={getattr(args, 'route_evidence_token_attn_weight', 0.0)} "
            f"lora_targets={','.join(args.lora_target_modules)} "
            f"trainable={trainable_count}/{total_count}",
        )
        torch.cuda.reset_peak_memory_stats(context.device)

        optimizer.zero_grad(set_to_none=True)
        _configure_dsa_official_step(model, args, start_step + 1)
        if args.validation_batches > 0 or ruler_val_loader is not None:
            # DSA validation samples auxiliary query positions with torch.randperm.
            # A resumed run must not consume that RNG stream a second time before
            # its first training batch.
            with _preserve_rng_state(enabled=resume_state is not None):
                if args.validation_batches > 0:
                    if bool(getattr(args, "route_force_remote_unit", False)):
                        _print_force_remote_ablations(
                            context.rank,
                            start_step,
                            evaluate_force_remote_ablations(
                                model, args, validation_loader, context.device
                            ),
                            args.output_dir,
                        )
                    else:
                        validation_loss, validation_targets, _ = evaluate_validation_loss(
                            model, args, validation_loader, context.device
                        )
                        _print(
                            context.rank,
                            f"step={start_step} validation_loss={validation_loss:.6f} "
                            f"validation_targets={validation_targets}",
                        )
                if ruler_val_loader is not None:
                    _print_ruler_val(
                        context.rank,
                        start_step,
                        evaluate_ruler_sn(
                            model, args, ruler_val_loader, context.device
                        ),
                        args.output_dir,
                    )
        if bool(getattr(args, "eval_only", False)):
            return
        last_saved_step = start_step if Path(args.output_dir, f"step-{start_step}").is_dir() else -1
        run_end_step = min(
            args.max_steps,
            args.stop_after_steps
            if args.stop_after_steps is not None
            else args.max_steps,
        )
        if run_end_step <= start_step:
            raise ValueError(
                f"run end step {run_end_step} must exceed checkpoint step {start_step}"
            )
        clipped_steps = 0
        qcal_clipped_steps = 0
        lora_q_clipped_steps = 0
        offset_clipped_steps = 0
        scale_clipped_steps = 0
        for step in range(start_step + 1, run_end_step + 1):
            dsa_stage = _configure_dsa_official_step(model, args, step)
            _apply_dsa_official_lrs(optimizer, args, step)
            if int(getattr(args, "route_allchunk_st_queries", 0) or 0) > 0:
                from prefilling_v2.allchunk_gumbel import (
                    allchunk_st_temperature,
                    set_allchunk_st_temperature,
                )

                tau_start_raw = getattr(args, "route_allchunk_st_temperature", 1.0)
                tau_start = 1.0 if tau_start_raw is None else float(tau_start_raw)
                tau_end_raw = getattr(args, "route_allchunk_st_temperature_end", None)
                tau_end = tau_start if tau_end_raw is None else float(tau_end_raw)
                args._allchunk_st_temperature_now = allchunk_st_temperature(
                    step,
                    args.max_steps,
                    tau_start,
                    tau_end,
                    getattr(args, "route_allchunk_st_anneal", "cosine") or "cosine",
                )
                set_allchunk_st_temperature(
                    _unwrap_training_model(model), args._allchunk_st_temperature_now
                )
            cpu_window = [
                next(iterator) for _ in range(args.gradient_accumulation_steps)
            ]
            answer_alpha = float(getattr(args, "ruler_answer_ce_weight", 1.0) or 1.0)
            global_count = global_target_count(
                [weighted_supervised_count(batch, answer_alpha) for batch in cpu_window],
                device=context.device,
            )
            sync_ruler = bool(getattr(args, "route_sync_ruler_ce", False))
            all_ruler_tasks = bool(getattr(args, "sync_ruler_all_tasks", False))
            sync_task_ce = sync_ruler or all_ruler_tasks
            ruler_task_ids = [
                int(task_id) for task_id in getattr(args, "ruler_task_ids", [0, 1, 2])
            ]
            task_weight = (
                1.0 / (1 + len(ruler_task_ids)) if all_ruler_tasks else 0.5
            )
            global_task_n: dict[str, torch.Tensor] = {}
            if sync_task_ce:
                global_task_n = _sync_named_window_target_counts(
                    cpu_window,
                    all_tasks=all_ruler_tasks,
                    task_ids=ruler_task_ids,
                    device=context.device,
                )
            global_dolma_n = global_task_n.get("dolma")
            global_ruler_n = global_task_n.get("ruler")
            local_loss_for_log = torch.zeros(
                (), device=context.device, dtype=torch.float64
            )
            window_sample_ids: list[int] = []
            window_dolma_ce = torch.zeros((), device=context.device, dtype=torch.float64)
            window_ruler_ce = torch.zeros((), device=context.device, dtype=torch.float64)
            window_dolma_n = torch.zeros((), device=context.device, dtype=torch.float64)
            window_ruler_n = torch.zeros((), device=context.device, dtype=torch.float64)
            window_ruler_views = torch.zeros(
                (), device=context.device, dtype=torch.float64
            )
            window_named_ce: dict[str, torch.Tensor] = {}
            window_named_n: dict[str, torch.Tensor] = {}

            for micro_step, cpu_batch in enumerate(cpu_window):
                batch = _move_batch(cpu_batch, context.device)
                labels = batch["labels"]
                target_count = int(batch["target_count"].sum().item())
                label_count = int(labels.ne(-100).sum().item())
                if target_count != label_count:
                    raise RuntimeError(
                        f"collator target_count={target_count} but labels={label_count}"
                    )
                window_sample_ids.extend(int(item) for item in batch["sample_ids"].tolist())
                last_micro = micro_step + 1 >= args.gradient_accumulation_steps
                asymmetric = bool(getattr(args, "route_asymmetric_gate_ce", False))
                balanced = bool(getattr(args, "route_balanced_view_ce", False))
                if asymmetric or balanced:
                    local_batch, remote_batch = _split_gate_views(batch)
                    train_root = _unwrap_training_model(model)
                    if asymmetric:
                        try:
                            _set_fusion_detach_for_forward(model, True)
                            with model.no_sync() if context.enabled else contextlib.nullcontext():
                                with torch.autocast(
                                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                                ):
                                    local_sum, _ = denoising_loss_sum_for_batch(
                                        model, args, local_batch
                                    )
                                    local_backward = scale_local_loss_for_ddp(
                                        local_sum,
                                        global_count=global_count,
                                        world_size=context.world_size,
                                    )
                                local_backward.backward()
                            _set_fusion_detach_for_forward(model, False)
                            sync_context = (
                                model.no_sync()
                                if context.enabled and not last_micro
                                else contextlib.nullcontext()
                            )
                            with sync_context:
                                with torch.autocast(
                                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                                ):
                                    remote_sum, _ = denoising_loss_sum_for_batch(
                                        model, args, remote_batch
                                    )
                                    remote_backward = scale_local_loss_for_ddp(
                                        remote_sum,
                                        global_count=global_count,
                                        world_size=context.world_size,
                                    )
                                remote_backward.backward()
                        finally:
                            _set_fusion_detach_for_forward(model, False)
                        loss_sum = local_sum.detach() + remote_sum.detach()
                    else:
                        _set_fusion_detach_for_forward(model, False)
                        with model.no_sync() if context.enabled else contextlib.nullcontext():
                            with torch.autocast(
                                device_type="cuda", dtype=torch.bfloat16, enabled=True
                            ):
                                local_sum, _ = denoising_loss_sum_for_batch(
                                    model, args, local_batch
                                )
                                local_backward = scale_local_loss_for_ddp(
                                    local_sum,
                                    global_count=global_count,
                                    world_size=context.world_size,
                                )
                            local_backward.backward()
                        sync_context = (
                            model.no_sync()
                            if context.enabled and not last_micro
                            else contextlib.nullcontext()
                        )
                        with sync_context:
                            with torch.autocast(
                                device_type="cuda", dtype=torch.bfloat16, enabled=True
                            ):
                                remote_sum, _ = denoising_loss_sum_for_batch(
                                    model, args, remote_batch
                                )
                                remote_backward = scale_local_loss_for_ddp(
                                    remote_sum,
                                    global_count=global_count,
                                    world_size=context.world_size,
                                )
                            remote_backward.backward()
                        loss_sum = local_sum.detach() + remote_sum.detach()
                    local_n = int(local_batch["target_count"].sum().item())
                    remote_n = int(remote_batch["target_count"].sum().item())
                    train_root._last_local_ce = float(
                        (local_sum.detach() / max(local_n, 1)).item()
                    )
                    train_root._last_remote_ce = float(
                        (remote_sum.detach() / max(remote_n, 1)).item()
                    )
                    denom = max(local_n + remote_n, 1)
                    train_root._last_balanced_ce = float(
                        (
                            float(local_sum.detach().item())
                            + float(remote_sum.detach().item())
                        )
                        / denom
                    )
                    split_task_stats = False
                elif sync_task_ce:
                    task_batches = list(
                        _iter_sync_task_batches(
                            batch,
                            all_tasks=all_ruler_tasks,
                            task_ids=ruler_task_ids,
                        )
                    )
                    train_root = _unwrap_training_model(model)
                    if args.attention_mode == "route":
                        _set_fusion_detach_for_forward(model, False)
                    loss_sum = None
                    task_means: dict[str, float] = {}
                    for task_index, (task_name, task_batch) in enumerate(task_batches):
                        last_task = task_index + 1 >= len(task_batches)
                        sync_context = (
                            model.no_sync()
                            if context.enabled and not (last_micro and last_task)
                            else contextlib.nullcontext()
                        )
                        with sync_context:
                            with torch.autocast(
                                device_type="cuda", dtype=torch.bfloat16, enabled=True
                            ):
                                task_sum, _ = denoising_loss_sum_for_batch(
                                    model, args, task_batch
                                )
                                task_n = task_batch["target_count"].sum().to(
                                    dtype=task_sum.dtype
                                )
                                if float(task_n.item()) <= 0:
                                    raise RuntimeError(
                                        f"sync ruler CE {task_name} view has no targets"
                                    )
                                task_backward = scale_sync_task_loss(
                                    task_sum,
                                    global_count=global_task_n[task_name],
                                    world_size=context.world_size,
                                    task_weight=task_weight,
                                )
                            task_backward.backward()
                        if loss_sum is None:
                            loss_sum = task_sum.detach()
                        else:
                            loss_sum = loss_sum + task_sum.detach()
                        task_means[task_name] = float(
                            (task_sum.detach() / task_n.detach()).item()
                        )
                        window_named_ce[task_name] = window_named_ce.get(
                            task_name,
                            torch.zeros((), device=context.device, dtype=torch.float64),
                        ) + task_sum.detach().double()
                        window_named_n[task_name] = window_named_n.get(
                            task_name,
                            torch.zeros((), device=context.device, dtype=torch.float64),
                        ) + task_n.detach().double()
                        if task_name == "dolma":
                            window_dolma_ce += task_sum.detach().double()
                            window_dolma_n += task_n.detach().double()
                        else:
                            window_ruler_ce += task_sum.detach().double()
                            window_ruler_n += task_n.detach().double()
                            window_ruler_views += torch.tensor(
                                float(task_batch["view_ids"].numel()),
                                device=context.device,
                                dtype=torch.float64,
                            )
                    train_root._last_local_ce = task_means["dolma"]
                    if all_ruler_tasks:
                        ruler_means = [
                            task_means[f"ruler{task_id}"] for task_id in ruler_task_ids
                        ]
                        train_root._last_remote_ce = float(sum(ruler_means) / len(ruler_means))
                    else:
                        train_root._last_remote_ce = task_means["ruler"]
                    train_root._last_balanced_ce = float(task_weight) * (
                        train_root._last_local_ce
                        + (
                            sum(
                                task_means[f"ruler{task_id}"]
                                for task_id in ruler_task_ids
                            )
                            if all_ruler_tasks
                            else train_root._last_remote_ce
                        )
                    )
                    split_task_stats = True
                else:
                    sync_context = (
                        model.no_sync()
                        if context.enabled
                        and micro_step + 1 < args.gradient_accumulation_steps
                        else contextlib.nullcontext()
                    )
                    with sync_context:
                        with torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16, enabled=True
                        ):
                            loss_sum, _ = denoising_loss_sum_for_batch(
                                model,
                                args,
                                batch,
                            )
                            backward_loss = scale_local_loss_for_ddp(
                                loss_sum,
                                global_count=global_count,
                                world_size=context.world_size,
                            )
                        backward_loss.backward()
                    split_task_stats = False
                local_loss_for_log += loss_sum.detach().double()
                if not split_task_stats:
                    train_core = _unwrap_training_model(model)
                    window_dolma_ce += float(
                        getattr(train_core, "_last_dolma_ce_sum", 0.0)
                    )
                    window_ruler_ce += float(
                        getattr(train_core, "_last_ruler_ce_sum", 0.0)
                    )
                    window_dolma_n += float(
                        getattr(train_core, "_last_dolma_targets", 0)
                    )
                    window_ruler_n += float(
                        getattr(train_core, "_last_ruler_targets", 0)
                    )
                    window_ruler_views += float(
                        getattr(train_core, "_last_ruler_views", 0)
                    )

            audit_frozen_gradients(model)
            gradient_stats = _gradient_audit(
                model,
                require_lora_nonzero=(
                    getattr(args, "route_trainable_scope", "full")
                    not in _FROZEN_LORA_SCOPES
                    and not (
                        args.attention_mode == "dsa"
                        and float(args.dsa_lm_loss_weight) == 0
                    )
                ),
            )
            if (
                getattr(args, "route_trainable_scope", "full")
                in _QCAL_SCOPES
                and gradient_stats["qcal_nonzero"] == 0
                and float(getattr(args, "route_support_attn_kl_weight", 0) or 0) <= 0
                and not bool(getattr(args, "route_gate_offset_only", False))
                and not bool(getattr(args, "route_gate_affine_only", False))
                and not bool(getattr(args, "route_freeze_qcal", False))
            ):
                raise RuntimeError("all Q-Cal gradients are zero or missing")
            if (
                args.attention_mode == "route"
                and gradient_stats["entropy_nonzero"]
                != gradient_stats["entropy_tensors"]
            ):
                raise RuntimeError(
                    "one or more route entropy parameters has a zero/missing gradient: "
                    f"{gradient_stats}"
                )
            if (
                args.attention_mode == "dsa"
                and gradient_stats["dsa_indexer_nonzero"] == 0
            ):
                raise RuntimeError(
                    "all Dream DSA indexer gradients are zero/missing: "
                    f"{gradient_stats}"
                )
            group_grad_norms = _gradient_group_norms(model)
            lora_q_clip_norm = _clip_matching_grad_norm(
                model,
                float(getattr(args, "route_lora_q_max_grad_norm", 0.0) or 0.0),
                lambda name: "lora_" in name and ".q_proj." in name,
            )
            qcal_clip_norm = _clip_matching_grad_norm(
                model,
                float(getattr(args, "route_qcal_max_grad_norm", 0.0) or 0.0),
                lambda name: _is_qcal_param(name),
            )
            clip_each = bool(getattr(args, "route_gate_cal_clip_each", False))
            offset_layer_clip = _as_float_map(
                getattr(args, "route_gate_offset_layer_clip", None)
            )
            scale_layer_clip = _as_float_map(
                getattr(args, "route_gate_scale_layer_clip", None)
            )
            if offset_layer_clip or scale_layer_clip:
                offset_clip_norm = _clip_gate_cal_params(
                    model,
                    suffix="fusion_gate_offset",
                    default_max_norm=float(
                        getattr(args, "route_gate_offset_max_grad_norm", 0.0) or 0.0
                    ),
                    layer_clips=offset_layer_clip,
                )
                scale_clip_norm = _clip_gate_cal_params(
                    model,
                    suffix="fusion_gate_scale",
                    default_max_norm=float(
                        getattr(args, "route_gate_scale_max_grad_norm", 0.0) or 0.0
                    ),
                    layer_clips=scale_layer_clip,
                )
            else:
                offset_clip_norm = _clip_matching_grad_norm(
                    model,
                    float(getattr(args, "route_gate_offset_max_grad_norm", 0.0) or 0.0),
                    lambda name: name.endswith("fusion_gate_offset"),
                    each=clip_each,
                )
                scale_clip_norm = _clip_matching_grad_norm(
                    model,
                    float(getattr(args, "route_gate_scale_max_grad_norm", 0.0) or 0.0),
                    lambda name: name.endswith("fusion_gate_scale"),
                    each=clip_each,
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, args.max_grad_norm
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite grad norm at step {step}: {grad_norm}")
            clip_applied = float(grad_norm) > args.max_grad_norm
            clipped_steps += int(clip_applied)
            lora_q_max_norm = float(getattr(args, "route_lora_q_max_grad_norm", 0.0) or 0.0)
            qcal_max_norm = float(getattr(args, "route_qcal_max_grad_norm", 0.0) or 0.0)
            if lora_q_max_norm > 0:
                lora_q_clipped_steps += int(float(lora_q_clip_norm) > lora_q_max_norm)
            if qcal_max_norm > 0:
                qcal_clipped_steps += int(float(qcal_clip_norm) > qcal_max_norm)
            offset_max_norm = float(
                getattr(args, "route_gate_offset_max_grad_norm", 0.0) or 0.0
            )
            if offset_max_norm > 0:
                offset_clipped_steps += int(float(offset_clip_norm) > offset_max_norm)
            scale_max_norm = float(
                getattr(args, "route_gate_scale_max_grad_norm", 0.0) or 0.0
            )
            if scale_max_norm > 0:
                scale_clipped_steps += int(float(scale_clip_norm) > scale_max_norm)
            optimizer.step()
            scheduler.step()
            offset_abs_max = float(getattr(args, "route_gate_offset_abs_max", 0.0) or 0.0)
            scale_min = float(getattr(args, "route_gate_scale_min", 0.05) or 0.05)
            scale_max = float(getattr(args, "route_gate_scale_max", 4.0) or 4.0)
            unwrapped = model.module if hasattr(model, "module") else model
            for name, parameter in unwrapped.named_parameters():
                if offset_abs_max > 0 and name.endswith("fusion_gate_offset"):
                    parameter.data.clamp_(-offset_abs_max, offset_abs_max)
                if name.endswith("fusion_gate_scale"):
                    parameter.data.clamp_(scale_min, scale_max)
            optimizer.zero_grad(set_to_none=True)

            global_loss_sum = all_reduce_detached_sum(local_loss_for_log)
            global_mean_loss = float((global_loss_sum / global_count).item())
            fallback_count = _kernel_fallback_count(model)
            if fallback_count:
                raise RuntimeError(f"kernel fallback count became {fallback_count}")
            if step == start_step + 1:
                gathered_ids = all_gather_object(window_sample_ids)
                if context.is_main:
                    _print(
                        0,
                        f"invocation_first_step={step} "
                        f"first_window_sample_ids={gathered_ids}",
                    )
            if step == 1 or step % args.log_steps == 0:
                allocation_stats = _token_allocation_stats(model)
                peak_allocated_mib = torch.cuda.max_memory_allocated(
                    context.device
                ) / (1024**2)
                peak_reserved_mib = torch.cuda.max_memory_reserved(
                    context.device
                ) / (1024**2)
                g_dolma_ce = all_reduce_detached_sum(window_dolma_ce)
                g_ruler_ce = all_reduce_detached_sum(window_ruler_ce)
                g_dolma_n = all_reduce_detached_sum(window_dolma_n)
                g_ruler_n = all_reduce_detached_sum(window_ruler_n)
                g_ruler_views = all_reduce_detached_sum(window_ruler_views)
                dolma_ce_mean = (
                    float((g_dolma_ce / g_dolma_n).item()) if float(g_dolma_n.item()) > 0 else 0.0
                )
                ruler_ce_mean = (
                    float((g_ruler_ce / g_ruler_n).item()) if float(g_ruler_n.item()) > 0 else 0.0
                )
                task_ce_csv = ""
                if bool(getattr(args, "sync_ruler_all_tasks", False)):
                    named_means = []
                    for task_id, label in ((0, "sn"), (1, "mkmq"), (2, "vt")):
                        name = f"ruler{task_id}"
                        local_ce = window_named_ce.get(name)
                        local_n = window_named_n.get(name)
                        if local_ce is None or local_n is None:
                            named_means.append(0.0)
                            task_ce_csv += f" {label}_ce=0.000000"
                            continue
                        g_ce = all_reduce_detached_sum(local_ce)
                        g_n = all_reduce_detached_sum(local_n)
                        mean = (
                            float((g_ce / g_n).item()) if float(g_n.item()) > 0 else 0.0
                        )
                        named_means.append(mean)
                        task_ce_csv += f" {label}_ce={mean:.6f}"
                    global_mean_loss = 0.25 * (
                        dolma_ce_mean + named_means[0] + named_means[1] + named_means[2]
                    )
                    ruler_ce_mean = float(sum(named_means) / max(len(named_means), 1))
                elif bool(getattr(args, "route_sync_ruler_ce", False)):
                    global_mean_loss = 0.5 * dolma_ce_mean + 0.5 * ruler_ce_mean
                from prefilling_v2.force_remote import (
                    collect_fusion_gate_offsets,
                    collect_fusion_gate_scales,
                    collect_labeled_gate_vectors,
                    layer_gate_metrics,
                    summarize_gate_vectors,
                )

                packed_gate = collect_labeled_gate_vectors(
                    _unwrap_training_model(model)
                )
                train_gate = summarize_gate_vectors(
                    packed_gate["logit"] if packed_gate else torch.zeros((0,)),
                    packed_gate["mass"] if packed_gate else torch.zeros((0,)),
                )
                train_layer = layer_gate_metrics(
                    packed_gate["layers"] if packed_gate else []
                )
                offset_values = collect_fusion_gate_offsets(
                    _unwrap_training_model(model)
                )
                offset_csv = (
                    ",".join(f"{value:.4f}" for value in offset_values)
                    if offset_values
                    else "na"
                )
                scale_values = collect_fusion_gate_scales(
                    _unwrap_training_model(model)
                )
                scale_csv = (
                    ",".join(f"{value:.4f}" for value in scale_values)
                    if scale_values
                    else "na"
                )
                layer_w_csv = _format_layer_field(train_layer, "live_w_remote")
                layer_logit_csv = _format_layer_field(train_layer, "gate_logit_mean")
                layer_logit_std_csv = _format_layer_field(train_layer, "gate_logit_std")
                seen = max(step - start_step, 1)
                lmk_grad_norm = (
                    group_grad_norms["external_lmk"] ** 2
                    + group_grad_norms["lmk_type"] ** 2
                ) ** 0.5
                _print(
                    context.rank,
                    f"step={step} loss={global_mean_loss:.6f} "
                    f"global_targets={float(global_count.item()):.1f} "
                    f"dolma_ce={dolma_ce_mean:.6f} "
                    f"ruler_ce={ruler_ce_mean:.6f}"
                    f"{task_ce_csv} "
                    f"dolma_targets={int(g_dolma_n.item())} "
                    f"ruler_targets={int(g_ruler_n.item())} "
                    f"ruler_views={int(g_ruler_views.item())} "
                    f"ruler_answer_ce_weight={float(getattr(args, 'ruler_answer_ce_weight', 1.0) or 1.0):.4f} "
                    f"allchunk_st_temperature={float(getattr(args, '_allchunk_st_temperature_now', getattr(args, 'route_allchunk_st_temperature', 1.0) or 1.0)):.6f} "
                    f"grad_norm={float(grad_norm):.6f} "
                    f"lora_grad_norm={group_grad_norms['lora']:.6f} "
                    f"q_grad_norm={group_grad_norms['lora_q']:.6f} "
                    f"k_grad_norm={group_grad_norms['lora_k']:.6f} "
                    f"v_grad_norm={group_grad_norms['lora_v']:.6f} "
                    f"o_grad_norm={group_grad_norms['lora_o']:.6f} "
                    f"entropy_grad_norm={group_grad_norms['entropy']:.6f} "
                    f"external_lmk_grad_norm={group_grad_norms['external_lmk']:.6f} "
                    f"lmk_type_grad_norm={group_grad_norms['lmk_type']:.6f} "
                    f"lmk_grad_norm={lmk_grad_norm:.6f} "
                    f"dsa_stage={dsa_stage} "
                    f"dsa_indexer_grad_norm={group_grad_norms['dsa_indexer']:.6f} "
                    f"dsa_index_loss={float(getattr(_unwrap_training_model(model), '_last_dsa_index_loss', 0.0)):.6f} "
                    f"route_chunk_kl={float(getattr(_unwrap_training_model(model), '_last_route_chunk_loss', 0.0)):.6f} "
                    f"route_support_attn_kl={float(getattr(_unwrap_training_model(model), '_last_route_support_attn_kl', 0.0)):.6f} "
                    f"full_dense_teacher_kl={float(getattr(_unwrap_training_model(model), '_last_full_dense_teacher_kl', 0.0)):.6f} "
                    f"route_evidence_route_loss={float(getattr(_unwrap_training_model(model), '_last_route_evidence_route_loss', 0.0)):.6f} "
                    f"route_evidence_token_attn={float(getattr(_unwrap_training_model(model), '_last_route_evidence_token_attn_loss', 0.0)):.6f} "
                    f"remote_needle_qk_mass={float(getattr(_unwrap_training_model(model), '_last_remote_needle_qk_mass', 0.0)):.6f} "
                    f"evidence_local_weight={float(getattr(_unwrap_training_model(model), '_last_evidence_local_weight', 0.0)):.6f} "
                    f"route_gate_bce={float(getattr(_unwrap_training_model(model), '_last_route_gate_bce', 0.0)):.6f} "
                    f"route_w_remote={float(getattr(_unwrap_training_model(model), '_last_route_w_remote', 0.0)):.6f} "
                    f"train_frac_w_gt_0.8={train_gate['live_frac_w_gt_0.8']:.6f} "
                    f"gate_logit_mean={train_gate['gate_logit_mean']:.6f} "
                    f"gate_logit_std={train_gate['gate_logit_std']:.6f} "
                    f"gate_logit_p50={train_gate['gate_logit_p50']:.6f} "
                    f"gate_logit_p90={train_gate['gate_logit_p90']:.6f} "
                    f"fusion_gate_offset={_mean_fusion_gate_offset(model):.6f} "
                    f"fusion_gate_offsets={offset_csv} "
                    f"fusion_gate_scales={scale_csv} "
                    f"layer_w_remote={layer_w_csv} "
                    f"layer_gate_logit_mean={layer_logit_csv} "
                    f"layer_gate_logit_std={layer_logit_std_csv} "
                    f"offset_preclip_norm={offset_clip_norm:.6f} "
                    f"scale_preclip_norm={scale_clip_norm:.6f} "
                    f"local_ce={float(getattr(_unwrap_training_model(model), '_last_local_ce', 0.0)):.6f} "
                    f"remote_ce={float(getattr(_unwrap_training_model(model), '_last_remote_ce', 0.0)):.6f} "
                    f"balanced_ce={float(getattr(_unwrap_training_model(model), '_last_balanced_ce', 0.0)):.6f} "
                    f"qcal_grad_norm={group_grad_norms['qcal']:.6f} "
                    f"qcal_preclip_norm={qcal_clip_norm:.6f} "
                    f"lora_q_preclip_norm={lora_q_clip_norm:.6f} "
                    f"other_grad_norm={group_grad_norms['other']:.6f} "
                    f"clip_applied={int(clip_applied)} "
                    f"clip_fraction={clipped_steps / seen:.4f} "
                    f"qcal_clip_fraction={qcal_clipped_steps / seen:.4f} "
                    f"lora_q_clip_fraction={lora_q_clipped_steps / seen:.4f} "
                    f"offset_clip_fraction={offset_clipped_steps / seen:.4f} "
                    f"scale_clip_fraction={scale_clipped_steps / seen:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.8g} fallback_count=0 "
                    f"lora_nonzero_grads={gradient_stats['lora_nonzero']} "
                    f"entropy_nonzero_grads={gradient_stats['entropy_nonzero']}/"
                    f"{gradient_stats['entropy_tensors']} "
                    f"dsa_indexer_nonzero_grads={gradient_stats['dsa_indexer_nonzero']}/"
                    f"{gradient_stats['dsa_indexer_tensors']} "
                    f"peak_allocated_mib={peak_allocated_mib:.1f} "
                    f"peak_reserved_mib={peak_reserved_mib:.1f}"
                    + (
                        ""
                        if allocation_stats is None
                        else " "
                        f"token_alloc_min={allocation_stats['min']:.0f} "
                        f"token_alloc_mean={allocation_stats['mean']:.3f} "
                        f"token_alloc_max={allocation_stats['max']:.0f}"
                    ),
                )
                _append_metrics_jsonl(
                    args.output_dir,
                    context.rank,
                    {
                        "step": int(step),
                        "split": "train",
                        "loss": float(global_mean_loss),
                        "dolma_ce": float(dolma_ce_mean),
                        "ruler_ce": float(ruler_ce_mean),
                        "dolma_targets": int(g_dolma_n.item()),
                        "ruler_targets": int(g_ruler_n.item()),
                        "ruler_views": int(g_ruler_views.item()),
                        "lr": float(scheduler.get_last_lr()[0]),
                        "grad_norm": float(grad_norm),
                        "local_ce": float(
                            getattr(_unwrap_training_model(model), "_last_local_ce", 0.0)
                        ),
                        "remote_ce": float(
                            getattr(_unwrap_training_model(model), "_last_remote_ce", 0.0)
                        ),
                        "route_w_remote": float(
                            getattr(
                                _unwrap_training_model(model), "_last_route_w_remote", 0.0
                            )
                        ),
                    },
                )
                if context.rank == 0:
                    trace_path = Path(args.output_dir) / "gate_layer_trace.jsonl"
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    record = {
                        "step": int(step),
                        "split": "train",
                        "offsets": offset_values,
                        "scales": scale_values,
                        **train_gate,
                        **train_layer,
                    }
                    with trace_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")

            if args.eval_steps > 0 and step % args.eval_steps == 0:
                if args.validation_batches > 0:
                    if bool(getattr(args, "route_force_remote_unit", False)):
                        _print_force_remote_ablations(
                            context.rank,
                            step,
                            evaluate_force_remote_ablations(
                                model, args, validation_loader, context.device
                            ),
                            args.output_dir,
                        )
                    else:
                        validation_loss, validation_targets, _ = evaluate_validation_loss(
                            model, args, validation_loader, context.device
                        )
                        _print(
                            context.rank,
                            f"step={step} validation_loss={validation_loss:.6f} "
                            f"validation_targets={validation_targets}",
                        )
                        _append_metrics_jsonl(
                            args.output_dir,
                            context.rank,
                            {
                                "step": int(step),
                                "split": "val",
                                "validation_loss": float(validation_loss),
                                "validation_targets": int(validation_targets),
                            },
                        )
                if ruler_val_loader is not None:
                    _print_ruler_val(
                        context.rank,
                        step,
                        evaluate_ruler_sn(
                            model, args, ruler_val_loader, context.device
                        ),
                        args.output_dir,
                    )

            if args.save_steps > 0 and step % args.save_steps == 0:
                save_training_checkpoint(
                    model=_unwrap_parallel_training_model(model),
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=args.output_dir,
                    step=step,
                    sampler_epoch=iterator.epoch,
                    batches_in_epoch=iterator.batches_in_epoch,
                    config=vars(args),
                    dataset_manifest_hash=corpus_metadata["source_manifest_sha256"],
                    rank=context.rank,
                    world_size=context.world_size,
                    resume_contract=_training_resume_contract(args),
                )
                last_saved_step = step

        if last_saved_step != run_end_step:
            save_training_checkpoint(
                model=_unwrap_parallel_training_model(model),
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                output_dir=args.output_dir,
                step=run_end_step,
                sampler_epoch=iterator.epoch,
                batches_in_epoch=iterator.batches_in_epoch,
                config=vars(args),
                dataset_manifest_hash=corpus_metadata["source_manifest_sha256"],
                rank=context.rank,
                world_size=context.world_size,
                resume_contract=_training_resume_contract(args),
            )
        barrier()
    finally:
        destroy_distributed(context)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
