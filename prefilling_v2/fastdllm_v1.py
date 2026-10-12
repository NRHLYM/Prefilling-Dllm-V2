"""Fast-dLLM v1 style cached decoding for the trained Dream+route model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from prefilling_v2.attention import (
    KernelDreamFullRouteAttention,
    KernelDreamSlidingWindowAttention,
)
from prefilling_v2.dsa_attention import DreamDsaAttention
from prefilling_v2.nsa_attention import DreamNsaAttention
from prefilling_v2.fastdllm_cache import RouteModelCache
from prefilling_v2.longbench_eval import FastDLLMBlockLayout


def cached_attention_forward(*args, **kwargs):
    # CPU schedule tests and manifest preparation do not load CUDA kernels.
    from prefilling_v2.fastdllm_attention import cached_attention_forward as forward
    return forward(*args, **kwargs)


@dataclass(frozen=True)
class PrefillOutput:
    logits: torch.Tensor
    cache: RouteModelCache | Any


@dataclass(frozen=True)
class CachedForwardOutput:
    logits: torch.Tensor
    query_positions: torch.Tensor
    routing_calls: int


@dataclass(frozen=True)
class FastDLLMGenerationStats:
    full_prefills: int
    cached_forwards: int
    routing_calls: int
    recomputed_tokens: int
    peak_memory_bytes: int


def _validate_logits(logits: torch.Tensor, mask_token_id: int) -> None:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("logits must have shape [1,N,V]")
    if logits.shape[-1] < 2:
        raise ValueError("confidence decoding requires at least two vocabulary items")
    if mask_token_id < 0 or mask_token_id >= logits.shape[-1]:
        raise ValueError("mask_token_id is outside the logits vocabulary")


def select_confidence_transfers(
    logits: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    threshold: float,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select every confident active row and always retain the best row."""

    _validate_logits(logits, mask_token_id)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0,1], got {threshold}")
    active_rows = active_rows.to(device=logits.device, dtype=torch.long)
    if active_rows.ndim != 1 or active_rows.numel() == 0:
        raise ValueError("active_rows must be a nonempty one-dimensional tensor")
    if torch.unique(active_rows).numel() != active_rows.numel():
        raise ValueError("active_rows must be unique")
    if bool(((active_rows < 0) | (active_rows >= logits.shape[1])).any()):
        raise ValueError("active_rows are outside the logits row range")

    candidates = logits[0].index_select(0, active_rows).float().clone()
    candidates[:, mask_token_id] = float("-inf")
    probabilities = torch.softmax(candidates, dim=-1)
    confidence, token_ids = probabilities.max(dim=-1)
    if not torch.isfinite(confidence).all():
        raise RuntimeError("candidate probabilities contain non-finite values")

    selected = confidence >= threshold
    selected[torch.argmax(confidence)] = True
    return (
        active_rows[selected],
        token_ids[selected],
        confidence[selected],
    )


def _rows_for_positions(
    query_positions: torch.Tensor,
    requested_positions: torch.Tensor,
) -> torch.Tensor:
    requested_positions = requested_positions.to(
        device=query_positions.device,
        dtype=torch.long,
    )
    rows = torch.searchsorted(query_positions, requested_positions)
    if bool((rows >= query_positions.numel()).any()) or not torch.equal(
        query_positions.index_select(0, rows),
        requested_positions,
    ):
        raise ValueError("requested positions are missing from partial queries")
    return rows


def _unwrap_dream_model(model: torch.nn.Module) -> torch.nn.Module:
    candidate = model
    get_base_model = getattr(candidate, "get_base_model", None)
    if callable(get_base_model):
        candidate = get_base_model()
    if hasattr(candidate, "model") and hasattr(candidate.model, "layers"):
        return candidate
    if (
        hasattr(candidate, "model")
        and hasattr(candidate.model, "model")
        and hasattr(candidate.model.model, "layers")
    ):
        return candidate.model
    raise TypeError("expected a DreamModel or PEFT-wrapped DreamModel")


def _model_config(model: torch.nn.Module):
    unwrapped = model.module if hasattr(model, "module") else model
    return getattr(unwrapped, "config", None), unwrapped


def _landmark_mask_from_positions(
    input_ids: torch.Tensor,
    landmark_positions: torch.Tensor | None,
    *,
    lmk_token_id: int,
) -> torch.Tensor:
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if landmark_positions is None:
        mask = input_ids.eq(int(lmk_token_id))
    else:
        positions = landmark_positions.to(device=input_ids.device, dtype=torch.long)
        if positions.ndim != 1:
            raise ValueError("landmark_positions must be one-dimensional")
        if positions.numel():
            if bool(((positions < 0) | (positions >= input_ids.shape[1])).any()):
                raise ValueError("landmark_positions are outside the sequence")
            mask[:, positions] = True
            if torch.any(mask & input_ids.ne(int(lmk_token_id))):
                raise ValueError("landmark_positions mark tokens that are not landmarks")
    return mask


def _prepare_landmark_inputs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    landmark_positions: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    config, unwrapped = _model_config(model)
    embed = unwrapped.get_input_embeddings()
    input_ids = input_ids.to(next(embed.parameters()).device)
    mode = getattr(config, "dream_route_lmk_token_mode", "mask")
    if mode not in {"external", "mask_type"}:
        return input_ids, None
    lmk_token_id = int(getattr(config, "dream_route_lmk_token_id"))
    lmk_mask = _landmark_mask_from_positions(
        input_ids,
        landmark_positions if mode == "mask_type" else None,
        lmk_token_id=lmk_token_id,
    )
    if not bool(lmk_mask.any()):
        return input_ids, None
    if mode == "external":
        if not hasattr(unwrapped, "dream_route_lmk_embed"):
            raise RuntimeError("external LMK mode is missing dream_route_lmk_embed")
        safe_ids = input_ids.masked_fill(lmk_mask, 0)
        token_embeds = unwrapped.get_input_embeddings()(safe_ids)
        replacement = unwrapped.dream_route_lmk_embed.to(
            device=token_embeds.device, dtype=token_embeds.dtype
        )
        inputs_embeds = torch.where(
            lmk_mask.unsqueeze(-1),
            replacement.view(1, 1, -1),
            token_embeds,
        )
    else:
        if not hasattr(unwrapped, "dream_route_lmk_type_embed"):
            raise RuntimeError("mask_type LMK mode is missing dream_route_lmk_type_embed")
        token_embeds = unwrapped.get_input_embeddings()(input_ids)
        offset = unwrapped.dream_route_lmk_type_embed.to(
            device=token_embeds.device, dtype=token_embeds.dtype
        )
        inputs_embeds = token_embeds + (
            lmk_mask.unsqueeze(-1).to(device=token_embeds.device, dtype=token_embeds.dtype)
            * offset.view(1, 1, -1)
        )
    return None, inputs_embeds


def _embed_partial_inputs(
    model: torch.nn.Module,
    core: torch.nn.Module,
    input_ids: torch.Tensor,
    query_positions: torch.Tensor,
    landmark_positions: torch.Tensor | None,
) -> torch.Tensor:
    partial_ids = input_ids.index_select(1, query_positions)
    config, unwrapped = _model_config(model)
    mode = getattr(config, "dream_route_lmk_token_mode", "mask")
    lookup_ids = partial_ids
    if mode == "external":
        # External landmark IDs are outside the token embedding vocabulary.
        lmk_token_id = int(getattr(config, "dream_route_lmk_token_id"))
        is_landmark = partial_ids.eq(lmk_token_id)
        lookup_ids = partial_ids.masked_fill(is_landmark, 0)
    hidden_states = core.model.embed_tokens(lookup_ids)
    if mode == "mask_type" and landmark_positions is not None:
        positions = landmark_positions.to(device=query_positions.device, dtype=torch.long)
        is_landmark = (query_positions[:, None] == positions[None, :]).any(dim=1)
        if bool(is_landmark.any()):
            if not hasattr(unwrapped, "dream_route_lmk_type_embed"):
                raise RuntimeError("mask_type LMK mode is missing dream_route_lmk_type_embed")
            offset = unwrapped.dream_route_lmk_type_embed.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            hidden_states = hidden_states + (
                is_landmark.view(1, -1, 1).to(hidden_states.dtype)
                * offset.view(1, 1, -1)
            )
    elif mode == "external":
        if bool(is_landmark.any()):
            if not hasattr(unwrapped, "dream_route_lmk_embed"):
                raise RuntimeError("external LMK mode is missing dream_route_lmk_embed")
            replacement = unwrapped.dream_route_lmk_embed.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            hidden_states = torch.where(
                is_landmark.unsqueeze(-1),
                replacement.view(1, 1, -1),
                hidden_states,
            )
    return hidden_states


class DreamRouteFastDLLM:
    """Two-block confidence decoder with compact, cache-aware route forwards."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        mask_token_id: int,
        threshold: float = 0.9,
        use_cache: bool = True,
        bootstrap: str = "first_token",
    ) -> None:
        if mask_token_id < 0:
            raise ValueError("mask_token_id must be non-negative")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0,1]")
        self.model = model
        self.mask_token_id = int(mask_token_id)
        self.threshold = float(threshold)
        self.use_cache = bool(use_cache)
        if bootstrap not in {"first_token", "confidence"}:
            raise ValueError("bootstrap must be first_token or confidence")
        self.bootstrap = bootstrap

    def _core(self) -> torch.nn.Module:
        return _unwrap_dream_model(self.model)

    def _attention_modules(self) -> tuple[torch.nn.Module, ...]:
        core = self._core()
        modules = tuple(layer.self_attn for layer in core.model.layers)
        if not modules:
            raise TypeError("Dream model has no attention modules")
        kernel_ok = all(
            isinstance(
                module,
                (KernelDreamSlidingWindowAttention, KernelDreamFullRouteAttention,
                 DreamDsaAttention, DreamNsaAttention),
            )
            for module in modules
        )
        dense_ok = all(
            module.__class__.__name__ == "DenseDreamAttentionAdapter"
            for module in modules
        )
        if not kernel_ok and not dense_ok:
            raise TypeError("all Dream layers must use route, DSA, NSA, sliding, or dense attention")
        if dense_ok and self.use_cache:
            raise TypeError("dense Fast-dLLM decoding requires use_cache=False")
        return modules

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        landmark_positions: torch.Tensor | None = None,
    ) -> PrefillOutput:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("prefill input_ids must have shape [1,L]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if position_ids.shape != input_ids.shape or position_ids.dtype != torch.long:
            raise ValueError("position_ids must be torch.long and match input_ids")

        attention_modules = self._attention_modules()
        if self.use_cache:
            for attention in attention_modules:
                attention.begin_prefill_capture()
        model_input_ids, inputs_embeds = _prepare_landmark_inputs(
            self.model,
            input_ids,
            landmark_positions,
        )
        outputs = self.model(
            input_ids=model_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        if not self.use_cache:
            return PrefillOutput(logits=outputs.logits, cache=None)
        layer_caches = tuple(
            attention.end_prefill_capture() for attention in attention_modules
        )
        cache = RouteModelCache(
            layers=layer_caches,
            position_ids=position_ids.detach().clone(),
            physical_length=input_ids.shape[1],
        )
        return PrefillOutput(logits=outputs.logits, cache=cache)

    @torch.inference_mode()
    def cached_forward(
        self,
        input_ids: torch.Tensor,
        block: FastDLLMBlockLayout,
        cache: RouteModelCache,
        landmark_positions: torch.Tensor | None = None,
    ) -> CachedForwardOutput:
        core = self._core()
        if input_ids.shape != cache.position_ids.shape:
            raise ValueError("input_ids must match the cached physical sequence")
        if len(core.model.layers) != len(cache.layers):
            raise ValueError("cache layer count does not match the Dream model")

        home = input_ids.device
        query_home = block.query_positions.to(device=home, dtype=torch.long)
        hidden_states = _embed_partial_inputs(
            self.model,
            core,
            input_ids,
            query_home,
            landmark_positions,
        )
        pos_home = query_home.to(device=cache.position_ids.device)
        partial_position_ids = cache.position_ids.index_select(1, pos_home)
        position_embeddings = core.model.rotary_emb(
            hidden_states,
            partial_position_ids.to(device=hidden_states.device),
        )

        def _on(value, device):
            if torch.is_tensor(value):
                return value.to(device=device, non_blocking=True)
            if isinstance(value, (tuple, list)):
                return type(value)(_on(item, device) for item in value)
            return value

        routing_calls = 0
        for layer_idx, layer in enumerate(core.model.layers):
            layer_dev = next(layer.parameters()).device
            hidden_states = hidden_states.to(layer_dev, non_blocking=True)
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            qpos = query_home.to(layer_dev)
            upd = block.kv_update_positions.to(device=layer_dev, dtype=torch.long)
            chunks = block.affected_chunks.to(device=layer_dev, dtype=torch.long)
            layer_affected_chunks = (
                chunks
                if isinstance(layer.self_attn, KernelDreamFullRouteAttention)
                else chunks.new_empty(0)
            )
            attention_output, layer_stats = cached_attention_forward(
                layer.self_attn,
                normalized,
                position_ids=_on(partial_position_ids, layer_dev),
                position_embeddings=_on(position_embeddings, layer_dev),
                query_positions=qpos,
                kv_update_positions=upd,
                affected_chunks=layer_affected_chunks,
                cache=cache.layers[layer_idx],
            )
            hidden_states = residual + attention_output.to(layer_dev)
            residual = hidden_states
            hidden_states = residual + layer.mlp(
                layer.post_attention_layernorm(hidden_states)
            )
            routing_calls += layer_stats.routing_calls

        norm_dev = next(core.model.norm.parameters()).device
        hidden_states = core.model.norm(hidden_states.to(norm_dev, non_blocking=True))
        head_dev = next(core.lm_head.parameters()).device
        logits = core.lm_head(hidden_states.to(head_dev, non_blocking=True))
        return CachedForwardOutput(
            logits=logits.to(home, non_blocking=True),
            query_positions=query_home,
            routing_calls=routing_calls,
        )

    @torch.inference_mode()
    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        blocks: Sequence[FastDLLMBlockLayout],
        landmark_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, FastDLLMGenerationStats]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("generation currently requires input_ids [1,L]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must match input_ids")
        if not blocks:
            raise ValueError("at least one logical block is required")

        generated = input_ids.clone()
        full_prefills = 0
        cached_forwards = 0
        routing_calls = 0
        recomputed_tokens = 0
        answer_token_count = sum(block.answer_positions.numel() for block in blocks)

        for block in blocks:
            answer_positions = block.answer_positions.to(generated.device)
            predictor_positions = block.predictor_positions.to(generated.device)
            if not torch.all(generated[0, answer_positions] == self.mask_token_id):
                raise ValueError("each logical block must begin fully masked")

            prefill = self.prefill(
                generated,
                attention_mask,
                position_ids,
                landmark_positions=landmark_positions,
            )
            full_prefills += 1
            recomputed_tokens += generated.shape[1]
            # Keep the historical bootstrap available for paired regressions.
            initial_positions = predictor_positions[:1] if self.bootstrap == "first_token" else predictor_positions
            first_logits = prefill.logits.to(generated.device).index_select(
                1, initial_positions
            )
            block_cache = prefill.cache
            first_rows, first_token, _ = select_confidence_transfers(
                first_logits,
                torch.arange(initial_positions.numel(), device=generated.device),
                threshold=self.threshold,
                mask_token_id=self.mask_token_id,
            )
            generated[0, answer_positions[first_rows]] = first_token
            del first_logits, prefill

            while torch.any(generated[0, answer_positions] == self.mask_token_id):
                if self.use_cache:
                    cached = self.cached_forward(
                        generated, block, block_cache,
                        landmark_positions=landmark_positions,
                    )
                    cached_forwards += 1
                    routing_calls += cached.routing_calls
                    recomputed_tokens += cached.query_positions.numel()
                else:
                    exact = self.prefill(
                        generated, attention_mask, position_ids,
                        landmark_positions=landmark_positions,
                    )
                    cached = CachedForwardOutput(
                        logits=exact.logits.to(generated.device).index_select(
                            1, block.query_positions.to(generated.device)
                        ),
                        query_positions=block.query_positions.to(generated.device),
                        routing_calls=0,
                    )
                    del exact
                    full_prefills += 1
                    recomputed_tokens += generated.shape[1]

                predictor_rows = _rows_for_positions(
                    cached.query_positions,
                    predictor_positions,
                )
                candidate_logits = cached.logits.index_select(1, predictor_rows)
                active_rows = torch.where(
                    generated[0, answer_positions] == self.mask_token_id
                )[0]
                selected_rows, token_ids, _ = select_confidence_transfers(
                    candidate_logits,
                    active_rows,
                    threshold=self.threshold,
                    mask_token_id=self.mask_token_id,
                )
                generated[0, answer_positions[selected_rows]] = token_ids

                if full_prefills + cached_forwards > answer_token_count:
                    raise RuntimeError("confidence decoding exceeded its forward bound")

        all_answer_positions = torch.cat(
            [block.answer_positions.to(generated.device) for block in blocks]
        )
        if torch.any(generated[0, all_answer_positions] == self.mask_token_id):
            raise RuntimeError("generation completed with masked answer tokens")
        peak_memory = (
            int(torch.cuda.max_memory_allocated(generated.device))
            if generated.device.type == "cuda"
            else 0
        )
        return generated, FastDLLMGenerationStats(
            full_prefills=full_prefills,
            cached_forwards=cached_forwards,
            routing_calls=routing_calls,
            recomputed_tokens=recomputed_tokens,
            peak_memory_bytes=peak_memory,
        )
