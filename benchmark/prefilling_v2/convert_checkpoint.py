"""Build an SGLang model directory from Dream and a Prefilling-dLLM v2 training checkpoint.

The training checkpoint (`trainable_state.pt`) names tensors by their path in
the PEFT-wrapped training model. This writes them under short names, in the
float32 the checkpoint stores, next to a JSON file with the attention layout.
The base Dream weights and tokenizer are linked, not copied, and `config.json`
names the Prefilling-dLLM v2 model class.

    python convert_checkpoint.py --checkpoint step-1000 \
        --training-config v2-s1000.json \
        --base-model Dream-v0-Base-7B --output-dir Dream-v0-Base-7B-PrefillingV2
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file

_LORA = re.compile(
    r"^base_model\.model\.model\.layers\.(\d+)\.self_attn\."
    r"([qkvo]_proj)\.lora_([AB])\.default\.weight$"
)
_ROUTE = re.compile(
    r"^base_model\.model\.model\.layers\.(\d+)\.self_attn\."
    r"(qcal\.0|qcal\.1|qcal_norm)\.weight$"
)
_ROUTE_NAME = {"qcal.0": "down", "qcal.1": "up", "qcal_norm": "norm_weight"}


def convert_state(state):
    tensors, route_layers, lora_layers = {}, set(), set()
    for name, tensor in state.items():
        tensor = tensor.detach().to(torch.float32).contiguous()
        if name == "dream_route_lmk_type_embed":
            tensors["prefilling_v2.role_offset"] = tensor
            continue
        match = _LORA.match(name)
        if match:
            layer, proj, side = match.groups()
            lora_layers.add(int(layer))
            tensors[f"prefilling_v2.layers.{layer}.{proj}.lora_{side}"] = tensor
            continue
        match = _ROUTE.match(name)
        if match:
            layer, part = match.groups()
            route_layers.add(int(layer))
            tensors[f"prefilling_v2.layers.{layer}.route_query.{_ROUTE_NAME[part]}"] = (
                tensor
            )
            continue
        raise ValueError(f"unrecognized checkpoint tensor {name!r}")
    if "prefilling_v2.role_offset" not in tensors:
        raise ValueError("checkpoint has no routing role offset")
    return tensors, sorted(route_layers), sorted(lora_layers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    state_path = checkpoint / "trainable_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    tensors, route_layers, lora_layers = convert_state(state)
    train = json.loads(Path(args.training_config).read_text())

    interleave = int(train["route_interleave"])
    expected = [
        i for i in range(max(lora_layers) + 1) if i % interleave == interleave - 1
    ]
    if route_layers != expected:
        raise ValueError(
            f"route layers {route_layers} do not match interleave {interleave}"
        )
    rank = int(train["lora_r"])
    if any(t.shape[0] != rank for n, t in tensors.items() if n.endswith("lora_A")):
        raise ValueError("LoRA rank in the checkpoint differs from the training config")

    config = {
        "num_layers": max(lora_layers) + 1,
        "route_layers": route_layers,
        "chunk_size": int(train["chunk_size"]),
        "route_local_radius": int(train["local_window"]),
        "route_top_k": int(train["route_topk"]),
        "sliding_radius": int(train["swa_local_window"]),
        "lora_rank": rank,
        "lora_scaling": float(train["lora_alpha"]) / rank,
        "lora_targets": train["lora_target_modules"],
        "route_query_rank": int(train["route_qcal_rank"]),
        "entropy_prior_scale": 1.0,
        "rope_theta": float(train["model_rope_theta"]),
        "rope_scaling": train["model_rope_scaling"],
        "source_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.base_model).resolve()
    for source in sorted(base.iterdir()):
        target = out / source.name
        if source.name == "config.json" or source.is_dir():
            continue
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(source)
    model_config = json.loads((base / "config.json").read_text())
    model_config["architectures"] = ["PrefillingDreamV2Model"]
    (out / "config.json").write_text(json.dumps(model_config, indent=2) + "\n")
    save_file(tensors, str(out / "prefilling_v2_adapter.safetensors"))
    (out / "prefilling_v2_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"tensors": len(tensors), **config}, indent=2))


if __name__ == "__main__":
    main()
