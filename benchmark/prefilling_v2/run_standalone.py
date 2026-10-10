"""Run the Prefilling-dLLM v2 stack on dumped prompts without a server.

Loads the base weights straight from the model directory, so it checks the
stack and the decoder on their own:

    python run_standalone.py --model-path Dream-v0-Base-7B-PrefillingV2 \
        --prompts mk.jsonl --output mk.out.jsonl [--rope-factor 8]
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.dllm.prefilling_v2.decoder import generate
from sglang.srt.dllm.prefilling_v2.stack import PrefillingV2Stack
from sglang.srt.dllm.prefilling_v2.weights import PrefillingV2Config, load_from_hf_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--rope-factor", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    hf_config = SimpleNamespace(
        **json.loads((Path(args.model_path) / "config.json").read_text())
    )
    config = PrefillingV2Config.from_files(args.model_path, hf_config)
    device = torch.device("cuda")
    stack = PrefillingV2Stack(config, load_from_hf_dir(args.model_path, config, device))

    done = set()
    output = Path(args.output)
    if output.exists():
        done = {json.loads(line)["index"] for line in output.read_text().splitlines()}
    with open(args.prompts) as source, open(output, "a") as sink:
        for line in source:
            record = json.loads(line)
            if record["index"] in done:
                continue
            if args.limit and record["index"] >= args.limit:
                break
            prompt = record["prompt_ids"]
            token_ids = torch.tensor(
                prompt + [config.mask_token_id] * record["answer_tokens"],
                dtype=torch.long,
            )
            started = time.perf_counter()
            answer, stats = generate(
                stack,
                token_ids,
                answer_start=len(prompt),
                block_size=args.block_size,
                threshold=args.threshold,
                rope_factor=args.rope_factor,
            )
            torch.cuda.synchronize()
            result = {
                "index": record["index"],
                "task": record["task"],
                "answer_ids": answer.tolist(),
                "outputs": record["outputs"],
                "seconds": time.perf_counter() - started,
                "full_forwards": stats.full_forwards,
                "cached_forwards": stats.cached_forwards,
                "rope_factor": stats.rope_factor,
                "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
            }
            sink.write(json.dumps(result) + "\n")
            sink.flush()
            print(
                json.dumps({k: v for k, v in result.items() if k != "outputs"}),
                flush=True,
            )


if __name__ == "__main__":
    main()
