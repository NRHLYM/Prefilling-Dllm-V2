"""Send dumped prompts to an SGLang server running a Prefilling-dLLM v2 Dream model.

    python run_server_eval.py --base-url http://127.0.0.1:30000 \
        --prompts mk.jsonl --output mk.out.jsonl [--limit 20]
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    output = Path(args.output)
    done = set()
    if output.exists():
        done = {json.loads(line)["index"] for line in output.read_text().splitlines()}
    with open(args.prompts) as source, open(output, "a") as sink:
        for line in source:
            record = json.loads(line)
            if args.limit and record["index"] >= args.limit:
                break
            if record["index"] in done:
                continue
            started = time.perf_counter()
            reply = post(
                f"{args.base_url}/generate",
                {
                    "input_ids": record["prompt_ids"],
                    "sampling_params": {
                        "max_new_tokens": record["answer_tokens"],
                        "temperature": 0,
                    },
                },
                args.timeout,
            )
            result = {
                "index": record["index"],
                "task": record["task"],
                "answer_ids": reply["output_ids"],
                "outputs": record["outputs"],
                "seconds": time.perf_counter() - started,
                "cached_forwards": -1,
            }
            sink.write(json.dumps(result) + "\n")
            sink.flush()
            print(
                json.dumps({k: v for k, v in result.items() if k != "outputs"}),
                flush=True,
            )


if __name__ == "__main__":
    main()
