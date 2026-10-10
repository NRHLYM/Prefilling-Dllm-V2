"""Send dumped prompts to an SGLang server running a Prefilling-dLLM v2 Dream model.

    python run_server_eval.py --base-url http://127.0.0.1:30000 \
        --prompts mk.jsonl --output mk.out.jsonl [--limit 20]
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--answer-tokens", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    output = Path(args.output)
    done = set()
    if output.exists():
        done = {json.loads(line)["index"] for line in output.read_text().splitlines()}
    records = []
    with open(args.prompts) as source:
        for line in source:
            record = json.loads(line)
            if args.limit and record["index"] >= args.limit:
                break
            if record["index"] not in done:
                records.append(record)

    def ask(record):
        started = time.perf_counter()
        reply = post(
            f"{args.base_url}/generate",
            {
                "input_ids": record["prompt_ids"],
                "sampling_params": {
                    "max_new_tokens": args.answer_tokens or record["answer_tokens"],
                    "temperature": 0,
                },
            },
            args.timeout,
        )
        return {
            "index": record["index"],
            "task": record["task"],
            "answer_ids": reply["output_ids"],
            "outputs": record["outputs"],
            "seconds": time.perf_counter() - started,
            "cached_forwards": -1,
        }

    started = time.perf_counter()
    with open(output, "a") as sink, ThreadPoolExecutor(args.concurrency) as pool:
        for result in pool.map(ask, records):
            sink.write(json.dumps(result) + "\n")
            sink.flush()
            shown = {k: v for k, v in result.items() if k != "outputs"}
            print(json.dumps(shown), flush=True)
    elapsed = time.perf_counter() - started
    if records:
        print(
            json.dumps(
                {
                    "requests": len(records),
                    "concurrency": args.concurrency,
                    "seconds": elapsed,
                    "requests_per_second": len(records) / elapsed,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
