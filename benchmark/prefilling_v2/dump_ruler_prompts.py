"""Write the RULER probe prompts of the Prefilling-dLLM v2 training tree as token ids.

The probes are synthesized from the example index, so there is no dataset to
download. Run with the training tree (the `training` branch of this repository)
on PYTHONPATH:

    PYTHONPATH=<training tree> python dump_ruler_prompts.py \
        --model-path Dream-v0-Base-7B --task mkmq --length 16384 --output mk.jsonl
"""

import argparse
import json

from prefilling_v2.data import RulerDenoisingSynthesizer
from transformers import AutoTokenizer

from scripts.from_dense16k.eval_ruler import (
    TASK_SYNTH_ID,
    pack_route_style_prompt,
    text_slot_budget,
)

# The training tree prefixes its task names; the command line takes the short ones.
TASKS = {name.split("_", 1)[1]: name for name in TASK_SYNTH_ID}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    synthesizer = RulerDenoisingSynthesizer(
        tokenizer, task_ids=(TASK_SYNTH_ID[TASKS[args.task]],)
    )
    slots = text_slot_budget(args.length, is_route=True, chunk_size=args.chunk_size)
    with open(args.output, "w") as handle:
        for index in range(args.num_samples):
            prompt_ids, answer_tokens, outputs = pack_route_style_prompt(
                tokenizer=tokenizer,
                synthesizer=synthesizer,
                task=TASKS[args.task],
                text_slots=slots,
                sample_index=index,
                example={},
                block_length=32,
            )
            record = {
                "index": index,
                "task": args.task,
                "length": args.length,
                "prompt_ids": prompt_ids,
                "answer_tokens": answer_tokens,
                "outputs": outputs,
            }
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
