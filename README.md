# Prefilling-dLLM v2 on SGLang

This branch serves [Dream-v0-Base-7B](https://huggingface.co/Dream-org/Dream-v0-Base-7B)
continued-trained with Prefilling-dLLM v2 on top of
[SGLang](https://github.com/sgl-project/sglang). The training code, the
baselines and the RULER metrics are on the
[`training`](https://github.com/NRHLYM/Prefilling-Dllm-V2/tree/training)
branch of this repository. Everything outside the files listed below is SGLang
as of its commit `1c2abedb2`; see the
[upstream README](https://github.com/sgl-project/sglang#readme) and
[documentation](https://docs.sglang.io/) for SGLang itself.

[Prefilling-dLLM v1](https://github.com/menik1126/prefilling-dllm) is
training-free: it picks the chunks of a long prompt before generation and
caches them. v2 continued-trains the model so that its attention picks the
chunks itself, per query and per layer.

## What the method does

A diffusion language model attends over the whole context at every denoising
step. v2 continued-trains the model under a sparse budget:

- One routing slot, holding the model's own mask token, follows every 63 text
  tokens. It pools the keys of its chunk into a summary.
- On every fourth layer a query reads a bidirectional local window plus the
  32 chunks it scores highest from those summaries.
- The other layers read a wider bidirectional window only.

At inference a request is denoised in blocks of 32 tokens. A block starts with
one forward of the whole sequence, which keeps every layer's keys, values and
chunk summaries; the remaining steps recompute only the block.

## Status

| Piece | State |
| --- | --- |
| Attention stack (sliding and route layers, cached denoising) | done |
| Served through SGLang's dLLM interface (`PrefillingDreamV2`) | done |
| Contexts from 16K to 128K, YaRN factor following the request length | done |
| Keys and values kept in SGLang's KV pool | done |
| Answers longer than one block, early stop at the end token | done |
| Prefill-decode disaggregation on stock SGLang PD | done, as a feature; see Limits |
| Requests batched inside the route kernels | not done; see Limits |

## Running it

Convert a training checkpoint, then launch the server:

```bash
python benchmark/prefilling_v2/convert_checkpoint.py \
  --checkpoint <run>/step-1000 --training-config <run config>.json \
  --base-model Dream-v0-Base-7B --output-dir Dream-v0-Base-7B-PrefillingV2

python -m sglang.launch_server --model-path Dream-v0-Base-7B-PrefillingV2 \
  --trust-remote-code --dtype bfloat16 --dllm-algorithm PrefillingDreamV2 \
  --disable-radix-cache

curl -s http://127.0.0.1:30000/generate -H 'Content-Type: application/json' \
  -d '{"text": "<document and question>", "sampling_params": {"max_new_tokens": 32}}'
```

Requests are ordinary `/generate` calls: the server inserts the routing slots
itself. It needs `tilelang` for the route-layer kernels on top of SGLang's own
requirements, and runs in BF16 only.

[`benchmark/prefilling_v2/README.md`](benchmark/prefilling_v2/README.md) has
the algorithm config keys, the PD launch commands, memory sizing, and the
RULER probe tooling.

## Where the code is

| Path | Contents |
| --- | --- |
| `python/sglang/srt/dllm/prefilling_v2/` | layout, weights, the attention stack, the block decoder, the PD handoff, and the route kernels copied from the `training` branch |
| `python/sglang/srt/models/prefilling_dream_v2.py` | model class |
| `python/sglang/srt/dllm/algorithm/prefilling_dream_v2.py` | dLLM algorithm |
| `python/sglang/srt/dllm/config.py`, `dllm/mixin/`, `disaggregation/`, `arg_groups/dllm_hook.py` | the edits to SGLang itself |
| `benchmark/prefilling_v2/` | checkpoint conversion and probe clients |

## Limits

- Requests that run together are denoised one after another. The route
  kernels take a batch of one, and one long request already saturates the
  GPU, so concurrency queues requests without raising throughput.
- A request is answered within one scheduling round; a long request is not
  interleaved with others.
- Prefill-decode disaggregation works but does not speed anything up: nearly
  all of a request's time is the full forward, which stays on the prefill
  server. Under PD a request must fit the context length at 71 slots per 63
  tokens, about 116K tokens.
- Tensor parallel size 1.

## Acknowledgment

Built on [SGLang](https://github.com/sgl-project/sglang) and its diffusion LLM
support for Dream.

The route layer was inspired by hierarchical sparse attention
([HiLS-Attention](https://github.com/Tencent-Hunyuan/HiLS-Attention),
[arXiv:2607.02980](https://arxiv.org/abs/2607.02980)), and we thank its
authors. Two kernels under `python/sglang/srt/dllm/prefilling_v2/ops/`,
`hils_fwd_bwd_head.py` (attention over the selected chunks) and
`topk_head_softmax.py` (chunk selection), are adapted from the official
HiLS-Attention implementation; the author notices in those files are kept.
