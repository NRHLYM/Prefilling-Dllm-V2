# Prefilling-dLLM v2 on SGLang

Serves Dream-v0-Base-7B continued-trained with Prefilling-dLLM v2. The training code is on the `training` branch of this repository.

## Model directory

```bash
python benchmark/prefilling_v2/convert_checkpoint.py \
  --checkpoint <run>/step-1000 --training-config <run config>.json \
  --base-model Dream-v0-Base-7B --output-dir Dream-v0-Base-7B-PrefillingV2
```

The directory links the Dream weights and adds `prefilling_v2_adapter.safetensors`
(LoRA, routing-query residual, role offset; kept in float32 and applied
unmerged) and `prefilling_v2_config.json` (route layers, chunk size, window radii,
top-k, YaRN parameters).

## Server

```bash
python -m sglang.launch_server --model-path Dream-v0-Base-7B-PrefillingV2 \
  --trust-remote-code --dtype bfloat16 --dllm-algorithm PrefillingDreamV2 \
  --disable-radix-cache
```

`--dllm-algorithm-config` takes a YAML file with `block_size` (default 32),
`threshold` (default 0.9) and `rope_factor`. Requests are ordinary `/generate`
calls; the server inserts the routing slots itself.

The YaRN factor follows the request length: the trained factor up to the
trained length (8 up to 16K), then doubling until the scaled context covers
the request (16 up to 32K, 32 up to 64K, 64 up to 128K). Set `rope_factor` to
keep one factor for every length.

A request keeps its keys and values in the KV pool, one slot per physical
position (64 slots for every 63 tokens), so `--max-total-tokens` bounds how
many requests are admitted together and must cover the longest one. The
forward of a 128K request needs about 10 GB of working memory on top of the
weights and the pool.

Requirements beyond SGLang's: `tilelang` for the route-layer kernels, and the
FlashAttention 4 build SGLang ships (`flash_attn.cute`) for the windowed
attention. BF16 only.

## RULER probes

The probes are synthesized by the training tree. With it on `PYTHONPATH`:

```bash
python benchmark/prefilling_v2/dump_ruler_prompts.py --model-path Dream-v0-Base-7B \
  --task hils_mkmq --length 16384 --output mk.jsonl
python benchmark/prefilling_v2/run_server_eval.py --base-url http://127.0.0.1:30000 \
  --prompts mk.jsonl --output mk.out.jsonl
```

`run_standalone.py` runs the same prompts through the Prefilling-dLLM v2 stack without a
server.

## Limits

- Requests that run together are denoised one after another: the route
  kernels take a batch of one, and one long request already saturates the
  GPU (two processes on one H20 each ran at half speed). Concurrency queues
  requests; it does not raise throughput.
- A request is answered within one scheduling round, so a long request is
  not interleaved with others.
- The chunk summaries of the route layers are kept outside the KV pool
  (about 100 MB for a 128K request).
- Tensor parallel size 1, no prefill-decode disaggregation.
