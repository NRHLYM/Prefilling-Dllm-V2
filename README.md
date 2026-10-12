# Prefilling-dLLM v2: training code

Continued training of Dream-v0-Base-7B under a sparse attention budget. A mask
token at the end of each chunk summarizes that chunk, and on every fourth layer
(a *route layer*) a query keeps a bidirectional local window plus the top-K
chunks scored from those summaries. The other layers use a bidirectional
sliding window.

The SGLang serving code is on the `main` branch of this repository.

## Layout

| Path | Contents |
| --- | --- |
| `prefilling_v2/` | attention layers, the query residual, the training loop, Fast-dLLM decoding |
| `ops/` | TileLang kernels |
| `configs/` | the configs of the paper |
| `scripts/from_dense16k/` | RULER evaluation, the routing-mask probe, the latency benchmark |
| `scripts/prefilling_v2/` | corpus packing and LongBench evaluation |
| `unittests/` | unit tests |

Run everything from the repository root with `PYTHONPATH` set to it.

## Configs

| Config | Model |
| --- | --- |
| `dense-extension-s500.json` | Dream extended to 16K with YaRN factor 8, dense attention, 500 steps. Every other run starts from its checkpoint. |
| `v2-s1000.json` | Prefilling-dLLM v2 |
| `nsa-s1000.json`, `dsa-s1000.json`, `swa-s1000.json` | bidirectional NSA, DSA and sliding-window baselines |
| `hybrid-dense-s1000.json`, `full-dense-s1000.json` | dense references |
| `v2-ablate-no-residual-s1000.json`, `v2-ablate-no-role-offset-s1000.json`, `v2-ablate-mean-summary-s1000.json`, `v2-ablate-no-entropy-s1000.json` | component ablations |
| `v2-ablate-eos-token-s1000.json`, `v2-ablate-new-token-s1000.json` | routing-token ablations |

The paths in the configs are relative: `models/Dream-v0-Base-7B` for the base
model, `data/dolma3_longmino/dream-16k-onedoc/` for the packed corpus, and
`outputs/<name>` for checkpoints. Point them at your own locations.

The main recipe: route layers `{3,7,11,15,19,23,27}`, local radius 256 plus
32 chunks of 64 positions on a route layer, sliding-window radius 1280 on the
other 21 layers, LoRA rank 16 on the query, key, value and output projections,
a rank-64 query residual, 1000 steps at learning rate `1e-4` with a cosine
schedule and 50 warmup steps, four GPUs with one 16K sequence each. The loss
is the mean of a language-model term and a RULER term.

## Data

- Base model: `Dream-v0-Base-7B`.
- Language-model windows: one-document 16K windows from
  [Dolma 3 Longmino](https://huggingface.co/datasets/allenai/dolma3_longmino_mix-50B-1025),
  packed with `scripts/prefilling_v2/build_dream_packed_corpus.py`.
- RULER examples are synthesized during training and evaluation; there is no
  dataset to download.

## Training

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m prefilling_v2.train_fulltext --config configs/v2-s1000.json
```

## RULER evaluation

Three task families: `ruler_sn` (single needle), `ruler_mkmq` (multi-key
multi-query), `ruler_vt` (variable tracking). One task at one length:

```bash
python scripts/from_dense16k/eval_ruler.py --mode eval \
  --training_config configs/v2-s1000.json --checkpoint outputs/v2-s1000/step-1000 \
  --output_dir outputs/v2-s1000/ruler --data_dir data/ruler-probes \
  --task ruler_sn --max_seq_len 16384 \
  --rank 0 --world_size 1 --device cuda:0 --block_length 32 --threshold 0.9
python scripts/from_dense16k/eval_ruler.py --mode merge \
  --output_dir outputs/v2-s1000/ruler --data_dir data/ruler-probes \
  --task ruler_sn --max_seq_len 16384
```

Lengths are 16384, 32768, 65536 and 131072. By default the YaRN factor is the
length divided by 2048. Set `RULER_KEEP_TRAIN_YARN=1 RULER_YARN_FACTOR=8` to
keep the training factor. At 131072 a sequence needs two GPUs: make two
visible and add `--layer_parallel_gpus 2`.

The span that the decoder fills has the length of the gold answer, and the
routing slot holds the token the model was trained with.

## LongBench, probe, latency

- LongBench: `scripts/prefilling_v2/eval_longbench_fastdllm_route.py` for
  Prefilling-dLLM v2, and the `_nsa`, `_dsa`, `_swa`, `_dense_ckpt` scripts next
  to it for the other models.
- Routing-mask probe: `scripts/from_dense16k/probe_lmk_summary.py`, then
  `plot_lmk_summary.py`.
- Latency: `scripts/from_dense16k/bench_fastdllm_latency.py`.

## Status

This tree is the latest version of the code, with internal names replaced for
release. The renaming was checked statically; the experiments were not run
again on this tree.

## Acknowledgment

The route layer was inspired by hierarchical sparse attention
([HiLS-Attention](https://github.com/Tencent-Hunyuan/HiLS-Attention),
[arXiv:2607.02980](https://arxiv.org/abs/2607.02980)), and we thank its
authors. Two kernels under `ops/`, `hils_fwd_bwd_head.py` (attention over the
selected chunks) and `topk_head_softmax.py` (chunk selection), are adapted from
the official HiLS-Attention implementation; the author notices in those files
are kept.
