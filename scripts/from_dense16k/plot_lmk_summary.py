#!/usr/bin/env python3
"""Bar chart + chunk-vs-routing-token heatmaps for the summary probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("mask", "vocab", "eos")
ARM_LABEL = {"mask": "MASK", "vocab": "new token", "eos": "EOS"}
CONDS = ("clean", "shuffle_lmk", "shuffle_interior", "surgery")
COND_LABEL = {
    "clean": "clean",
    "shuffle_lmk": "shuffle routing token",
    "shuffle_interior": "shuffle tokens",
    "surgery": "move needle",
}


def _metrics(root: Path, arm: str, cond: str) -> float | None:
    p = root / arm / cond / "metrics.json"
    if not p.is_file():
        return None
    return float(json.loads(p.read_text())["score"])


def _heat(root: Path, arm: str, cond: str):
    p = root / arm / cond / "heat_mean.npy"
    if not p.is_file():
        return None
    return np.load(p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="outputs/lmk_summary_probe_sn16k",
    )
    parser.add_argument("--out_dir", default="")
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out_dir) if args.out_dir else root
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(CONDS))
    width = 0.24
    for i, arm in enumerate(ARMS):
        ys = [_metrics(root, arm, c) for c in CONDS]
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            [0 if y is None else y for y in ys],
            width,
            label=ARM_LABEL[arm],
            color=["#1f77b4", "#ff7f0e", "#2ca02c"][i],
        )
        for bar, y in zip(bars, ys):
            if y is None:
                bar.set_alpha(0.15)
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    f"{y:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    ax.set_xticks(x, [COND_LABEL[c] for c in CONDS])
    ax.set_ylabel("S-N 16k")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False)
    ax.set_title("Routing token as summary (higher drop after routing-token shuffle is better)")
    fig.tight_layout()
    fig.savefig(out / "lmk_summary_bars.pdf")
    fig.savefig(out / "lmk_summary_bars.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.2))
    vmins, vmaxs = [], []
    mats = {}
    for arm in ARMS:
        for cond in ("clean", "shuffle_lmk"):
            mat = _heat(root, arm, cond)
            mats[(arm, cond)] = mat
            if mat is not None:
                vmins.append(float(np.percentile(mat, 5)))
                vmaxs.append(float(np.percentile(mat, 95)))
    vmin = min(vmins) if vmins else 0.0
    vmax = max(vmaxs) if vmaxs else 1.0
    last = None
    for row, cond in enumerate(("clean", "shuffle_lmk")):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            mat = mats.get((arm, cond))
            if mat is None:
                ax.set_axis_off()
                ax.set_title(f"{ARM_LABEL[arm]} / {COND_LABEL[cond]}\nmissing")
                continue
            last = ax.imshow(
                mat, origin="upper", cmap="magma", vmin=vmin, vmax=vmax, aspect="auto"
            )
            ax.set_title(f"{ARM_LABEL[arm]} / {COND_LABEL[cond]}")
            ax.set_xlabel("routing token bin")
            ax.set_ylabel("chunk bin")
            ax.set_xticks([])
            ax.set_yticks([])
    if last is not None:
        fig.colorbar(last, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.suptitle(
        "Cosine(chunk content, routing token). Diagonal = each routing token summarizes its chunk."
    )
    fig.tight_layout()
    fig.savefig(out / "lmk_summary_heat.pdf")
    fig.savefig(out / "lmk_summary_heat.png", dpi=160)
    plt.close(fig)

    rows = []
    for arm in ARMS:
        rec = {"arm": arm}
        for cond in CONDS:
            rec[cond] = _metrics(root, arm, cond)
        rows.append(rec)
    (out / "summary.json").write_text(json.dumps({"scores": rows}, indent=2) + "\n")
    print(json.dumps({"out": str(out), "scores": rows}, indent=2))
    print("LMK_SUMMARY_PROBE_PLOTS_DONE")


if __name__ == "__main__":
    main()
