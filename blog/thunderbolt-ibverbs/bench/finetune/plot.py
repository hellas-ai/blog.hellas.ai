#!/usr/bin/env python3
"""Plots for the 2-node Strix-Halo FSDP fine-tuning benchmarks.

Reads every `*.csv` in this directory (produced by
`strix-halo-finetune-bench-<arch>` from the nix-strix-halo flake) and
emits two SVGs:

  - runtime_vs_transport.svg : train wall time per transport, bar plot.
  - bw_vs_transport.svg      : effective per-direction bandwidth per
                               transport, computed from delta counters
                               (br0.lan rx for ETH; usb4_rdma packet
                               counts × MTU for RDMA).

The CSV is sparse on purpose — every column from the per-row JSON gets
its own header but only transports that exercise a given counter
populate it. Columns we don't know about are ignored.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BENCH_DIR = Path(__file__).parent

# Display order + colours.
TRANSPORT_ORDER = ["local", "eth", "rdma", "rdma-4hca"]
TRANSPORT_LABEL = {
    "local": "single-node (no networking)",
    "eth": "TCP over br0.lan (Ethernet)",
    "rdma": "RDMA (single-HCA, new module)",
    "rdma-4hca": "RDMA (4-HCA, new module per-port)",
}
TRANSPORT_COLOR = {
    "local": "#7f7f7f",
    "eth": "#2ca02c",
    "rdma": "#1f77b4",
    "rdma-4hca": "#d62728",
}

# Per-frame size assumed for the usb4_rdma packet counters. The TX ring
# is configured for 4 KiB frames in the current driver; if that ever
# changes we'd want to read it from `path_cfg` in the snapshot.
RDMA_FRAME_BYTES = 4096


def load_rows() -> list[dict]:
    out: list[dict] = []
    for csv_path in sorted(BENCH_DIR.glob("*.csv")):
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                out.append(r)
    return out


def to_float(s) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def to_int(s) -> int | None:
    if s is None or s == "":
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


# Loss > LOSS_OK_THRESHOLD on full FT of SmolLM2-135M (1 epoch, 400
# steps) signals a likely numerical-correctness regression on the
# transport. Used to flag bars in the loss + runtime plots.
LOSS_OK_THRESHOLD = 1.5


def _barh(ax, labels, values, colors, value_fmt, hatches=None):
    """Shared bar-chart helper with consistent styling + value labels."""
    bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    if hatches:
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
    vmax = max(values) if values else 0
    for bar, val in zip(bars, values):
        ax.text(
            val + vmax * 0.012,
            bar.get_y() + bar.get_height() / 2,
            value_fmt(val),
            va="center",
            fontsize=9,
        )
    # Pad so value labels never clip.
    ax.set_xlim(right=vmax * 1.18)
    ax.invert_yaxis()


def runtime_plot(rows: list[dict]) -> None:
    """Bar chart: training wall time per transport, headline view.

    Bars whose train_loss didn't converge to the same neighbourhood as
    the others get hatched to signal an apples-to-oranges comparison.
    """
    grouped_rt: dict[str, list[float]] = defaultdict(list)
    grouped_loss: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        rt = to_float(r.get("training.train_runtime_s"))
        loss = to_float(r.get("training.train_loss"))
        t = r.get("row.transport")
        if rt is None or t is None:
            continue
        grouped_rt[t].append(rt)
        if loss is not None:
            grouped_loss[t].append(loss)

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    transports = [t for t in TRANSPORT_ORDER if t in grouped_rt]
    times = [sum(grouped_rt[t]) / len(grouped_rt[t]) for t in transports]
    colors = [TRANSPORT_COLOR.get(t, "#7f7f7f") for t in transports]
    hatches = [
        "///"
        if grouped_loss.get(t)
        and (sum(grouped_loss[t]) / len(grouped_loss[t])) > LOSS_OK_THRESHOLD
        else ""
        for t in transports
    ]
    _barh(
        ax,
        [TRANSPORT_LABEL.get(t, t) for t in transports],
        times,
        colors,
        lambda v: f"{v:.1f} s",
        hatches=hatches,
    )
    ax.set_xlabel("Wall clock (s, 1 epoch / 400 steps, lower = better)")
    title = "SmolLM2-135M full FSDP — train wall time on 2× Strix Halo"
    if any(hatches):
        title += "\n(hatched = train_loss did not match the other runs)"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(BENCH_DIR / "runtime_vs_transport.svg")
    plt.close(fig)


def bw_plot(rows: list[dict]) -> None:
    """Bar chart: effective sustained per-direction bandwidth per transport."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        rt = to_float(r.get("training.train_runtime_s"))
        if rt is None:
            continue
        t = r.get("row.transport")
        if t is None:
            continue
        bytes_total = 0
        if t == "eth":
            rx = to_int(r.get("delta.master.eth.br0.lan.rx_bytes")) or 0
            tx = to_int(r.get("delta.master.eth.br0.lan.tx_bytes")) or 0
            bytes_total = max(rx, tx)
        else:
            # Sum all per-rail tx_completed packets we recognise.
            for k, v in r.items():
                if not k.startswith("delta.master.dfs.") or "data_tx_completed" not in k:
                    continue
                n = to_int(v)
                if n:
                    bytes_total += n * RDMA_FRAME_BYTES
        if bytes_total == 0:
            continue
        gbps = (bytes_total * 8) / (rt * 1e9)
        grouped[t].append(gbps)

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    transports = [t for t in TRANSPORT_ORDER if t in grouped]
    vals = [sum(grouped[t]) / len(grouped[t]) for t in transports]
    colors = [TRANSPORT_COLOR.get(t, "#7f7f7f") for t in transports]
    _barh(
        ax,
        [TRANSPORT_LABEL.get(t, t) for t in transports],
        vals,
        colors,
        lambda v: f"{v:.1f} Gb/s",
    )
    ax.set_xlabel("Effective sustained per-direction throughput (Gb/s, app payload)")
    ax.set_title(
        "SmolLM2-135M full FSDP — interconnect throughput on 2× Strix Halo",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(BENCH_DIR / "bw_vs_transport.svg")
    plt.close(fig)


def loss_plot(rows: list[dict]) -> None:
    """Bar chart: final train_loss + eval_loss per transport.

    Same model, same dataset, same step count → these should sit on
    the same horizontal line. A bar that towers above the others means
    the gradient transport silently corrupted something.
    """
    grouped_tl: dict[str, list[float]] = defaultdict(list)
    grouped_el: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        t = r.get("row.transport")
        if t is None:
            continue
        tl = to_float(r.get("training.train_loss"))
        el = to_float(r.get("training.eval_loss"))
        if tl is not None:
            grouped_tl[t].append(tl)
        if el is not None:
            grouped_el[t].append(el)

    transports = [t for t in TRANSPORT_ORDER if t in grouped_tl or t in grouped_el]
    if not transports:
        return
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    n = len(transports)
    y = list(range(n))
    height = 0.38
    tl_vals = [
        sum(grouped_tl[t]) / len(grouped_tl[t]) if grouped_tl.get(t) else 0
        for t in transports
    ]
    el_vals = [
        sum(grouped_el[t]) / len(grouped_el[t]) if grouped_el.get(t) else 0
        for t in transports
    ]
    colors = [TRANSPORT_COLOR.get(t, "#7f7f7f") for t in transports]
    bars_tl = ax.barh(
        [yi - height / 2 for yi in y],
        tl_vals,
        height=height,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        label="train_loss",
    )
    bars_el = ax.barh(
        [yi + height / 2 for yi in y],
        el_vals,
        height=height,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.55,
        label="eval_loss",
    )
    for bar, val in list(zip(bars_tl, tl_vals)) + list(zip(bars_el, el_vals)):
        if val > 0:
            ax.text(
                val + 0.04,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=8,
            )
    ax.axvline(
        LOSS_OK_THRESHOLD,
        linestyle=":",
        color="#444",
        linewidth=1,
        label=f"loss-anomaly threshold ({LOSS_OK_THRESHOLD})",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([TRANSPORT_LABEL.get(t, t) for t in transports])
    ax.invert_yaxis()
    ax.set_xlabel("Loss (lower = better; same dataset / step count across rows)")
    ax.set_title(
        "SmolLM2-135M full FSDP — train_loss / eval_loss by transport",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(BENCH_DIR / "loss_vs_transport.svg")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    if not rows:
        print("no CSV rows found")
        return
    runtime_plot(rows)
    bw_plot(rows)
    loss_plot(rows)
    print(
        f"wrote runtime_vs_transport.svg, bw_vs_transport.svg, loss_vs_transport.svg ({len(rows)} rows)"
    )


if __name__ == "__main__":
    main()
