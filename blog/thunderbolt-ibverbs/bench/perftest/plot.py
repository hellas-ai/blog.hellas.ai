#!/usr/bin/env python3
"""Plots for the strix-strix perftest sweep.

Reads the four 2026-05-27 lowlevel-sweep CSVs in this directory
(unified schema with kind, verb, direction, bidirectional, qps,
size_bytes, bw_avg_gbps, lat_avg_us). Writes three SVGs:

  - bw_vs_size.svg     bandwidth vs size, faceted by verb (read | write
                       | send), lines per (transport, qps),
                       unidirectional (bidi=0), forward direction,
                       per-rail for native (single-link comparison
                       with the single-device RXE transports).
  - lat_vs_size.svg    one-way latency vs size, faceted by verb,
                       line per transport at qps=1, sweep extends to
                       1 MiB.
  - write_bw_vs_size.svg  single-panel landing-page headline: just
                          ib_write_bw.
"""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

BENCH_DIR = Path(__file__).parent

# CSV-name substring → transport key
CSV_TO_TRANSPORT = [
    ("native-4rail",   "native"),
    ("rxe-ethernet",   "rxe_ethernet"),
    ("rxe-tbnet0",     "rxe_tbnet"),
    # rxe-tbnet1 is a sanity copy of the second cable, skipped from the
    # chart to keep the legend tight.
]

TRANSPORT_COLOR = {
    "native":       "#1f77b4",
    "rxe_tbnet":    "#ff7f0e",
    "rxe_ethernet": "#2ca02c",
}
TRANSPORT_LABEL = {
    "native":       "native usb4_rdma (per rail)",
    "rxe_tbnet":    "RXE over TBnet",
    "rxe_ethernet": "RXE over 2.5G LAN",
}
QPS_LINESTYLE = {1: ":", 2: "-.", 4: "--", 8: "-"}
VERB_ORDER = ["read", "write", "send"]


def human_size(n_bytes, _pos=None):
    n = int(n_bytes)
    if n >= 1 << 20:
        return f"{n // (1 << 20)} MiB"
    if n >= 1 << 10:
        return f"{n // (1 << 10)} KiB"
    return f"{n} B"


def transport_for(csv_path: Path) -> str | None:
    for sub, t in CSV_TO_TRANSPORT:
        if sub in csv_path.name:
            return t
    return None


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for csv_path in sorted(BENCH_DIR.glob("*.csv")):
        transport = transport_for(csv_path)
        if transport is None:
            continue
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                if r.get("status") != "ok":
                    continue
                kind = r.get("kind")
                verb = r.get("verb")
                if kind not in ("bw", "lat") or verb not in VERB_ORDER:
                    continue
                try:
                    size = int(r["size_bytes"])
                    qps = int(r["qps"])
                except (KeyError, ValueError):
                    continue
                if kind == "bw":
                    value = to_float(r.get("bw_avg_gbps"))
                else:
                    value = to_float(r.get("lat_avg_us"))
                if value is None:
                    continue
                rows.append({
                    "transport": transport,
                    "verb":      verb,
                    "kind":      kind,
                    "size":      size,
                    "qps":       qps,
                    "direction": r.get("direction", "forward"),
                    "bidir":     r.get("bidirectional", "0") == "1",
                    "value":     value,
                })
    return rows


def aggregate(rows, kind, want_bidir=False, direction="forward"):
    by_key = defaultdict(list)
    for r in rows:
        if r["kind"] != kind or r["bidir"] != want_bidir:
            continue
        if r["direction"] != direction:
            continue
        by_key[(r["verb"], r["transport"], r["size"], r["qps"])].append(r["value"])
    return {k: mean(v) for k, v in by_key.items()}


def style_size_axis(ax, sizes):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(human_size))
    ax.set_xticks(sorted(set(sizes)))
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", labelsize=8)


def transport_legend_handles():
    return [
        Line2D([0], [0], color=c, marker="o", markersize=4, linewidth=2,
               label=TRANSPORT_LABEL[t])
        for t, c in TRANSPORT_COLOR.items()
    ]


def qps_legend_handles():
    return [
        Line2D([0], [0], color="black", linestyle=ls, linewidth=1.5,
               label=f"{q} QP{'s' if q > 1 else ''}")
        for q, ls in QPS_LINESTYLE.items()
    ]


def plot_bw_vs_size(rows):
    agg = aggregate(rows, kind="bw")
    all_sizes = {sz for (_v, _t, sz, _q) in agg.keys()}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, verb in zip(axes, VERB_ORDER):
        groups = defaultdict(list)
        for (v, t, sz, qps), val in agg.items():
            if v != verb:
                continue
            groups[(t, qps)].append((sz, val))
        for (t, qps), pts in sorted(groups.items()):
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys,
                    color=TRANSPORT_COLOR.get(t, "gray"),
                    linestyle=QPS_LINESTYLE.get(qps, "-"),
                    marker="o", markersize=4, linewidth=1.5)
        style_size_axis(ax, all_sizes)
        ax.set_title(f"ib_{verb}_bw", fontsize=11)
        ax.set_xlabel("Message size")
        ax.axhline(10, color="gray", linestyle=":", alpha=0.6, linewidth=1)
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("Average bandwidth (Gb/s, per rail)")
    axes[0].set_ylim(bottom=0, top=12)
    axes[-1].annotate(
        "10 Gb/s — per-cable USB4 ceiling",
        xy=(0.98, 10), xycoords=("axes fraction", "data"),
        xytext=(0, -3), textcoords="offset points",
        fontsize=8, color="gray", ha="right", va="top",
    )

    fig.suptitle(
        "Per-rail bandwidth by verb, strix-1 ↔ strix-2  "
        "(2026-05-27 IOMMU-off sweep, unidirectional)",
        fontsize=11,
    )
    fig.legend(handles=transport_legend_handles(), loc="upper left",
               bbox_to_anchor=(0.01, 0.94), title="Transport", frameon=True)
    fig.legend(handles=qps_legend_handles(), loc="lower right",
               bbox_to_anchor=(0.99, 0.08), title="QPs", frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = BENCH_DIR / "bw_vs_size.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_lat_vs_size(rows):
    agg = aggregate(rows, kind="lat")
    all_sizes = {sz for (_v, _t, sz, q) in agg.keys() if q == 1}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, verb in zip(axes, VERB_ORDER):
        groups = defaultdict(list)
        for (v, t, sz, qps), val in agg.items():
            if v != verb or qps != 1:
                continue
            groups[t].append((sz, val))
        for t, pts in sorted(groups.items()):
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys,
                    color=TRANSPORT_COLOR.get(t, "gray"),
                    marker="o", markersize=4, linewidth=1.6)
        style_size_axis(ax, all_sizes)
        ax.set_title(f"ib_{verb}_lat", fontsize=11)
        ax.set_xlabel("Message size")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.axvline(1 << 14, color="gray", linestyle=":", alpha=0.5, linewidth=1)

    axes[0].set_ylabel("Average one-way latency (µs, log scale)")
    axes[-1].annotate(
        "16 KiB — knee where size starts to dominate",
        xy=(1 << 14, 0.97), xycoords=("data", "axes fraction"),
        xytext=(4, -3), textcoords="offset points",
        fontsize=8, color="gray", ha="left", va="top",
    )

    fig.suptitle(
        "One-way latency by verb, strix-1 ↔ strix-2  "
        "(2026-05-27 IOMMU-off sweep, 1 QP, 64 B → 1 MiB)",
        fontsize=11,
    )
    fig.legend(handles=transport_legend_handles(), loc="upper left",
               bbox_to_anchor=(0.01, 0.94), title="Transport", frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = BENCH_DIR / "lat_vs_size.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_write_bw_only(rows):
    """Landing-page chart: ib_write_bw aggregated across rails, one line
    per transport. For every (transport, size) we take the best per-rail
    throughput across QPs and multiply by the transport's rail count.
    """
    # Number of physical rails each transport drives in our setup.
    RAILS = {
        "native":       4,  # 4× usb4_rdma HCAs across 4 cables
        "rxe_tbnet":    2,  # rxe_tb0 + rxe_tb1, one per cable
        "rxe_ethernet": 1,  # single 2.5 GbE link
    }
    AGG_LABEL = {
        "native":       "native usb4_rdma (4-rail aggregate)",
        "rxe_tbnet":    "RXE over TBnet (2-rail aggregate)",
        "rxe_ethernet": "RXE over 2.5G LAN (single link)",
    }

    agg = aggregate(rows, kind="bw")
    # (transport, size) → best per-rail bw across qps for ib_write_bw
    best: dict[tuple[str, int], float] = {}
    for (v, t, sz, qps), val in agg.items():
        if v != "write":
            continue
        key = (t, sz)
        if val > best.get(key, 0):
            best[key] = val

    # Scale up to aggregate by rail count.
    aggregated: dict[tuple[str, int], float] = {
        (t, sz): val * RAILS.get(t, 1) for (t, sz), val in best.items()
    }
    all_sizes = {sz for (_t, sz) in aggregated.keys()}

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (t, sz), val in aggregated.items():
        groups[t].append((sz, val))
    for t in [t for _, t in CSV_TO_TRANSPORT if t in groups]:
        pts = sorted(groups[t])
        xs, ys = zip(*pts)
        ax.plot(xs, ys,
                color=TRANSPORT_COLOR.get(t, "gray"),
                marker="o", markersize=5, linewidth=2,
                label=AGG_LABEL.get(t, t))
        # Annotate the peak value at the rightmost point of the line.
        peak_x, peak_y = max(pts, key=lambda p: p[1])
        ax.annotate(
            f"peak {peak_y:.1f} Gb/s",
            xy=(peak_x, peak_y), xytext=(6, 0),
            textcoords="offset points",
            fontsize=9, color=TRANSPORT_COLOR.get(t, "gray"),
            va="center", ha="left", fontweight="bold",
        )

    style_size_axis(ax, all_sizes)
    ax.set_xlabel("Message size")
    ax.set_ylabel("Aggregate bandwidth across all rails (Gb/s)")
    ymax = max(aggregated.values()) if aggregated else 1.0
    ax.set_ylim(bottom=0, top=ymax * 1.15)
    ax.set_xlim(right=(1 << 21))  # leave room for peak annotation
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(
        "ib_write_bw aggregated across rails, strix-1 ↔ strix-2  "
        "(2026-05-27 IOMMU-off sweep, best per-rail × rail count)",
        fontsize=11,
    )
    ax.legend(loc="upper left", title="Transport", frameon=True)

    fig.tight_layout()
    out = BENCH_DIR / "write_bw_vs_size.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    rows = load_rows()
    plot_bw_vs_size(rows)
    plot_lat_vs_size(rows)
    plot_write_bw_only(rows)


if __name__ == "__main__":
    main()
