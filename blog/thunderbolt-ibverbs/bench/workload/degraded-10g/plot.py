#!/usr/bin/env python3
"""Plot full-speed vs degraded-10g vLLM throughput for two dense models.

Loads four CSVs in this directory (vllm-pair schema):
  - qwen3-0p6b at 20G and 10G
  - llama3-8b at 20G and 10G

Each file has up to two rows (solo + TP=2 4-HCA RDMA). The TP=2 row at
10G is missing or status=failed for both models — the chart renders an
empty hatched bar with a FAILED annotation in that case.

Output: tps_vs_link_speed.svg (one panel per model).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BENCH_DIR = Path(__file__).parent

# (filename glob fragment) → (model label, panel order)
MODELS = [
    ("qwen3-0p6b", "Qwen3-0.6B"),
    ("llama3-8b",  "Llama 3.1 8B"),
]
LINKS = [
    (20, "20 Gb/s × 4 (full)"),
    (10, "10 Gb/s × 4 (degraded)"),
]
TRANSPORTS = [
    ("solo",      "Solo (single host)",       "#7f7f7f"),
    ("usb4_rdma", "TP=2 RDMA (4-HCA)",        "#d62728"),
]


def to_float(s) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_one(path: Path) -> dict[str, dict]:
    """transport → row dict for the rows in `path`."""
    out: dict[str, dict] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            out[r["transport"]] = r
    return out


def load_all() -> dict[tuple[str, int], dict[str, dict]]:
    """(model_key, link_g) → { transport → row }."""
    out: dict[tuple[str, int], dict[str, dict]] = {}
    for csv_path in sorted(BENCH_DIR.glob("*.csv")):
        name = csv_path.name
        link = 10 if "-10g" in name else 20 if "-20g" in name else None
        if link is None:
            continue
        model_key = next((k for k, _ in MODELS if k in name), None)
        if model_key is None:
            continue
        out[(model_key, link)] = load_one(csv_path)
    return out


def plot(rows: dict[tuple[str, int], dict[str, dict]]) -> None:
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11.0, 4.2), sharey=False)
    bar_w = 0.36

    for ax, (model_key, model_label) in zip(axes, MODELS):
        n_links = len(LINKS)
        x = list(range(n_links))
        for ti, (tkey, tlabel, color) in enumerate(TRANSPORTS):
            offsets = [px + (ti - (len(TRANSPORTS) - 1) / 2) * bar_w for px in x]
            ys: list[float] = []
            failed: list[bool] = []
            for link_g, _ in LINKS:
                row = rows.get((model_key, link_g), {}).get(tkey)
                val = to_float(row.get("total_tps")) if row else None
                if val is None or (row and row.get("status") != "ok"):
                    ys.append(0.0)
                    failed.append(True)
                else:
                    ys.append(val)
                    failed.append(False)
            # Draw solid bars for OK runs and hatched empty bars for failures.
            for ox, y, fl in zip(offsets, ys, failed):
                if fl:
                    # Tall faint hatched outline placeholder; height = a
                    # 25% sliver so the FAILED label has somewhere to sit.
                    placeholder = max([v for v in ys if v > 0], default=1.0) * 0.25
                    ax.bar(ox, placeholder, bar_w,
                           color="white", edgecolor=color, linewidth=1.0,
                           hatch="//", alpha=0.7)
                    ax.text(ox, placeholder + max(ys) * 0.02,
                            "FAILED", ha="center", va="bottom",
                            fontsize=8, color=color, fontweight="bold")
                else:
                    ax.bar(ox, y, bar_w, color=color,
                           edgecolor="black", linewidth=0.4)
                    ax.text(ox, y + max(ys) * 0.02,
                            f"{y:,.0f}", ha="center", va="bottom",
                            fontsize=8, color="#222")

        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in LINKS], fontsize=9)
        ax.set_title(model_label, fontsize=11)
        ax.set_ylabel("Total throughput (tok/s)")
        ax.set_ylim(bottom=0)
        # leave headroom for the value labels / FAILED tag
        ymax = max(
            (v for d in rows.get((model_key, link_g), {}).values()
             for link_g, _ in LINKS
             for v in [to_float(d.get("total_tps"))] if v),
            default=1.0,
        )
        ax.set_ylim(top=ymax * 1.25)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "vLLM throughput at concurrency 256 — full 20 Gb/s/cable vs degraded 10 Gb/s/cable  "
        "(strix-1 ↔ strix-2, 2026-05-25)",
        fontsize=10, y=0.99,
    )
    fig.legend(
        handles=[Patch(facecolor=c, edgecolor="black", label=l)
                 for _, l, c in TRANSPORTS]
        + [Patch(facecolor="white", edgecolor="#666", hatch="//",
                 label="failed run (no throughput)")],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3, frameon=True, fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out = BENCH_DIR / "tps_vs_link_speed.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    rows = load_all()
    if not rows:
        print("no CSV rows found")
        return
    plot(rows)


if __name__ == "__main__":
    main()
