"""
results/plot_scores.py
======================
TASK B-13 (Bonus) — Standalone score visualisation.

Reads results/comparison_table.csv and produces:
  1. results/scores_comparison.png  — grouped bar chart (model × Web-CodeBLEU mean + std)
  2. results/scores_by_framework.png — breakdown per framework

Usage
-----
    python results/plot_scores.py
    python results/plot_scores.py --csv results/comparison_table.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Palette (dark theme matching sprint document)
# ─────────────────────────────────────────────────────────────────────────────

BG       = "#0b0e14"
SURFACE  = "#13171f"
BORDER   = "#262d3d"
COLOURS  = {
    "Gemini":  "#3d8fff",
    "Llama":   "#c084fc",
    "Mixtral": "#ffa040",
    "Agent":   "#3dffa0",
}
TEXT_COL = "#cdd6f4"
MUTED    = "#6c7a9c"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(axis="y", color=BORDER, linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 — Overall model comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_overall(df: pd.DataFrame, out_path: Path) -> None:
    model_map = {
        "Gemini":  "gemini_codebleu",
        "Llama":   "llama_codebleu",
        "Mixtral": "mixtral_codebleu",
        "Agent":   "agent_codebleu",
    }

    labels = list(model_map.keys())
    means  = [df[c].mean()  for c in model_map.values()]
    stds   = [df[c].std()   for c in model_map.values()]
    colours = [COLOURS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    bars = ax.bar(
        labels, means,
        yerr=stds,
        color=colours,
        capsize=6,
        error_kw={"ecolor": TEXT_COL, "linewidth": 1.2, "capthick": 1.2},
        width=0.55,
        zorder=3,
    )
    ax.set_ylim(0, min(max(means) * 1.35, 1.0))
    ax.set_title("Web-LLM-Benchmark — Mean Web-CodeBLEU by Model",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Mean Web-CodeBLEU")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _style_axes(ax)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.015,
            f"{mean:.3f}",
            ha="center", va="bottom", color=TEXT_COL, fontsize=9, fontweight="bold",
        )

    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2 — Breakdown by framework
# ─────────────────────────────────────────────────────────────────────────────

def plot_by_framework(df: pd.DataFrame, out_path: Path) -> None:
    model_map = {
        "Gemini":  "gemini_codebleu",
        "Llama":   "llama_codebleu",
        "Mixtral": "mixtral_codebleu",
        "Agent":   "agent_codebleu",
    }

    frameworks = sorted(df["framework"].dropna().unique().tolist())
    n_fw       = len(frameworks)
    n_models   = len(model_map)

    if n_fw == 0:
        print("No framework data available for breakdown chart.")
        return

    x     = np.arange(n_fw)
    width = 0.18
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)

    for i, (label, col) in enumerate(model_map.items()):
        fw_means = [
            df[df["framework"] == fw][col].mean() if col in df.columns else 0
            for fw in frameworks
        ]
        fw_stds = [
            df[df["framework"] == fw][col].std() if col in df.columns else 0
            for fw in frameworks
        ]
        ax.bar(
            x + offsets[i], fw_means,
            width=width,
            label=label,
            color=COLOURS[label],
            yerr=fw_stds,
            capsize=4,
            error_kw={"ecolor": TEXT_COL, "linewidth": 1.0},
            alpha=0.88,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(frameworks)
    ax.set_ylim(0, 1.05)
    ax.set_title("Web-CodeBLEU by Framework", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Mean Web-CodeBLEU")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _style_axes(ax)
    legend = ax.legend(
        facecolor=SURFACE, labelcolor=TEXT_COL,
        framealpha=0.9, edgecolor=BORDER, fontsize=9,
    )

    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Web-LLM-Benchmark score visualiser")
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).parent / "comparison_table.csv"),
        help="Path to comparison_table.csv",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.  Run `python -m src.main --steps export` first.")
        raise SystemExit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(df[["id", "framework", "gemini_codebleu", "llama_codebleu",
              "mixtral_codebleu", "agent_codebleu"]].to_string(index=False))

    out_dir = csv_path.parent
    plot_overall(df,      out_dir / "scores_comparison.png")
    plot_by_framework(df, out_dir / "scores_by_framework.png")

    print("\nAll charts saved. Include these in the README under Section 6.")


if __name__ == "__main__":
    main()
