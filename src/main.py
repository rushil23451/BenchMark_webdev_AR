"""
src/main.py
===========
End-to-end execution script for the Web-LLM-Benchmark project.

Pipeline
--------
  1. Load dataset  (data/raw_github/dataset.json)
  2. Generate prompts file if not present
  3. Zero-shot evaluation → Llama-3.1-70B + Mixtral-8x7B
     a. Single-shot  (temp=0.2) → Web-CodeBLEU
     b. Pass@1 × 10  (temp=0.8) → Pass@1
  4. Build / update benchmark.json
  5. Agentic evaluation (ReAct multi-agent loop)
  6. Score agent outputs
  7. Export comparison_table.csv + midpoint_scores.txt
  8. (Bonus) Generate scores_comparison.png

Run
---
    python -m src.main
    # or with step control:
    python -m src.main --steps zero_shot,agentic,export
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ── project imports ────────────────────────────────────────────────────────────
from src.agents import run_agentic_eval
from src.metrics import (
    build_empty_benchmark,
    compute_web_codebleu,
    fill_scores,
    score_file,
    score_pass_at_1_file,
)
from src.nim_client import NIM_MODELS, run_pass_at_1, run_single_shot

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("run.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).parent.parent
DATA          = ROOT / "data"
RESULTS       = ROOT / "results"

DATASET       = DATA / "raw_github"   / "dataset.json"
PROMPTS       = DATA / "prompts"      / "prompts.json"
GEMINI_SINGLE = DATA / "gemini_outputs" / "gemini_single.json"
GEMINI_PASS1  = DATA / "gemini_outputs" / "gemini_pass1.json"

NIM_OUT       = DATA / "nim_outputs"
SCORES        = DATA / "scores"

BENCHMARK     = DATA / "benchmark.json"
AGENT_OUTPUT  = DATA / "agent_outputs" / "agent_final.json"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Dataset & Prompts
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset() -> list[dict]:
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET}\n"
            "Place your dataset.json in data/raw_github/"
        )
    with open(DATASET) as f:
        data = json.load(f)
    logger.info("Dataset loaded: %d samples", len(data))
    return data


def ensure_prompts(dataset: list[dict]) -> None:
    """
    If prompts.json does not exist, auto-generate one from the dataset.
    Each entry's 'prompt' field becomes the user-facing zero-shot prompt.
    """
    if PROMPTS.exists():
        logger.info("Prompts file already exists: %s", PROMPTS)
        return

    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    prompts = [{"id": d["id"], "prompt": d["prompt"]} for d in dataset]
    with open(PROMPTS, "w") as f:
        json.dump(prompts, f, indent=2)
    logger.info("Prompts file generated: %s  (%d entries)", PROMPTS, len(prompts))


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Zero-shot evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_zero_shot() -> None:
    """
    Generate single-shot and Pass@1×10 outputs for both NIM models.
    Skips any file that already exists (safe to re-run).
    """
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("STEP: Zero-shot evaluation — Llama + Mixtral")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    for key in ("llama", "mixtral"):
        single_out = NIM_OUT / f"{key}_single.json"
        pass1_out  = NIM_OUT / f"{key}_pass1.json"

        if not single_out.exists():
            run_single_shot(PROMPTS, DATASET, model_key=key, output_path=single_out)
        else:
            logger.info("[%s] Single-shot output already exists — skipping.", key)

        if not pass1_out.exists():
            run_pass_at_1(PROMPTS, DATASET, model_key=key, output_path=pass1_out, k=10)
        else:
            logger.info("[%s] Pass@1 output already exists — skipping.", key)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_all_models(benchmark: list[dict]) -> list[dict]:
    """
    Score Gemini (if available) + both NIM models.
    Updates benchmark in-place.
    """
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("STEP: Scoring all models")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    SCORES.mkdir(parents=True, exist_ok=True)

    # ── Gemini (Dev A output) ─────────────────────────────────────────────────
    if GEMINI_SINGLE.exists() and GEMINI_PASS1.exists():
        logger.info("Scoring Gemini outputs …")
        g_cb = score_file(GEMINI_SINGLE, DATASET, "gemini-1.5-flash")
        g_p1 = score_pass_at_1_file(GEMINI_PASS1, DATASET, "gemini-1.5-flash")
        _save_json(g_cb, SCORES / "gemini_codebleu.json")
        _save_json(g_p1, SCORES / "gemini_pass1_scores.json")
        benchmark = fill_scores(benchmark, "gemini-1.5-flash", g_cb, g_p1)
    else:
        logger.warning("Gemini output files not found — skipping Gemini scoring.")

    # ── NIM models ────────────────────────────────────────────────────────────
    for key, model_name in NIM_MODELS.items():
        single_out = NIM_OUT / f"{key}_single.json"
        pass1_out  = NIM_OUT / f"{key}_pass1.json"

        if not single_out.exists():
            logger.warning("%s not found — skipping %s CodeBLEU.", single_out.name, key)
            continue

        cb = score_file(single_out, DATASET, model_name)
        _save_json(cb, SCORES / f"{key}_codebleu.json")

        p1: list[dict] = []
        if pass1_out.exists():
            p1 = score_pass_at_1_file(pass1_out, DATASET, model_name)
            _save_json(p1, SCORES / f"{key}_pass1_scores.json")

        benchmark = fill_scores(benchmark, model_name, cb, p1 or None)

    return benchmark


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Agentic evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_agentic(benchmark: list[dict]) -> list[dict]:
    """
    Run the ReAct multi-agent loop and score agent outputs.
    """
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("STEP: Agentic evaluation")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if not AGENT_OUTPUT.exists():
        run_agentic_eval(PROMPTS, DATASET, AGENT_OUTPUT, max_iterations=5)
    else:
        logger.info("Agent output already exists — skipping generation.")

    # Convert agent output to single-shot schema for scoring
    with open(AGENT_OUTPUT) as f:
        agent_data = json.load(f)

    # Build a temporary single-shot format for score_file
    single_schema = [
        {"id": e["id"], "generated_code": e["final_code"]}
        for e in agent_data
    ]
    tmp_path = DATA / "agent_outputs" / "_agent_single_tmp.json"
    _save_json(single_schema, tmp_path)

    agent_cb = score_file(tmp_path, DATASET, "agent_system")
    _save_json(agent_cb, SCORES / "agent_codebleu.json")
    # Note: Pass@1 is not computed for agent (single deterministic output per sample)
    benchmark = fill_scores(benchmark, "agent_system", agent_cb)

    tmp_path.unlink(missing_ok=True)
    return benchmark


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Export results
# ─────────────────────────────────────────────────────────────────────────────

def export_results(benchmark: list[dict]) -> pd.DataFrame:
    """
    Build comparison_table.csv and print summary statistics.
    """
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("STEP: Exporting results")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in benchmark:
        sid       = entry["id"]
        framework = entry["framework"]
        s         = entry["scores"]

        def _get(model: str, key: str):
            return s.get(model, {}).get(key)

        rows.append({
            "id":              sid,
            "framework":       framework,
            "gemini_codebleu": _get("gemini-1.5-flash",                    "web_codebleu"),
            "gemini_pass1":    _get("gemini-1.5-flash",                    "pass_at_1"),
            "llama_codebleu":  _get("meta/llama-3.1-70b-instruct",         "web_codebleu"),
            "llama_pass1":     _get("meta/llama-3.1-70b-instruct",         "pass_at_1"),
            "mixtral_codebleu":_get("mistralai/mixtral-8x7b-instruct-v0.1","web_codebleu"),
            "mixtral_pass1":   _get("mistralai/mixtral-8x7b-instruct-v0.1","pass_at_1"),
            "agent_codebleu":  _get("agent_system",                        "web_codebleu"),
        })

    df = pd.DataFrame(rows)
    csv_path = RESULTS / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    logger.info("comparison_table.csv saved → %s", csv_path)

    # ── Summary statistics ────────────────────────────────────────────────────
    metric_cols = [c for c in df.columns if c not in ("id", "framework")]
    overall = df[metric_cols].mean(numeric_only=True).rename("Overall Mean")
    by_fw   = df.groupby("framework")[metric_cols].mean(numeric_only=True)

    summary_lines = [
        "=" * 65,
        "WEB-LLM-BENCHMARK — Results Summary",
        "=" * 65,
        "",
        "Overall Mean Scores",
        "-" * 40,
        overall.to_string(),
        "",
        "Mean Scores by Framework",
        "-" * 40,
        by_fw.to_string(),
        "",
        "=" * 65,
    ]
    summary = "\n".join(summary_lines)
    print(summary)

    txt_path = RESULTS / "midpoint_scores.txt"
    txt_path.write_text(summary)
    logger.info("Summary saved → %s", txt_path)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Bonus — Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_scores(df: pd.DataFrame) -> None:
    """
    Grouped bar chart: model × Web-CodeBLEU mean (with error bars).
    Saves results/scores_comparison.png
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot generation.")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)

    # ── Overall comparison ────────────────────────────────────────────────────
    model_map = {
        "Gemini":  "gemini_codebleu",
        "Llama":   "llama_codebleu",
        "Mixtral": "mixtral_codebleu",
        "Agent":   "agent_codebleu",
    }
    labels = list(model_map.keys())
    means  = [df[c].mean() for c in model_map.values()]
    stds   = [df[c].std()  for c in model_map.values()]

    colours = ["#3d8fff", "#c084fc", "#ffa040", "#3dffa0"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0b0e14")
    fig.suptitle("Web-LLM-Benchmark: Web-CodeBLEU Comparison",
                 color="white", fontsize=14, fontweight="bold")

    # Left: overall
    ax1 = axes[0]
    ax1.set_facecolor("#13171f")
    bars = ax1.bar(labels, means, yerr=stds, color=colours,
                   capsize=5, error_kw={"ecolor": "white", "linewidth": 1.2})
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Overall Mean Web-CodeBLEU", color="white")
    ax1.set_ylabel("Web-CodeBLEU", color="white")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#262d3d")
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{mean:.3f}", ha="center", va="bottom",
                 color="white", fontsize=9)

    # Right: by framework
    ax2 = axes[1]
    ax2.set_facecolor("#13171f")
    frameworks = df["framework"].unique().tolist()
    x = np.arange(len(frameworks))
    width = 0.2

    for i, (label, col) in enumerate(model_map.items()):
        fw_means = [df[df["framework"] == fw][col].mean() for fw in frameworks]
        ax2.bar(x + i * width, fw_means, width, label=label,
                color=colours[i], alpha=0.85)

    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(frameworks, color="white")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Web-CodeBLEU by Framework", color="white")
    ax2.set_ylabel("Web-CodeBLEU", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#13171f", labelcolor="white", framealpha=0.7)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#262d3d")

    plt.tight_layout()
    out = RESULTS / "scores_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0b0e14")
    plt.close()
    logger.info("Score visualisation saved → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

VALID_STEPS = {"setup", "zero_shot", "score", "agentic", "export", "plot", "all"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Web-LLM-Benchmark — end-to-end runner"
    )
    parser.add_argument(
        "--steps",
        default="all",
        help=(
            "Comma-separated list of pipeline steps to run. "
            f"Valid: {', '.join(sorted(VALID_STEPS))}. "
            "Default: all"
        ),
    )
    parser.add_argument(
        "--skip-nim",
        action="store_true",
        help="Skip NIM API calls (useful for testing metric code only).",
    )
    return parser.parse_args()


def main() -> None:
    args  = parse_args()
    steps = {s.strip().lower() for s in args.steps.split(",")}
    run_all = "all" in steps

    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║  Web-LLM-Benchmark  |  Python / Web Domain       ║")
    logger.info("╚══════════════════════════════════════════════════╝")
    logger.info("Steps requested: %s", steps)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
    dataset = load_dataset()
    ensure_prompts(dataset)

    # ── 2. Build / load benchmark skeleton ───────────────────────────────────
    if BENCHMARK.exists():
        with open(BENCHMARK) as f:
            benchmark = json.load(f)
        logger.info("Loaded existing benchmark.json (%d entries)", len(benchmark))
    else:
        benchmark = build_empty_benchmark(DATASET)
        _save_json(benchmark, BENCHMARK)
        logger.info("Built empty benchmark.json")

    # ── 3. Zero-shot NIM evaluation ───────────────────────────────────────────
    if (run_all or "zero_shot" in steps) and not args.skip_nim:
        run_zero_shot()

    # ── 4. Scoring ────────────────────────────────────────────────────────────
    if run_all or "score" in steps:
        benchmark = score_all_models(benchmark)
        _save_json(benchmark, BENCHMARK)

    # ── 5. Agentic evaluation ─────────────────────────────────────────────────
    if (run_all or "agentic" in steps) and not args.skip_nim:
        benchmark = run_agentic(benchmark)
        _save_json(benchmark, BENCHMARK)

    # ── 6. Export ─────────────────────────────────────────────────────────────
    if run_all or "export" in steps:
        df = export_results(benchmark)

        if run_all or "plot" in steps:
            plot_scores(df)

    logger.info("Pipeline complete. Results in: %s", RESULTS)


if __name__ == "__main__":
    main()
