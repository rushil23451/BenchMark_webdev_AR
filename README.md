# Web-LLM-Benchmark

> **Domain-Specific LLM Coding Benchmark for Python Web Programming**  
> *Comparing Zero-Shot LLM Performance against a ReAct Multi-Agent System using Web-CodeBLEU*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Domain & Dataset](#2-domain--dataset)
3. [Extended CodeBLEU — Web-CodeBLEU](#3-extended-codebleu--web-codebleu)
4. [Pass@1 — Functional Correctness Metric](#4-pass1--functional-correctness-metric)
5. [Benchmark Structure](#5-benchmark-structure)
6. [Results & Analysis](#6-results--analysis)
7. [Multi-Agent ReAct System](#7-multi-agent-react-system)
8. [Quickstart](#8-quickstart)
9. [Project Structure](#9-project-structure)
10. [References](#10-references)

---

## 1. Project Overview

This project implements a **domain-specific LLM coding benchmark** for **Python Web Programming**. It evaluates three large language models in a zero-shot setting and a multi-agent ReAct framework on a curated dataset of real-world web programming tasks sourced from GitHub.

| Dimension | Detail |
|-----------|--------|
| Language  | Python 3.11+ |
| Domain    | Web Programming (Flask · FastAPI · Django) |
| Models    | Gemini 1.5 Flash · Llama-3.1-70B · Mixtral-8×7B |
| Metric    | Web-CodeBLEU (5-component) + Pass@1 |
| Agent     | ReAct multi-agent system (CodeGen → Review → Debug) |

---

## 2. Domain & Dataset

### Why Web Programming?

Python web programming is an ideal benchmark domain because:

- **Rich, unambiguous vocabulary**: Framework-specific tokens (`@app.route`, `request.json`, `jsonify`, `BaseModel`, `@app.get`) are syntactically precise and semantically non-negotiable. A model that produces generic Python without these patterns has fundamentally failed the task.
- **Objectively verifiable structure**: Unlike scientific simulation or ML code, web code structure is verifiable without execution — AST inspection plus decorator/signature checks are sufficient for meaningful Pass@1 evaluation.
- **Real-world prevalence**: Flask and FastAPI repos are among the most-starred Python repositories on GitHub, ensuring an ample supply of high-quality reference implementations.

### Dataset Collection

Tasks were collected from GitHub pull requests and feature issues in popular Flask, FastAPI, and Django repositories. Selection criteria:

- The PR/issue contains a **clear natural-language description** suitable for a zero-shot prompt.
- The reference implementation is **self-contained** (no proprietary dependencies).
- The task covers a **representative web pattern** (auth, CRUD, pagination, middleware, etc.).

### Dataset Schema

```json
{
  "id":             "web_001",
  "domain":         "Web Programming",
  "language":       "Python",
  "framework":      "Flask",
  "source":         "GitHub PR link / description",
  "prompt":         "Write a Flask REST API that ...",
  "reference_code": "from flask import ..."
}
```

The sample dataset (`data/raw_github/dataset.json`) contains representative tasks spanning Flask (JWT auth, CRUD) and FastAPI (Pydantic models, pagination, CORS middleware).

---

## 3. Extended CodeBLEU — Web-CodeBLEU

### Motivation

The original CodeBLEU metric uses four sub-scores: n-gram match, weighted n-gram match, AST syntax match, and data-flow match. While these are effective for general code, they do not reward or penalise the presence of **domain-specific idioms**. A sorting function and a Flask route function may share identical AST structures, but only one is a correct web response.

**Web-CodeBLEU** adds a fifth sub-score — **KW-F1** — that directly measures alignment on web-domain keywords.

### Keyword Tiers

| Tier | Weight | Keywords | Rationale |
|------|--------|----------|-----------|
| **Tier 1** — Framework Core | **3.0×** | `route`, `endpoint`, `request`, `response`, `middleware`, `handler`, `app`, `router` | Fundamental web primitives; their absence means wrong domain |
| **Tier 2** — HTTP & Auth | **2.0×** | `status_code`, `header`, `auth`, `jwt`, `oauth`, `cors`, `session`, `cookie`, `token`, `jsonify` | HTTP-layer concepts; critical for correct web code |
| **Tier 3** — Data & Patterns | **1.5×** | `schema`, `validator`, `serializer`, `orm`, `database`, `cache`, `blueprint`, `body`, `model`, `pydantic`, `BaseModel`, `Depends` | Data handling and patterns; important but more general |

### KW-F1 Formula

```
tokens(code)   = set of Python identifiers via tokenize.generate_tokens()

weighted_match = Σ  w_k · I(k ∈ tokens(hyp) AND k ∈ tokens(ref))
weighted_hyp   = Σ  w_k · I(k ∈ tokens(hyp))
weighted_ref   = Σ  w_k · I(k ∈ tokens(ref))

KW_precision = weighted_match / (weighted_hyp + 1e-8)
KW_recall    = weighted_match / (weighted_ref  + 1e-8)
KW_F1        = 2 · KW_precision · KW_recall / (KW_precision + KW_recall + 1e-8)
```

### Final Web-CodeBLEU Formula

```
Web_CodeBLEU = 0.20·ngram
             + 0.20·weighted_ngram
             + 0.20·syntax
             + 0.20·dataflow
             + 0.20·KW_F1
```

All five sub-scores are equally weighted at **0.20** each. The KW_F1 component replaces 5% from each of the four base sub-scores, motivated by the observation that web-domain keyword presence is equally important to structural similarity for correctness in this domain.

---

## 4. Pass@1 — Functional Correctness Metric

Pass@1 is computed as: `n_passed / k` where `k = 10` (the model is sampled 10 times at temperature=0.8 per prompt).

A single generation **passes** if it satisfies **both** tiers:

| Tier | Check | Implementation |
|------|-------|----------------|
| **Tier 1** — Syntax | `ast.parse()` succeeds | `ast.parse(code)` |
| **Tier 2** — Web Structure | Framework-specific patterns present | See below |

**Framework-specific Tier 2 checks:**

- **Flask**: `"@"` in code OR `"request"` in code (decorator or request access)
- **FastAPI**: `"@"` in code OR `"Response"` in code OR `"APIRouter"` in code OR type-annotated function found via regex
- **Django**: `"request"` in code OR `"HttpResponse"` in code OR `"JsonResponse"` in code

This two-tier approach is stronger than syntax-only checking (academic contribution) while avoiding the complexity of spinning up a live web server.

---

## 5. Benchmark Structure

The central benchmark file (`data/benchmark.json`) accumulates scores from all models:

```json
[{
  "id":             "web_001",
  "domain":         "Web Programming",
  "language":       "Python",
  "framework":      "Flask",
  "prompt":         "...",
  "reference_code": "...",
  "scores": {
    "gemini-1.5-flash": {
      "web_codebleu": 0.71, "pass_at_1": 0.90, "kw_f1": 0.68
    },
    "meta/llama-3.1-70b-instruct": {
      "web_codebleu": null, "pass_at_1": null, "kw_f1": null
    },
    "mistralai/mixtral-8x7b-instruct-v0.1": {
      "web_codebleu": null, "pass_at_1": null, "kw_f1": null
    },
    "agent_system": {
      "web_codebleu": null,
      "pass_at_1": null
      // Pass@1 not computed for agent — single deterministic output per sample
    }
  }
}]
```

The `results/comparison_table.csv` provides a flat view across all models:

```
id, framework, gemini_codebleu, gemini_pass1, llama_codebleu, llama_pass1,
               mixtral_codebleu, mixtral_pass1, agent_codebleu
```

---

## 6. Results & Analysis

> *This section is populated after running `python -m src.main`. Values below are illustrative placeholders.*

### Overall Mean Scores

| Model | Web-CodeBLEU ↑ | Pass@1 ↑ | KW-F1 ↑ |
|-------|:--------------:|:--------:|:-------:|
| Gemini 1.5 Flash | — | — | — |
| Llama-3.1-70B | — | — | — |
| Mixtral-8×7B | — | — | — |
| **Agent System** | **—** | *N/A* | — |

### Results by Framework

| Framework | Gemini CB | Llama CB | Mixtral CB | Agent CB |
|-----------|:---------:|:--------:|:----------:|:--------:|
| Flask | — | — | — | — |
| FastAPI | — | — | — | — |
| Django | — | — | — | — |

### Interpretation

*(To be written by Dev A after receiving `comparison_table.csv` from Dev B on Day 18.)*

Expected findings to address:
- Did the ReAct agent improve Web-CodeBLEU over the best zero-shot model?
- Which framework benefited most from the agentic loop?
- Did the Reviewer's JSON feedback result in meaningful corrections by the Debugger?
- Which zero-shot model (Llama vs Mixtral) performs better on Flask vs FastAPI tasks?

Visualisations: `results/scores_comparison.png` (generated by `src/main.py`).

---

## 7. Multi-Agent ReAct System

The agentic framework follows the multi-agent ReAct architecture described in [Wang et al., 2024](https://arxiv.org/abs/2408.08927).

### Architecture

```
Orchestrator
  │
  ├─ Iteration 0  ──────────────→ CodeGenAgent (Llama-3.1-70B)
  │                                   │  output_code → context.current_code
  │
  ├─ Iterations 1..N
  │     │
  │     ├──────────────────────→ ReviewerAgent (Mixtral-8×7B)
  │     │    is_correct=True   └─→ DONE
  │     │    has issues        └─→ DebuggerAgent (Mixtral-8×7B)
  │     │                              │  fixed_code → context.current_code
  │     └──────────────────────────────┘  (loop back to REVIEW)
  │
  └─ Max iterations reached  ──→ Return best code so far
```

### Agent Roles

| Agent | Model | Temperature | Responsibility |
|-------|-------|:-----------:|----------------|
| **CodeGenAgent** | Llama-3.1-70B | 0.2 | Initial generation; regeneration when code is fundamentally wrong |
| **ReviewerAgent** | Mixtral-8×7B | 0.1 | Web-correctness analysis; structured JSON feedback (`is_correct`, `issues`, `suggestions`) |
| **DebuggerAgent** | Mixtral-8×7B | 0.2 | Applies all accumulated feedback to fix the code; validates output with `ast.parse()` |

### AgentContext (shared state)

```python
@dataclass
class AgentContext:
    original_prompt:  str
    framework:        str          # "Flask" | "FastAPI" | "Django"
    current_code:     str = ""
    feedback_history: List[str] = field(default_factory=list)
    iteration:        int = 0
```

### Error Handling

- `ReviewerAgent` wraps all JSON parsing in `try/except`; on failure the raw response text is used as feedback and the loop continues.
- `DebuggerAgent` validates its output with `ast.parse()`; if the output is not valid Python, `should_continue=True` is returned and the Orchestrator retries.
- `nim_generate()` retries up to 3 times on transient API errors with exponential back-off.

---

## 8. Quickstart

### Prerequisites

```bash
python --version   # 3.11+
pip install -r requirements.txt
```

### API Key

```bash
# Sign up at https://build.nvidia.com — free tier, no credit card needed
cp .env.example .env
# Edit .env and set NIM_API_KEY=nvapi-...
```

### Run the full pipeline

```bash
# Full pipeline (NIM calls included — ~45 min for both models)
python -m src.main

# Skip NIM calls (run metric code only, useful for testing)
python -m src.main --skip-nim --steps score,export,plot

# Run specific steps
python -m src.main --steps zero_shot,score

# Test Web-CodeBLEU independently
python -m src.metrics
```

### Add more dataset samples

Append entries to `data/raw_github/dataset.json` following the schema in Section 5.  
The prompts file at `data/prompts/prompts.json` is auto-generated on first run.

---

## 9. Project Structure

```
web-llm-benchmark/
├── .env.example                        # API key template
├── requirements.txt
├── README.md
│
├── src/
│   ├── metrics.py                      # Web-CodeBLEU + Pass@k + benchmark helpers
│   ├── nim_client.py                   # NVIDIA NIM API wrapper (OpenAI-compatible)
│   ├── agents.py                       # AgentContext/Result + 3 agents + Orchestrator
│   └── main.py                         # End-to-end pipeline runner
│
├── data/
│   ├── raw_github/
│   │   └── dataset.json                # Reference tasks + code
│   ├── prompts/
│   │   └── prompts.json                # Zero-shot prompts (auto-generated)
│   ├── gemini_outputs/                 # Dev A delivers: gemini_single.json, gemini_pass1.json
│   ├── nim_outputs/
│   │   ├── llama_single.json           # Single-shot Llama outputs
│   │   ├── llama_pass1.json            # Pass@1 Llama outputs (k=10)
│   │   ├── mixtral_single.json         # Single-shot Mixtral outputs
│   │   └── mixtral_pass1.json          # Pass@1 Mixtral outputs (k=10)
│   ├── scores/
│   │   ├── gemini_codebleu.json
│   │   ├── gemini_pass1_scores.json
│   │   ├── llama_codebleu.json
│   │   ├── llama_pass1_scores.json
│   │   ├── mixtral_codebleu.json
│   │   ├── mixtral_pass1_scores.json
│   │   └── agent_codebleu.json
│   ├── agent_outputs/
│   │   └── agent_final.json            # Final agent-generated code
│   └── benchmark.json                  # Central benchmark file
│
└── results/
    ├── comparison_table.csv            # All model scores in one table
    ├── midpoint_scores.txt             # Summary statistics
    └── scores_comparison.png           # Grouped bar chart (bonus)
```

---

## 10. References

1. Wang, Z. et al. (2024). *Executable Code Actions Elicit Better LLM Agents*. arXiv:2408.08927.
2. Lu, S. et al. (2021). *CodeBLEU: a Method for Automatic Evaluation of Code Synthesis*. arXiv:2009.10297.
3. Chen, M. et al. (2021). *Evaluating Large Language Models Trained on Code* (HumanEval / Pass@k). arXiv:2107.03374.
4. NVIDIA NIM API Documentation: https://build.nvidia.com
5. Flask Documentation: https://flask.palletsprojects.com
6. FastAPI Documentation: https://fastapi.tiangolo.com
