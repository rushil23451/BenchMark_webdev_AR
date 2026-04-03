"""
src/metrics.py
==============
Web-CodeBLEU + Pass@1 implementation for the Web-LLM-Benchmark.

Web-CodeBLEU Formula
--------------------
Web_CodeBLEU = 0.20·ngram + 0.20·weighted_ngram + 0.20·syntax
             + 0.20·dataflow + 0.20·KW_F1

KW_F1 is computed from three tiers of web-programming domain keywords, each
weighted by semantic importance:
  Tier 1 (3.0×) — Framework Core  : route, endpoint, request, response, etc.
  Tier 2 (2.0×) — HTTP & Auth     : status_code, jwt, cors, middleware, etc.
  Tier 3 (1.5×) — Data & Patterns : orm, serializer, cache, schema, etc.

Pass@1 uses a two-tier check:
  Tier 1 — Syntax  : ast.parse() succeeds
  Tier 2 — Web structure : framework-specific decorators / signatures present
"""

from __future__ import annotations

import ast
import json
import re
import tokenize
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Tiered Web-Domain Keyword Registry
# ─────────────────────────────────────────────────────────────────────────────

KEYWORD_TIERS: Dict[str, float] = {
    # ── Tier 1 — Framework Core (3.0×) ──────────────────────────────────────
    "route":        3.0,
    "endpoint":     3.0,
    "request":      3.0,
    "response":     3.0,
    "middleware":   3.0,
    "handler":      3.0,
    "app":          3.0,
    "router":       3.0,
    # ── Tier 2 — HTTP & Auth (2.0×) ─────────────────────────────────────────
    "status_code":  2.0,
    "header":       2.0,
    "auth":         2.0,
    "jwt":          2.0,
    "oauth":        2.0,
    "cors":         2.0,
    "session":      2.0,
    "cookie":       2.0,
    "token":        2.0,
    "jsonify":      2.0,
    "HTTPException": 2.0,
    "HTTPResponse": 2.0,
    # ── Tier 3 — Data & Patterns (1.5×) ─────────────────────────────────────
    "schema":       1.5,
    "validator":    1.5,
    "serializer":   1.5,
    "orm":          1.5,
    "database":     1.5,
    "cache":        1.5,
    "blueprint":    1.5,
    "query_param":  1.5,
    "body":         1.5,
    "model":        1.5,
    "pydantic":     1.5,
    "BaseModel":    1.5,
    "Depends":      1.5,
}

ALL_KEYWORDS: set[str] = set(KEYWORD_TIERS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Tokenisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _tokenise(code: str) -> set[str]:
    """Return the set of Python identifiers in *code* via tokenize."""
    tokens: set[str] = set()
    try:
        gen = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok_type, tok_val, *_ in gen:
            if tok_type == tokenize.NAME:
                tokens.add(tok_val)
    except tokenize.TokenError:
        # Partial/invalid code — fall back to a regex scan
        tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", code))
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# 3.  KW-F1 sub-score
# ─────────────────────────────────────────────────────────────────────────────

def compute_kw_f1(hypothesis: str, reference: str) -> float:
    """
    Compute the tiered Keyword F1 sub-score.

    KW_precision = weighted_match / (weighted_hyp + 1e-8)
    KW_recall    = weighted_match / (weighted_ref  + 1e-8)
    KW_F1        = 2 · P · R / (P + R + 1e-8)
    """
    hyp_tokens = _tokenise(hypothesis)
    ref_tokens = _tokenise(reference)

    weighted_match = 0.0
    weighted_hyp   = 0.0
    weighted_ref   = 0.0

    for kw, weight in KEYWORD_TIERS.items():
        in_hyp = kw in hyp_tokens
        in_ref = kw in ref_tokens
        if in_hyp and in_ref:
            weighted_match += weight
        if in_hyp:
            weighted_hyp += weight
        if in_ref:
            weighted_ref += weight

    precision = weighted_match / (weighted_hyp + 1e-8)
    recall    = weighted_match / (weighted_ref  + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return float(f1)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Web-CodeBLEU (5-component)
# ─────────────────────────────────────────────────────────────────────────────

def compute_web_codebleu(hypothesis: str, reference: str) -> Dict[str, float]:
    """
    Compute the full Web-CodeBLEU score.

    Returns a dict with keys:
        ngram, weighted_ngram, syntax, dataflow, kw_f1, web_codebleu

    The 4 base sub-scores are taken from the `codebleu` package
    (each originally weighted 0.25); here each is re-weighted to 0.20
    and the remaining 0.20 is given to KW_F1.
    """
    try:
        import codebleu.codebleu
        from codebleu import calc_codebleu  # type: ignore
        import tree_sitter
        import codebleu.utils
        
        # Monkeypatch: Handle tree-sitter-python 0.23 PyCapsule API change causing 'an integer is required'
        original_get = codebleu.utils.get_tree_sitter_language
        def patched_get(lang):
            if lang == "python":
                import tree_sitter_python
                lang_obj = tree_sitter_python.language()
                try:
                    return tree_sitter.Language(lang_obj)
                except TypeError:
                    import ctypes
                    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
                    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
                    ptr = ctypes.pythonapi.PyCapsule_GetPointer(lang_obj, b"tree_sitter.Language")
                    return tree_sitter.Language(ptr)
            return original_get(lang)
        codebleu.utils.get_tree_sitter_language = patched_get
        # MUST patch the local reference inside codebleu.codebleu where it was imported!
        codebleu.codebleu.get_tree_sitter_language = patched_get

    except ImportError as exc:
        raise ImportError(
            "codebleu package not found. Run: pip install codebleu"
        ) from exc

    if not hypothesis or not hypothesis.strip():
        return {
            "ngram": 0.0, "weighted_ngram": 0.0,
            "syntax": 0.0, "dataflow": 0.0,
            "kw_f1": 0.0, "web_codebleu": 0.0,
        }

    # calc_codebleu expects lists
    result = calc_codebleu(
        references=[[reference]],
        predictions=[hypothesis],
        lang="python",
        weights=(0.25, 0.25, 0.25, 0.25),  # default; we re-weight below
    )

    ngram    = float(result.get("ngram_match_score",          0.0))
    wngram   = float(result.get("weighted_ngram_match_score", 0.0))
    syntax   = float(result.get("syntax_match_score",         0.0))
    dataflow = float(result.get("dataflow_match_score",       0.0))
    kw_f1    = compute_kw_f1(hypothesis, reference)

    web_codebleu = 0.20 * ngram + 0.20 * wngram + 0.20 * syntax \
                 + 0.20 * dataflow + 0.20 * kw_f1

    return {
        "ngram":         round(ngram,        4),
        "weighted_ngram": round(wngram,      4),
        "syntax":        round(syntax,       4),
        "dataflow":      round(dataflow,     4),
        "kw_f1":         round(kw_f1,        4),
        "web_codebleu":  round(web_codebleu, 4),
    }


def score_file(
    generated_path: Path,
    dataset_path: Path,
    model_name: str,
) -> List[Dict]:
    """
    Score all generated codes in *generated_path* against their references
    in *dataset_path*.  Returns a list of score-dicts ready to write to JSON.
    """
    with open(dataset_path)    as f: dataset   = {r["id"]: r for r in json.load(f)}
    with open(generated_path)  as f: generated = json.load(f)

    scores = []
    for entry in generated:
        sid = entry["id"]
        ref = dataset.get(sid, {}).get("reference_code", "")
        hyp = entry.get("generated_code", "")
        if not ref:
            logger.warning("No reference found for id=%s — skipping.", sid)
            continue
        s = compute_web_codebleu(hyp, ref)
        scores.append({
            "id":             sid,
            "model":          model_name,
            **s,
        })
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Pass@1  — two-tier web correctness check
# ─────────────────────────────────────────────────────────────────────────────

def is_web_correct(code: str, framework: str) -> bool:
    """
    Two-tier Pass@1 check.

    Tier 1  — Syntax  : ast.parse() must succeed.
    Tier 2  — Structure: framework-specific signatures must be present.

    This is stronger than syntax-only (academic contribution) but avoids
    the complexity of actually running a web server.
    """
    # ── Tier 1: parse ────────────────────────────────────────────────────────
    try:
        ast.parse(code)
    except SyntaxError:
        return False

    # ── Tier 2: web structure ────────────────────────────────────────────────
    fw = framework.lower()

    if fw == "flask":
        # Must have a decorator (e.g. @app.route) or use the request object
        return "@" in code or "request" in code

    if fw == "fastapi":
        # Must have a route decorator or use FastAPI Response / type hints
        return (
            "@" in code
            or "Response" in code
            or "APIRouter" in code
            or re.search(r"def \w+\(.*:.*\)", code) is not None
        )

    if fw == "django":
        # Must handle the request param or return an HttpResponse
        return "request" in code or "HttpResponse" in code or "JsonResponse" in code

    # Generic fallback: any decorator or 'request' reference
    return "@" in code or "request" in code


def compute_pass_at_1(generations: List[str], framework: str) -> float:
    """
    Pass@1 = (number of correct generations) / len(generations).

    Args:
        generations: list of k generated code strings for one prompt.
        framework:   "Flask", "FastAPI", or "Django".

    Returns:
        float in [0, 1].
    """
    if not generations:
        return 0.0
    n_passed = sum(1 for g in generations if is_web_correct(g, framework))
    return round(n_passed / len(generations), 4)


def score_pass_at_1_file(
    pass1_path: Path,
    dataset_path: Path,
    model_name: str,
) -> List[Dict]:
    """
    Compute Pass@1 for every sample in *pass1_path*.
    pass1_path must be a JSON file where each entry has:
        { "id": ..., "generations": ["code1", "code2", ...] }
    """
    with open(dataset_path)  as f: dataset   = {r["id"]: r for r in json.load(f)}
    with open(pass1_path)    as f: pass1_data = json.load(f)

    results = []
    for entry in pass1_data:
        sid       = entry["id"]
        framework = dataset.get(sid, {}).get("framework", "Flask")
        gens      = entry.get("generations", [])
        p1        = compute_pass_at_1(gens, framework)
        results.append({
            "id":        sid,
            "model":     model_name,
            "framework": framework,
            "n_passed":  sum(1 for g in gens if is_web_correct(g, framework)),
            "n_total":   len(gens),
            "pass_at_1": p1,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Benchmark assembly helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_empty_benchmark(dataset_path: Path) -> List[Dict]:
    """
    Create the central benchmark.json skeleton from dataset.json.
    All model score slots are initialised to null.
    """
    with open(dataset_path) as f:
        dataset = json.load(f)

    benchmark = []
    for item in dataset:
        benchmark.append({
            "id":             item["id"],
            "domain":         item.get("domain", "Web Programming"),
            "language":       item.get("language", "Python"),
            "framework":      item.get("framework", "Flask"),
            "prompt":         item.get("prompt", ""),
            "reference_code": item.get("reference_code", ""),
            "scores": {
                "gemini-1.5-flash": {
                    "web_codebleu": None, "pass_at_1": None, "kw_f1": None
                },
                "meta/llama-3.1-70b-instruct": {
                    "web_codebleu": None, "pass_at_1": None, "kw_f1": None
                },
                "mistralai/mixtral-8x7b-instruct-v0.1": {
                    "web_codebleu": None, "pass_at_1": None, "kw_f1": None
                },
                "agent_system": {
                    # Pass@1 not computed for agent (deterministic single output)
                    "web_codebleu": None, "pass_at_1": None
                },
            },
        })
    return benchmark


def fill_scores(
    benchmark: List[Dict],
    model_name: str,
    codebleu_scores: List[Dict],
    pass1_scores: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Merge scored data into the benchmark list in-place.

    codebleu_scores : output of score_file()
    pass1_scores    : output of score_pass_at_1_file()  (optional)
    """
    cb_map = {s["id"]: s for s in codebleu_scores}
    p1_map = {s["id"]: s for s in (pass1_scores or [])}

    for entry in benchmark:
        sid = entry["id"]
        if sid in cb_map:
            cb = cb_map[sid]
            entry["scores"].setdefault(model_name, {})
            entry["scores"][model_name]["web_codebleu"] = cb.get("web_codebleu")
            entry["scores"][model_name]["kw_f1"]        = cb.get("kw_f1")
        if sid in p1_map:
            p1 = p1_map[sid]
            entry["scores"].setdefault(model_name, {})
            entry["scores"][model_name]["pass_at_1"] = p1.get("pass_at_1")

    return benchmark


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Quick self-test (run directly: python -m src.metrics)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint

    FLASK_A = """\
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/users', methods=['GET'])
def get_users():
    users = [{'id': 1, 'name': 'Alice'}]
    return jsonify(users), 200
"""

    FLASK_B = """\
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    name = data.get('name', '')
    return jsonify({'id': 2, 'name': name}), 201
"""

    GENERIC = """\
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""

    print("=== Similar Flask functions (expect Web-CodeBLEU ≈ 0.5+, KW_F1 ≈ 0.8+) ===")
    pprint.pprint(compute_web_codebleu(FLASK_A, FLASK_B))

    print("\n=== Flask vs generic code (expect KW_F1 ≈ 0.1) ===")
    pprint.pprint(compute_web_codebleu(GENERIC, FLASK_A))

    print("\n=== Pass@1 tests ===")
    print("Flask correct    :", is_web_correct(FLASK_A,  "Flask"))
    print("Flask incorrect  :", is_web_correct(GENERIC,  "Flask"))
    print("FastAPI correct  :", is_web_correct(
        "from fastapi import FastAPI\napp=FastAPI()\n@app.get('/a')\ndef a(): pass",
        "FastAPI"
    ))
    print("Django correct   :", is_web_correct(
        "from django.http import HttpResponse\ndef view(request): return HttpResponse('ok')",
        "Django"
    ))
