"""
src/nim_client.py
=================
NVIDIA NIM API client — OpenAI-compatible wrapper.

Usage
-----
    from src.nim_client import nim_generate, NIM_MODELS
    code = nim_generate(prompt="Write a Flask route...",
                        system="You are a Python web developer.",
                        model=NIM_MODELS["llama"])

Environment
-----------
    NIM_API_KEY=<your key from build.nvidia.com>
    Add to .env in project root (never commit .env to git).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

NIM_MODELS: Dict[str, str] = {
    "llama":   "meta/llama-3.1-70b-instruct",
    "mixtral": "mistralai/mixtral-8x7b-instruct-v0.1",
}

SYSTEM_PROMPT = (
    "You are a senior Python web developer. "
    "Generate complete, correct, production-quality Python web code. "
    "Pay strict attention to Python's case-sensitivity (e.g. always use lowercase 'from' and 'import'). "
    "Output ONLY the Python code — no markdown fences, no explanations."
)

# Seconds to sleep between API calls to respect NIM free-tier rate limits
RATE_LIMIT_SLEEP = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Client singleton
# ─────────────────────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    api_key = os.getenv("NIM_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "NIM_API_KEY not set. Add it to your .env file or export it as "
            "an environment variable.  Get a free key at build.nvidia.com"
        )
    return OpenAI(base_url=NIM_BASE_URL, api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Core generation function
# ─────────────────────────────────────────────────────────────────────────────

def nim_generate(
    prompt: str,
    system: str = SYSTEM_PROMPT,
    model: str = NIM_MODELS["llama"],
    temperature: float = 0.2,
    max_tokens: int = 1024,
    retries: int = 3,
    retry_sleep: float = 10.0,
) -> str:
    """
    Call NVIDIA NIM and return the generated text.

    Args:
        prompt      : user-facing prompt.
        system      : system message (persona + output format).
        model       : NIM model string from NIM_MODELS dict.
        temperature : 0.2 for single-shot, 0.8 for Pass@1 diversity.
        max_tokens  : max response tokens.
        retries     : number of retry attempts on transient errors.
        retry_sleep : seconds to wait before retrying.

    Returns:
        Generated text string (code only, per system prompt).
    """
    client = _get_client()

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "NIM call failed (attempt %d/%d): %s", attempt, retries, exc
            )
            if attempt < retries:
                time.sleep(retry_sleep)
            else:
                logger.error("All retries exhausted for model %s.", model)
                return ""

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Batch runners (single-shot + Pass@1 × 10)
# ─────────────────────────────────────────────────────────────────────────────

def run_single_shot(
    prompts_path: Path,
    dataset_path: Path,
    model_key: str,
    output_path: Path,
) -> List[Dict]:
    """
    Generate one code sample per prompt at temperature=0.2.
    Saves results to *output_path* and returns them as a list.

    Output format per entry:
        { "id": str, "model": str, "prompt": str, "generated_code": str }
    """
    with open(prompts_path)  as f: prompts  = json.load(f)
    with open(dataset_path)  as f: dataset  = {r["id"]: r for r in json.load(f)}

    model_name = NIM_MODELS[model_key]
    results: List[Dict] = []

    for entry in tqdm(prompts, desc=f"[single-shot] {model_key}"):
        sid    = entry["id"]
        prompt = entry["prompt"]
        framework = dataset.get(sid, {}).get("framework", "Flask")

        logger.info("Generating [%s] id=%s  model=%s", "single", sid, model_key)
        generated = nim_generate(prompt, model=model_name, temperature=0.2)
        generated = extract_code_block(generated)

        results.append({
            "id":             sid,
            "model":          model_name,
            "framework":      framework,
            "prompt":         prompt,
            "generated_code": generated,
        })
        time.sleep(RATE_LIMIT_SLEEP)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Single-shot results saved → %s", output_path)
    return results


def run_pass_at_1(
    prompts_path: Path,
    dataset_path: Path,
    model_key: str,
    output_path: Path,
    k: int = 10,
) -> List[Dict]:
    """
    Generate *k* code samples per prompt at temperature=0.8 (diversity).
    Saves results to *output_path* and returns them as a list.

    Output format per entry:
        { "id": str, "model": str, "framework": str, "generations": [str, ...] }
    """
    with open(prompts_path)  as f: prompts  = json.load(f)
    with open(dataset_path)  as f: dataset  = {r["id"]: r for r in json.load(f)}

    model_name = NIM_MODELS[model_key]
    results: List[Dict] = []

    for entry in tqdm(prompts, desc=f"[pass@1 ×{k}] {model_key}"):
        sid       = entry["id"]
        prompt    = entry["prompt"]
        framework = dataset.get(sid, {}).get("framework", "Flask")
        gens: List[str] = []

        for i in range(k):
            logger.info(
                "Generating [pass@1 %d/%d] id=%s  model=%s", i + 1, k, sid, model_key
            )
            code = nim_generate(prompt, model=model_name, temperature=0.8)
            code = extract_code_block(code)
            gens.append(code)
            time.sleep(RATE_LIMIT_SLEEP)

        results.append({
            "id":          sid,
            "model":       model_name,
            "framework":   framework,
            "generations": gens,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Pass@1 results saved → %s", output_path)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Code extraction utility
# ─────────────────────────────────────────────────────────────────────────────

def extract_code_block(text: str) -> str:
    """
    Strip markdown fences if the model wrapped its output in them.
    Returns the raw code string.
    """
    # Match ```python ... ``` or ``` ... ```
    match = re.search(r"```(?:python)?\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


import re  # noqa: E402  (imported here to keep extract_code_block co-located)
