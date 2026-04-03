"""
src/agents.py
=============
ReAct Multi-Agent System for the Web-LLM-Benchmark.

Architecture
------------
  Orchestrator
    ├── CodeGenAgent   (Llama-3.1-70B)   — initial / fallback code generation
    ├── ReviewerAgent  (Mixtral-8x7B)    — web-correctness JSON feedback
    └── DebuggerAgent  (Mixtral-8x7B)    — targeted bug-fixing

ReAct Loop (per benchmark sample)
----------------------------------
  Iteration 0  → GENERATE  (CodeGenAgent)
  Iteration n  → REVIEW    (ReviewerAgent)
               → if issues  → DEBUG (DebuggerAgent)  → REVIEW again
               → if clean   → DONE

Reference: https://arxiv.org/abs/2408.08927
"""

from __future__ import annotations

import ast
import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.nim_client import NIM_MODELS, RATE_LIMIT_SLEEP, extract_code_block, nim_generate

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Agent Interface Contract  (mirrors agent_types.py from Dev A)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentContext:
    """Shared mutable state threaded through every agent in the ReAct loop."""
    original_prompt:  str
    framework:        str          # "Flask" | "FastAPI" | "Django"
    current_code:     str = ""
    feedback_history: List[str] = field(default_factory=list)
    iteration:        int = 0


@dataclass
class AgentResult:
    """Returned by every agent's .run() method."""
    agent_name:      str
    output_code:     str
    feedback:        str          # Issues / reasoning text
    reasoning:       str          # Short explanation of what was done
    should_continue: bool         # True → hand back to Orchestrator for next step


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_python(code: str) -> bool:
    """Return True if code parses successfully with ast.parse."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Worker Agents
# ─────────────────────────────────────────────────────────────────────────────

class CodeGenAgent:
    """
    TASK B-09 — Initial / fallback code generator.
    Model: Llama-3.1-70B via NVIDIA NIM.
    """

    name  = "CodeGenAgent"
    model = NIM_MODELS["llama"]

    SYSTEM_PROMPT = (
        "You are a senior Python web developer. "
        "Generate complete, correct, production-quality Python web code. "
        "Output ONLY the Python code — no markdown fences, no explanations."
    )

    def run(self, context: AgentContext) -> AgentResult:
        """Generate (or regenerate) code from the original prompt."""
        logger.info("[%s] Iteration %d — generating code (framework=%s)",
                    self.name, context.iteration, context.framework)

        user_prompt = context.original_prompt

        # If previous iterations have feedback, prepend the latest issue summary
        if context.feedback_history:
            last_feedback = context.feedback_history[-1]
            user_prompt = (
                f"The previous attempt had these issues:\n{last_feedback}\n\n"
                f"Regenerate the code fixing ALL of the above issues.\n\n"
                f"Original task:\n{context.original_prompt}"
            )

        code = nim_generate(
            prompt=user_prompt,
            system=self.SYSTEM_PROMPT,
            model=self.model,
            temperature=0.2,
        )
        code = extract_code_block(code)
        time.sleep(RATE_LIMIT_SLEEP)

        return AgentResult(
            agent_name=self.name,
            output_code=code,
            feedback="",
            reasoning="Generated initial code from task prompt.",
            should_continue=True,
        )


class ReviewerAgent:
    """
    TASK B-10 — Web-correctness reviewer.
    Model: Mixtral-8x7B via NVIDIA NIM.
    Outputs structured JSON feedback.
    """

    name  = "ReviewerAgent"
    model = NIM_MODELS["mixtral"]

    SYSTEM_PROMPT = (
        "You are a senior Python web developer performing a code review. "
        "Respond ONLY with a valid JSON object — no extra text, no markdown."
    )

    REVIEWER_PROMPT_TEMPLATE = """\
You are a senior Python web developer reviewing generated code.
Framework: {framework}
Task: {original_prompt}

Code to review:
{current_code}

Check:
1. Does it use correct {framework} idioms? (decorators, request handling, response format)
2. Does it handle HTTP status codes correctly?
3. Are there security issues? (SQL injection, missing auth check, etc.)
4. Does it solve the stated task?

Respond ONLY as JSON with EXACTLY this structure (no extra keys):
{{"is_correct": true_or_false, "issues": ["issue1", "issue2"], "suggestions": ["suggestion1"]}}
"""

    def run(self, context: AgentContext) -> AgentResult:
        """Analyse current_code and return JSON feedback."""
        logger.info("[%s] Iteration %d — reviewing code", self.name, context.iteration)

        prompt = self.REVIEWER_PROMPT_TEMPLATE.format(
            framework=context.framework,
            original_prompt=context.original_prompt,
            current_code=context.current_code,
        )

        raw = nim_generate(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            model=self.model,
            temperature=0.1,   # low temp for deterministic JSON
        )
        time.sleep(RATE_LIMIT_SLEEP)

        # ── Robust JSON parsing ───────────────────────────────────────────────
        review: dict = {}
        try:
            # Strip any accidental markdown fences
            clean = raw.strip()
            if clean.startswith("```"):
                import re
                clean = re.sub(r"```(?:json)?\n?", "", clean).strip("`").strip()
            review = json.loads(clean)
        except (json.JSONDecodeError, ValueError) as exc:
            # Treat full response text as feedback; loop continues
            logger.warning("[%s] JSON parse failed (%s) — using raw text as feedback.", self.name, exc)
            return AgentResult(
                agent_name=self.name,
                output_code=context.current_code,  # reviewer never rewrites
                feedback=raw,
                reasoning="JSON parse failed; full response used as feedback.",
                should_continue=True,
            )

        is_correct = review.get("is_correct", False)
        issues     = review.get("issues", [])
        feedback   = "; ".join(issues) if issues else ""

        should_continue = not (is_correct and not issues)

        return AgentResult(
            agent_name=self.name,
            output_code=context.current_code,   # reviewer does NOT rewrite
            feedback=feedback,
            reasoning=f"Review complete. is_correct={is_correct}. Issues: {issues}",
            should_continue=should_continue,
        )


class DebuggerAgent:
    """
    TASK B-11 — Code fixer.
    Model: Mixtral-8x7B via NVIDIA NIM.
    Applies reviewer feedback to produce a corrected version.
    """

    name  = "DebuggerAgent"
    model = NIM_MODELS["mixtral"]

    SYSTEM_PROMPT = (
        "You are a senior Python web developer fixing buggy code. "
        "Output ONLY the corrected Python code — no markdown, no explanations."
    )

    DEBUGGER_PROMPT_TEMPLATE = """\
You are fixing a {framework} code snippet.
Original task: {original_prompt}

Current (buggy) code:
{current_code}

All feedback collected so far:
{all_feedback}

Fix ALL the issues listed above.
Output ONLY the corrected Python {framework} code. No explanations.
"""

    def run(self, context: AgentContext) -> AgentResult:
        """Apply all accumulated feedback to produce a corrected code version."""
        logger.info("[%s] Iteration %d — debugging code", self.name, context.iteration)

        all_feedback = "\n".join(
            f"[Iteration {i+1}] {fb}"
            for i, fb in enumerate(context.feedback_history)
        ) or "No specific feedback — improve code quality and web correctness."

        prompt = self.DEBUGGER_PROMPT_TEMPLATE.format(
            framework=context.framework,
            original_prompt=context.original_prompt,
            current_code=context.current_code,
            all_feedback=all_feedback,
        )

        code = nim_generate(
            prompt=prompt,
            system=self.SYSTEM_PROMPT,
            model=self.model,
            temperature=0.2,
        )
        code = extract_code_block(code)
        time.sleep(RATE_LIMIT_SLEEP)

        valid = is_valid_python(code)
        if not valid:
            logger.warning("[%s] Debugger produced non-parseable Python.", self.name)

        return AgentResult(
            agent_name=self.name,
            output_code=code,
            feedback="",
            reasoning="Applied all reviewer feedback to fix the code.",
            should_continue=not valid,  # if still broken, signal to loop again
        )


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — ReAct loop controller
# ─────────────────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Manages the ReAct loop for a single benchmark sample.

    Loop decisions
    --------------
      iteration == 0            → GENERATE (CodeGenAgent)
      iteration > 0, no code   → GENERATE (CodeGenAgent)
      code exists               → REVIEW  (ReviewerAgent)
        reviewer says stop      → DONE
        reviewer finds issues   → DEBUG   (DebuggerAgent)
          debugger done         → REVIEW  (back to top)
          debugger still broken → GENERATE (fallback)
    """

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.codegen  = CodeGenAgent()
        self.reviewer = ReviewerAgent()
        self.debugger = DebuggerAgent()

    def run(self, original_prompt: str, framework: str) -> dict:
        """
        Run the full ReAct loop for one task.

        Returns:
            {
                "final_code":   str,
                "iterations":   int,
                "trace":        [{"agent": str, "reasoning": str, "feedback": str}, ...],
            }
        """
        context = AgentContext(
            original_prompt=original_prompt,
            framework=framework,
        )
        trace: list[dict] = []

        for _ in range(self.max_iterations):
            context.iteration += 1
            logger.info("=== Orchestrator: Iteration %d ===", context.iteration)

            # ── Decision logic ───────────────────────────────────────────────
            if not context.current_code.strip():
                # No code yet → generate
                result = self.codegen.run(context)
            else:
                # Code exists → review
                result = self.reviewer.run(context)

                if not result.should_continue:
                    # Reviewer is satisfied → DONE
                    trace.append(_trace_entry(result))
                    logger.info("Orchestrator: Reviewer satisfied — stopping.")
                    break

                # Reviewer found issues → debug
                context.feedback_history.append(result.feedback)
                trace.append(_trace_entry(result))

                debug_result = self.debugger.run(context)
                trace.append(_trace_entry(debug_result))

                if debug_result.output_code.strip():
                    context.current_code = debug_result.output_code
                # If debugger says should_continue=True the outer loop retries

                if not debug_result.should_continue:
                    continue   # → next iteration goes to REVIEW
                else:
                    continue   # loop continues, REVIEW will catch remaining issues

            # Update shared state
            trace.append(_trace_entry(result))
            context.current_code = result.output_code

        return {
            "final_code": context.current_code,
            "iterations": context.iteration,
            "trace":      trace,
        }


def _trace_entry(result: AgentResult) -> dict:
    return {
        "agent":     result.agent_name,
        "reasoning": result.reasoning,
        "feedback":  result.feedback,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch agentic evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_agentic_eval(
    prompts_path,
    dataset_path,
    output_path,
    max_iterations: int = 5,
) -> list[dict]:
    """
    Run the Orchestrator on all prompts in *prompts_path*.
    Saves agent_final.json to *output_path*.

    Output format per entry:
        {
            "id":          str,
            "framework":   str,
            "final_code":  str,
            "iterations":  int,
            "trace":       [...],
        }
    """
    import json
    from pathlib import Path
    from tqdm import tqdm

    with open(prompts_path)  as f: prompts = json.load(f)
    with open(dataset_path)  as f: dataset = {r["id"]: r for r in json.load(f)}

    orch    = Orchestrator(max_iterations=max_iterations)
    results = []

    for entry in tqdm(prompts, desc="[agentic eval]"):
        sid       = entry["id"]
        prompt    = entry["prompt"]
        framework = dataset.get(sid, {}).get("framework", "Flask")

        logger.info("Agent loop: id=%s  framework=%s", sid, framework)
        output = orch.run(original_prompt=prompt, framework=framework)

        results.append({
            "id":         sid,
            "framework":  framework,
            "final_code": output["final_code"],
            "iterations": output["iterations"],
            "trace":      output["trace"],
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Agent outputs saved → %s", output_path)
    return results
