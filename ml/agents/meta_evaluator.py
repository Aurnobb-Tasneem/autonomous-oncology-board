"""
ml/agents/meta_evaluator.py
============================
Meta-Evaluator — Scores consensus between draft and revised oncology plans.

This agent acts as a neutral arbiter in the Agent Debate loop:
  1. Receives the original draft plan and the revised plan
  2. Uses Llama 3.3 70B to score how much the revision addressed the critique
  3. Returns a consensus_score (0–100) and reasoning
  4. If score < 70, the board triggers another debate round (max 3 total)

Consensus score interpretation:
  90–100: Full consensus — revision fully addressed all challenges
  70–89:  Sufficient consensus — proceed to final report
  50–69:  Partial consensus — trigger another debate round
  0–49:   Low consensus — further revision required
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from ml.models.llm_client import OllamaClient

log = logging.getLogger(__name__)

META_SYSTEM = """You are a neutral medical peer reviewer evaluating whether a revised oncology
management plan has adequately addressed a clinical critique. Be objective and rigorous.
Output valid JSON only — no markdown fences, no preamble."""


class MetaEvaluator:
    """
    Neutral consensus scorer for the Agent Debate loop.

    Scores whether the Oncologist's revision adequately addressed
    the Researcher's clinical challenge.
    """

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.llm = llm_client or OllamaClient()
        log.info("MetaEvaluator: initialised")

    def evaluate(
        self,
        original_first_line: str,
        original_actions: list[str],
        critique: str,
        revised_first_line: str,
        revised_actions: list[str],
        revised_notes: str,
    ) -> dict:
        """
        Score consensus between original draft and revised plan.

        Args:
            original_first_line: First-line treatment from the original plan.
            original_actions:    Immediate actions from the original plan.
            critique:            The Researcher's challenge text.
            revised_first_line:  First-line treatment from the revised plan.
            revised_actions:     Immediate actions from the revised plan.
            revised_notes:       Oncologist's revision notes.

        Returns:
            dict with keys: consensus_score (int), reasoning (str),
            addressed_points (list), unaddressed_points (list)
        """
        prompt = f"""You are evaluating whether an oncology management plan revision
adequately addressed a clinical critique.

ORIGINAL PLAN (before critique):
  First-line treatment: {original_first_line}
  Immediate actions: {'; '.join(original_actions[:4])}

CLINICAL CRITIQUE FROM RESEARCHER:
  {critique}

REVISED PLAN (after critique):
  First-line treatment: {revised_first_line}
  Immediate actions: {'; '.join(revised_actions[:4])}
  Oncologist revision notes: {revised_notes}

Score the revision on a 0–100 scale:
  90-100: Critique fully addressed — plan is now guideline-compliant
  70-89:  Critique substantially addressed — minor gaps remain
  50-69:  Critique partially addressed — key concerns still outstanding
  0-49:   Critique not adequately addressed — significant revision needed

Return a JSON object with exactly these fields:
{{
  "consensus_score": 82,
  "reasoning": "The revision appropriately added EGFR testing before TKI selection...",
  "addressed_points": ["Added molecular testing before targeted therapy"],
  "unaddressed_points": ["PD-L1 threshold for pembrolizumab monotherapy not specified"]
}}

Return only the JSON. No markdown."""

        try:
            if not self.llm.ping():
                log.warning("MetaEvaluator: LLM unavailable — using heuristic scoring")
                return self._heuristic_score(original_first_line, revised_first_line, critique)

            response = self.llm.generate_sync(
                prompt=prompt,
                system=META_SYSTEM,
                temperature=0.05,
                max_tokens=400,
            )
            text = response.text.strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            result = json.loads(match.group() if match else text)
            score = int(result.get("consensus_score", 75))
            log.info(f"MetaEvaluator: consensus_score={score}")
            return result

        except Exception as e:
            log.warning(f"MetaEvaluator: evaluation failed ({e}), using heuristic")
            return self._heuristic_score(original_first_line, revised_first_line, critique)

    def _heuristic_score(
        self,
        original: str,
        revised: str,
        critique: str,
    ) -> dict:
        """Heuristic fallback: did the plan text actually change?"""
        changed = original.strip().lower() != revised.strip().lower()
        score = 80 if changed else 45
        return {
            "consensus_score": score,
            "reasoning": "Plan was revised in response to critique." if changed
                         else "Plan text unchanged — critique may not have been incorporated.",
            "addressed_points": ["Plan updated"] if changed else [],
            "unaddressed_points": [] if changed else ["No changes detected"],
        }
