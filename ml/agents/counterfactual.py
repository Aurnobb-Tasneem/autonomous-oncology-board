"""
ml/agents/counterfactual.py
============================
Counterfactual Reasoning Agent — "What if the biopsy showed X?"

Interactive agent that takes a completed ManagementPlan and a set of
hypothetical edits (e.g. {"egfr_status": "negative"}) and produces a
revised plan showing what would change.

This is the "What if..." button in the UI.

Usage:
    agent = CounterfactualAgent(llm_client=llm)
    result = agent.replan(
        original_plan=plan,
        edits={"egfr_status": "EGFR negative (wild-type)"},
    )
    # result.changed_sections shows a diff-like view
    # result.revised_plan is the full alternative plan text
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from ml.models.llm_client import OllamaClient

log = logging.getLogger(__name__)

COUNTERFACTUAL_SYSTEM = """You are a senior oncologist on a multidisciplinary tumour board.
You are presented with an existing patient management plan and a hypothetical change to one or
more clinical parameters.

Your task is to revise the management plan based on the counterfactual assumption and explain
what changes (and why). Be specific about which treatment lines change, which tests are no longer
needed, and which guideline categories now apply.

Output ONLY a JSON object with keys:
  revised_first_line (string — the new first-line treatment under the counterfactual),
  revised_staging (string — how TNM staging would change, or "unchanged"),
  changed_sections (list of strings — describe what changed and why, 2-5 bullet points),
  unchanged_sections (list of strings — what stays the same),
  clinical_reasoning (string — 2-3 sentences explaining the key logic),
  confidence (float 0-1 — how certain you are about this counterfactual prediction).
Output only valid JSON."""


@dataclass
class CounterfactualPlan:
    """Result of a counterfactual replanning step."""
    edits:               dict       # The hypothetical changes applied
    revised_first_line:  str
    revised_staging:     str
    changed_sections:    list[str]
    unchanged_sections:  list[str]
    clinical_reasoning:  str
    confidence:          float
    source:              str        # "llm" | "fallback"
    error:               Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def diff_summary(self) -> str:
        """One-line summary of what changed."""
        if not self.changed_sections:
            return "No significant changes identified."
        return "; ".join(self.changed_sections[:2])


def _fallback_counterfactual(
    original_plan,
    edits: dict,
) -> CounterfactualPlan:
    """Return a basic counterfactual when LLM is unavailable."""
    changes = []
    for key, value in edits.items():
        changes.append(f"Counterfactual: {key.replace('_', ' ')} → {value} — clinical impact requires full pipeline re-run.")

    return CounterfactualPlan(
        edits=edits,
        revised_first_line="Requires full pipeline re-run with updated biomarker status",
        revised_staging="Unable to determine without full re-analysis",
        changed_sections=changes or ["Unable to assess without LLM re-planning"],
        unchanged_sections=["Pathology report (GigaPath analysis unchanged)"],
        clinical_reasoning="LLM unavailable — counterfactual inference requires Llama 3.3 70B.",
        confidence=0.0,
        source="fallback",
        error="LLM unavailable",
    )


class CounterfactualAgent:
    """
    Counterfactual Reasoning Agent.

    Takes a completed ManagementPlan and hypothetical parameter edits,
    uses Llama 3.3 70B to produce a revised plan with diff reasoning.
    """

    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client

    def replan(
        self,
        original_plan,
        edits: dict[str, str],
    ) -> CounterfactualPlan:
        """
        Generate a counterfactual management plan.

        Args:
            original_plan: ManagementPlan from the standard AOB run.
            edits:         Dict of parameter name → new hypothetical value.
                           e.g. {"egfr_status": "EGFR negative (wild-type)"}
                           or   {"tnm_stage": "Stage II (T2a N0 M0)"}

        Returns:
            CounterfactualPlan showing what changed + clinical reasoning.
        """
        if not edits:
            return CounterfactualPlan(
                edits={}, revised_first_line="", revised_staging="unchanged",
                changed_sections=["No edits provided — plan unchanged"],
                unchanged_sections=[], clinical_reasoning="", confidence=1.0,
                source="fallback",
            )

        # Summarise original plan
        plan_dict = original_plan.to_dict() if hasattr(original_plan, "to_dict") else {}
        diagnosis = plan_dict.get("diagnosis", {})
        tx_plan   = plan_dict.get("treatment_plan", {})

        orig_summary = (
            f"Original diagnosis: {diagnosis.get('primary', 'unknown')} "
            f"({diagnosis.get('tnm_stage', 'unknown')}).\n"
            f"Original first-line treatment: {tx_plan.get('first_line', 'unknown')}.\n"
            f"Original rationale: {tx_plan.get('rationale', 'unknown')}."
        )

        edits_str = "\n".join(f"  • {k.replace('_', ' ')}: {v}" for k, v in edits.items())

        user_prompt = (
            f"EXISTING MANAGEMENT PLAN:\n{orig_summary}\n\n"
            f"HYPOTHETICAL CHANGES TO APPLY:\n{edits_str}\n\n"
            f"How would the management plan change if these hypothetical values were true? "
            f"Focus on treatment line changes, staging updates, and guideline category shifts."
        )

        try:
            raw = self.llm.generate(
                system_prompt=COUNTERFACTUAL_SYSTEM,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=512,
            )
            result = self._parse(raw, edits)
            if result:
                return result
        except Exception as e:
            log.warning(f"CounterfactualAgent LLM failed: {e}")

        return _fallback_counterfactual(original_plan, edits)

    def _parse(self, raw: str, edits: dict) -> Optional[CounterfactualPlan]:
        parsed = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

        if parsed is None:
            return None

        return CounterfactualPlan(
            edits=edits,
            revised_first_line=str(parsed.get("revised_first_line", "")),
            revised_staging=str(parsed.get("revised_staging", "unchanged")),
            changed_sections=list(parsed.get("changed_sections", [])),
            unchanged_sections=list(parsed.get("unchanged_sections", [])),
            clinical_reasoning=str(parsed.get("clinical_reasoning", "")),
            confidence=float(parsed.get("confidence", 0.5)),
            source="llm",
        )
