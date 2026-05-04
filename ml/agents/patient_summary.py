"""
ml/agents/patient_summary.py
=============================
Patient-Facing Plain-Language Summary Agent.

Converts the clinical ManagementPlan (written for oncologists) into a
clear, accessible summary for patients at an 8th-grade reading level.

Sections produced:
    "What we found"
    "What we recommend next"
    "Questions to ask your doctor"
    Prominent disclaimer (research tool, not for clinical use)

Language target: plain English, short sentences, no unexplained jargon.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from ml.models.llm_client import OllamaClient

log = logging.getLogger(__name__)

PATIENT_SUMMARY_SYSTEM = """You are a patient communication specialist at a major cancer centre.
Your job is to translate a complex oncology report into a plain-English summary for a patient.

Rules:
1. Use short sentences (maximum 20 words).
2. Avoid medical jargon. When you must use a technical term, immediately explain it in plain language.
3. Write at an 8th-grade reading level (aim for Flesch-Kincaid Grade Level ≤ 8).
4. Be honest and clear. Do not minimise serious findings, but do not cause unnecessary alarm.
5. Structure the response EXACTLY as three sections separated by the headers below:
   === WHAT WE FOUND ===
   [2-4 plain-language sentences about the diagnosis]

   === WHAT WE RECOMMEND NEXT ===
   [3-5 bullet points of recommended actions, in plain language]

   === QUESTIONS TO ASK YOUR DOCTOR ===
   [3-5 questions the patient should ask at their next appointment]

6. End with exactly this disclaimer on a new line:
   ⚠️ IMPORTANT: This is a research tool only. This summary is NOT medical advice and should NOT be used to make treatment decisions. Always discuss your diagnosis and treatment options with your own doctor.

Output plain text only — no JSON, no markdown, just the structured plain-English summary."""


def generate_patient_summary(
    llm: OllamaClient,
    management_plan,
    max_tokens: int = 600,
) -> str:
    """
    Generate a patient-facing plain-language summary from a ManagementPlan.

    Args:
        llm:             OllamaClient connected to Llama 3.3 70B.
        management_plan: ManagementPlan dataclass from oncologist.py.
        max_tokens:      Maximum tokens for the summary.

    Returns:
        Plain-text patient summary (structured with section headers).
        Returns a fallback summary on error.
    """
    plan_dict = management_plan.to_dict() if hasattr(management_plan, "to_dict") else {}
    diagnosis = plan_dict.get("diagnosis", {})
    tx_plan   = plan_dict.get("treatment_plan", {})
    actions   = plan_dict.get("immediate_actions", [])
    referrals = plan_dict.get("multidisciplinary_referrals", [])
    follow_up = plan_dict.get("follow_up", "")

    clinical_summary = (
        f"Diagnosis: {diagnosis.get('primary', 'unknown')}. "
        f"Stage: {diagnosis.get('tnm_stage', 'unknown')}. "
        f"Confidence: {diagnosis.get('confidence', 0.0):.0%}. "
        f"First-line treatment recommendation: {tx_plan.get('first_line', 'not specified')}. "
        f"Rationale: {tx_plan.get('rationale', '')}. "
        f"Immediate actions: {'; '.join(actions[:3]) if actions else 'none'}. "
        f"Referrals: {'; '.join(referrals[:3]) if referrals else 'none'}. "
        f"Follow-up: {follow_up}."
    )

    user_prompt = (
        f"Here is the oncologist's clinical management plan:\n\n{clinical_summary}\n\n"
        f"Please write a patient-friendly plain-English summary following the required structure."
    )

    try:
        raw = llm.generate(
            system_prompt=PATIENT_SUMMARY_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=max_tokens,
        )
        if raw and len(raw.strip()) > 50:
            # Ensure disclaimer is present
            if "research tool" not in raw.lower():
                raw += (
                    "\n\n⚠️ IMPORTANT: This is a research tool only. "
                    "This summary is NOT medical advice and should NOT be used to make "
                    "treatment decisions. Always discuss your diagnosis and treatment options "
                    "with your own doctor."
                )
            return raw.strip()
    except Exception as e:
        log.warning(f"PatientSummaryAgent LLM failed: {e}")

    return _fallback_summary(plan_dict)


def _fallback_summary(plan_dict: dict) -> str:
    """Return a minimal fallback summary when the LLM is unavailable."""
    diag    = plan_dict.get("diagnosis", {})
    primary = diag.get("primary", "cancer")
    stage   = diag.get("tnm_stage", "unknown")
    tx      = plan_dict.get("treatment_plan", {})
    fl      = tx.get("first_line", "to be determined by your oncologist")

    return (
        f"=== WHAT WE FOUND ===\n"
        f"Our system found signs of {primary}. The stage is {stage}.\n\n"
        f"=== WHAT WE RECOMMEND NEXT ===\n"
        f"• Speak with your oncologist about this finding.\n"
        f"• Ask about the suggested treatment: {fl}.\n"
        f"• Ask your doctor what tests you need next.\n\n"
        f"=== QUESTIONS TO ASK YOUR DOCTOR ===\n"
        f"• What does this diagnosis mean for me?\n"
        f"• What treatment options do I have?\n"
        f"• What happens if I do not start treatment right away?\n"
        f"• Are there clinical trials I could join?\n\n"
        f"⚠️ IMPORTANT: This is a research tool only. "
        f"This summary is NOT medical advice and should NOT be used to make "
        f"treatment decisions. Always discuss your diagnosis and treatment options "
        f"with your own doctor."
    )


class PatientSummaryAgent:
    """
    Patient-Facing Summary Agent.

    Wraps `generate_patient_summary` with the board's OllamaClient.
    """

    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client

    def generate(self, management_plan, max_tokens: int = 600) -> str:
        """
        Generate a plain-language patient summary from a ManagementPlan.

        Args:
            management_plan: ManagementPlan from oncologist.py.
            max_tokens:      Token budget for the summary.

        Returns:
            Plain-text patient summary string.
        """
        return generate_patient_summary(self.llm, management_plan, max_tokens)
