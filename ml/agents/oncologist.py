"""
ml/agents/oncologist.py
=======================
Agent 3: Oncologist — Final Patient Management Plan synthesis.

Responsibilities:
  - Receive PathologyReport (Agent 1) + ResearchSummary (Agent 2)
  - Use Llama 3.3 70B (via Ollama) to act as a senior oncologist
  - Synthesise both inputs into a complete Patient Management Plan
  - Return ManagementPlan JSON (the final output of the AOB system)

Output schema (ManagementPlan):
  {
    "case_id": str,
    "generated_at": str,          # ISO timestamp
    "patient_summary": str,
    "diagnosis": {
      "primary": str,
      "tnm_stage": str,
      "confidence": float
    },
    "immediate_actions": [str],
    "treatment_plan": {
      "first_line": str,
      "rationale": str,
      "alternatives": [str]
    },
    "further_investigations": [str],
    "multidisciplinary_referrals": [str],
    "follow_up": str,
    "confidence_score": float,
    "board_consensus": str,
    "disclaimer": str,
    "citations": [str]
  }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from ml.models.llm_client import OllamaClient
from ml.agents.pathologist import PathologyReport
from ml.agents.researcher import ResearchSummary

log = logging.getLogger(__name__)

ONCOLOGIST_SYSTEM = """You are a senior consultant oncologist chairing a multidisciplinary tumour board.
You have received a digital pathology report from an AI pathologist and a clinical research brief
from an AI researcher. Your role is to synthesise these into a complete Patient Management Plan.

Be thorough, evidence-based, and precise. Always reference the provided evidence.
Output valid JSON only — no markdown fences, no preamble."""

# ── Tissue → stage mapping (simplified for demo) ─────────────────────────────
TISSUE_STAGE_MAP = {
    "lung_adenocarcinoma":         "Stage IV NSCLC (adenocarcinoma) — pending molecular workup",
    "lung_squamous_cell_carcinoma": "Stage III-IV NSCLC (squamous cell) — pending staging CT",
    "colon_adenocarcinoma":        "Stage III colorectal adenocarcinoma — pending MSI/KRAS",
    "colon_benign_tissue":         "Benign colonic tissue — no malignancy detected",
    "lung_benign_tissue":          "Benign pulmonary tissue — no malignancy detected",
}


@dataclass
class Diagnosis:
    primary: str
    tnm_stage: str
    confidence: float


@dataclass
class TreatmentPlan:
    first_line: str
    rationale: str
    alternatives: list[str] = field(default_factory=list)


@dataclass
class ManagementPlan:
    case_id: str
    generated_at: str
    patient_summary: str
    diagnosis: Diagnosis
    immediate_actions: list[str]
    treatment_plan: TreatmentPlan
    further_investigations: list[str]
    multidisciplinary_referrals: list[str]
    follow_up: str
    confidence_score: float
    board_consensus: str
    disclaimer: str
    citations: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def format_report(self) -> str:
        """Human-readable report for the frontend."""
        lines = [
            "=" * 70,
            "  AUTONOMOUS ONCOLOGY BOARD — PATIENT MANAGEMENT PLAN",
            f"  Case ID: {self.case_id}   |   Generated: {self.generated_at}",
            "=" * 70,
            f"\n📋 PATIENT SUMMARY\n{self.patient_summary}",
            f"\n🔬 DIAGNOSIS\n  Primary: {self.diagnosis.primary}",
            f"  TNM Stage: {self.diagnosis.tnm_stage}",
            f"  Diagnostic confidence: {self.diagnosis.confidence:.0%}",
            f"\n⚡ IMMEDIATE ACTIONS",
        ]
        for a in self.immediate_actions:
            lines.append(f"  → {a}")
        lines += [
            f"\n💊 TREATMENT PLAN",
            f"  First-line: {self.treatment_plan.first_line}",
            f"  Rationale:  {self.treatment_plan.rationale}",
        ]
        if self.treatment_plan.alternatives:
            lines.append("  Alternatives:")
            for alt in self.treatment_plan.alternatives:
                lines.append(f"    • {alt}")
        lines.append("\n🔍 FURTHER INVESTIGATIONS")
        for inv in self.further_investigations:
            lines.append(f"  • {inv}")
        lines.append("\n👥 MULTIDISCIPLINARY REFERRALS")
        for ref in self.multidisciplinary_referrals:
            lines.append(f"  • {ref}")
        lines += [
            f"\n📅 FOLLOW-UP\n  {self.follow_up}",
            f"\n✅ BOARD CONSENSUS\n  Confidence: {self.confidence_score:.0%}",
            f"  {self.board_consensus}",
            f"\n📚 CITATIONS",
        ]
        for i, cite in enumerate(self.citations, 1):
            lines.append(f"  [{i}] {cite}")
        lines += [
            f"\n⚠️  DISCLAIMER\n  {self.disclaimer}",
            "=" * 70,
        ]
        return "\n".join(lines)


class OncologistAgent:
    """
    Agent 3 of the Autonomous Oncology Board — the final synthesiser.

    Receives the outputs of Agents 1 & 2 and uses Llama 3.3 70B
    (running via Ollama on AMD MI300X ROCm) to produce the final
    Patient Management Plan.
    """

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.llm = llm_client or OllamaClient()
        log.info("OncologistAgent: initialised")

    def _build_prompt(
        self,
        pathology: PathologyReport,
        research: ResearchSummary,
    ) -> str:
        tissue = pathology.tissue_type.replace("_", " ").title()
        stage  = TISSUE_STAGE_MAP.get(pathology.tissue_type, "Unknown stage")
        avg_abnorm = sum(p.abnormality_score for p in pathology.patch_findings) / max(len(pathology.patch_findings), 1)

        research_brief = research.format_for_oncologist()

        return f"""You are chairing an Autonomous Oncology Board meeting.

=== PATHOLOGY REPORT (from AI Pathologist using Prov-GigaPath) ===
Case ID: {pathology.case_id}
Tissue Classification: {tissue}
Diagnostic Confidence: {pathology.confidence:.0%}
Patches Analysed: {pathology.n_patches}
Mean Abnormality Score: {avg_abnorm:.2f} (0=normal, 1=highly abnormal)
Summary: {pathology.summary}
Flags: {', '.join(pathology.flags) if pathology.flags else 'none'}

=== RESEARCH BRIEF (from AI Researcher using RAG + Oncology Literature) ===
{research_brief}

=== YOUR TASK ===
As the senior oncologist, synthesise the above into a complete Patient Management Plan.
Return a JSON object with exactly these fields:
{{
  "patient_summary": "2-3 sentence overview of this case",
  "diagnosis": {{
    "primary": "{tissue}",
    "tnm_stage": "estimated stage with caveats",
    "confidence": 0.85
  }},
  "immediate_actions": ["action 1", "action 2", "action 3"],
  "treatment_plan": {{
    "first_line": "specific regimen with doses if known",
    "rationale": "why this regimen based on evidence",
    "alternatives": ["alternative 1", "alternative 2"]
  }},
  "further_investigations": ["test 1", "test 2"],
  "multidisciplinary_referrals": ["referral 1", "referral 2"],
  "follow_up": "follow-up schedule",
  "confidence_score": 0.82,
  "board_consensus": "1-2 sentence board consensus statement",
  "citations": ["citation 1", "citation 2"]
}}

Return only the JSON. No markdown fences."""

    def _fallback_plan(
        self,
        pathology: PathologyReport,
        research: ResearchSummary,
    ) -> dict:
        """Rule-based fallback when LLM is unavailable."""
        tissue = pathology.tissue_type.replace("_", " ").title()
        stage  = TISSUE_STAGE_MAP.get(pathology.tissue_type, "Undetermined")
        tests  = research.recommended_tests[:4]
        opts   = research.treatment_options

        return {
            "patient_summary": (
                f"Case presents with histopathological features consistent with {tissue}. "
                f"{pathology.summary} Clinical staging and molecular profiling required."
            ),
            "diagnosis": {
                "primary": tissue,
                "tnm_stage": stage,
                "confidence": round(pathology.confidence, 2),
            },
            "immediate_actions": [
                "Obtain complete staging CT (chest/abdomen/pelvis)",
                "Order comprehensive molecular panel",
                "Multidisciplinary tumour board review",
                "Discuss case with patient and family",
            ],
            "treatment_plan": {
                "first_line": opts[0].regimen if opts else "Pending molecular results",
                "rationale": f"Based on {opts[0].evidence_level if opts else 'NCCN guidelines'}",
                "alternatives": [o.regimen for o in opts[1:3]],
            },
            "further_investigations": tests,
            "multidisciplinary_referrals": [
                "Medical oncology",
                "Radiation oncology",
                "Thoracic/colorectal surgery (as appropriate)",
                "Palliative care (symptom management)",
            ],
            "follow_up": "Response assessment CT at 8–12 weeks. Clinic review every 3 weeks during active treatment.",
            "confidence_score": round(pathology.confidence * 0.85, 2),
            "board_consensus": (
                f"Board consensus: {tissue} with {research.evidence_quality.lower()} quality evidence. "
                "Molecular testing required before initiating targeted therapy."
            ),
            "citations": research.citations[:5],
        }

    def synthesise(
        self,
        pathology_report: PathologyReport,
        research_summary: ResearchSummary,
    ) -> ManagementPlan:
        """
        Synthesise the final Patient Management Plan.

        Args:
            pathology_report: Output from PathologistAgent.
            research_summary: Output from ResearcherAgent.

        Returns:
            ManagementPlan — the final AOB output.
        """
        log.info(f"OncologistAgent: synthesising case '{pathology_report.case_id}'")

        prompt = self._build_prompt(pathology_report, research_summary)

        try:
            if self.llm.ping():
                response = self.llm.generate_sync(
                    prompt=prompt,
                    system=ONCOLOGIST_SYSTEM,
                    temperature=0.15,
                    max_tokens=1200,
                )
                text = response.text.strip()
                match = re.search(r'\{.*\}', text, re.DOTALL)
                plan_dict = json.loads(match.group() if match else text)
                log.info(f"OncologistAgent: LLM synthesis complete ({response.completion_tokens} tokens)")
            else:
                log.warning("OncologistAgent: LLM unavailable, using rule-based fallback")
                plan_dict = self._fallback_plan(pathology_report, research_summary)

        except Exception as e:
            log.warning(f"OncologistAgent: LLM failed ({e}), using rule-based fallback")
            plan_dict = self._fallback_plan(pathology_report, research_summary)

        # ── Build dataclass ──────────────────────────────────────────────────
        diag_raw = plan_dict.get("diagnosis", {})
        tx_raw   = plan_dict.get("treatment_plan", {})

        plan = ManagementPlan(
            case_id=pathology_report.case_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            patient_summary=plan_dict.get("patient_summary", ""),
            diagnosis=Diagnosis(
                primary=diag_raw.get("primary", pathology_report.tissue_type),
                tnm_stage=diag_raw.get("tnm_stage", "Unknown"),
                confidence=float(diag_raw.get("confidence", pathology_report.confidence)),
            ),
            immediate_actions=plan_dict.get("immediate_actions", []),
            treatment_plan=TreatmentPlan(
                first_line=tx_raw.get("first_line", ""),
                rationale=tx_raw.get("rationale", ""),
                alternatives=tx_raw.get("alternatives", []),
            ),
            further_investigations=plan_dict.get("further_investigations", []),
            multidisciplinary_referrals=plan_dict.get("multidisciplinary_referrals", []),
            follow_up=plan_dict.get("follow_up", ""),
            confidence_score=float(plan_dict.get("confidence_score", 0.75)),
            board_consensus=plan_dict.get("board_consensus", ""),
            disclaimer="This output is generated by an AI research tool and is NOT for clinical use. Always consult a qualified oncologist.",
            citations=plan_dict.get("citations", research_summary.citations),
        )

        log.info(
            f"OncologistAgent: plan complete — "
            f"confidence={plan.confidence_score:.0%}, "
            f"first_line='{plan.treatment_plan.first_line[:50]}...'"
        )
        return plan
