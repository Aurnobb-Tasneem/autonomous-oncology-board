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
    # ── Digital Twin: 12-month PFS simulation ───────────────────────────
    pfs_12mo: float = 0.0
    pfs_curve: list[dict] = field(default_factory=list)
    pfs_model: str = ""
    pfs_assumptions: list[str] = field(default_factory=list)
    # ── Agent Debate fields (populated after debate phase) ────────────────
    debate_transcript: list[dict] = field(default_factory=list)
    revision_history: list[dict] = field(default_factory=list)
    revision_notes: str = ""
    consensus_score: int = 100   # 100 = no debate needed

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
        similar_cases: Optional[list[dict]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        tissue = pathology.tissue_type.replace("_", " ").title()
        stage  = TISSUE_STAGE_MAP.get(pathology.tissue_type, "Unknown stage")
        avg_abnorm = sum(p.abnormality_score for p in pathology.patch_findings) / max(len(pathology.patch_findings), 1)

        research_brief = research.format_for_oncologist()

        biomarker_lines = []
        for req in research.biomarker_requirements:
            biomarker = req.get("biomarker", "")
            status = req.get("status", "unknown")
            action = req.get("action", "")
            biomarker_lines.append(f"- {biomarker}: status={status}. {action}")

        gating_lines = []
        for gate in research.gated_treatments:
            regimen = gate.get("regimen", "")
            rule = gate.get("gate", "")
            gating_lines.append(f"- {regimen}: {rule}")

        biomarker_section = ""
        if biomarker_lines or gating_lines:
            parts = ["=== BIOMARKER GATING REQUIREMENTS ==="]
            if biomarker_lines:
                parts.append("BIOMARKER STATUS AND ACTIONS:")
                parts.extend(biomarker_lines)
            if gating_lines:
                parts.append("GATED TREATMENTS:")
                parts.extend(gating_lines)
            biomarker_section = "\n".join(parts) + "\n\n"

        # ── Similar cases block (Board Memory) ────────────────────────────────
        similar_section = ""
        if similar_cases:
            lines = ["=== SIMILAR PAST CASES FROM BOARD MEMORY ==="]
            for i, case in enumerate(similar_cases, 1):
                lines.append(
                    f"[{i}] Case '{case.get('case_id', 'unknown')}' "
                    f"({case.get('tissue_type', '?').replace('_', ' ').title()}, "
                    f"similarity {case.get('similarity', 0):.0%}) — "
                    f"First-line: {case.get('first_line_tx', '?')} | "
                    f"Summary: {case.get('plan_summary', '')[:120]}"
                )
            similar_section = "\n".join(lines) + "\n\n"

        metadata_lines = []
        if metadata:
            age = metadata.get("patient_age")
            sex = metadata.get("sex")
            notes = metadata.get("clinical_notes")
            if age:
                metadata_lines.append(f"- Age: {age}")
            if sex:
                metadata_lines.append(f"- Sex: {sex}")
            if notes:
                metadata_lines.append(f"- Clinical notes: {notes}")
            biomarker_status = metadata.get("biomarker_status")
            if isinstance(biomarker_status, dict) and biomarker_status:
                for key, value in biomarker_status.items():
                    metadata_lines.append(f"- Biomarker {key}: {value}")

        metadata_block = "\n".join(metadata_lines) if metadata_lines else "none"

        return f"""You are chairing an Autonomous Oncology Board meeting.

=== PATHOLOGY REPORT (from AI Pathologist using Prov-GigaPath) ===
Case ID: {pathology.case_id}
Tissue Classification: {tissue}
Diagnostic Confidence: {pathology.confidence:.0%}
Patches Analysed: {pathology.n_patches}
Mean Abnormality Score: {avg_abnorm:.2f} (0=normal, 1=highly abnormal)
Summary: {pathology.summary}
Flags: {', '.join(pathology.flags) if pathology.flags else 'none'}

=== PATIENT METADATA ===
{metadata_block}

=== RESEARCH BRIEF (from AI Researcher using RAG + Oncology Literature) ===
{research_brief}

{biomarker_section}{similar_section}=== YOUR TASK ===
As the senior oncologist, synthesise the above into a complete Patient Management Plan.
You MUST gate any biomarker-linked or targeted therapies behind biomarker status.
If status is unknown, mark the regimen as PENDING and add the required tests to
immediate actions and further investigations.
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
                "first_line": (opts[0].regimen + " (PENDING biomarker results)") if opts else "Pending molecular results",
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
        similar_cases: Optional[list[dict]] = None,
        metadata: Optional[dict] = None,
    ) -> ManagementPlan:
        """
        Synthesise the final Patient Management Plan.

        Args:
            pathology_report: Output from PathologistAgent.
            research_summary: Output from ResearcherAgent.
            similar_cases:    Optional list of similar past cases from BoardMemory.

        Returns:
            ManagementPlan — the final AOB output.
        """
        log.info(f"OncologistAgent: synthesising case '{pathology_report.case_id}'")

        prompt = self._build_prompt(
            pathology_report,
            research_summary,
            similar_cases=similar_cases,
            metadata=metadata,
        )

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

    # ── Agent Debate: Revise method ──────────────────────────────────────────
    def revise(
        self,
        current_plan: ManagementPlan,
        critique: dict,
        pathology_report: PathologyReport,
        research_summary: ResearchSummary,
        referee_update: Optional[dict] = None,
        round_num: int = 1,
    ) -> ManagementPlan:
        """
        Revise the management plan based on Researcher critique.

        Called during the Agent Debate loop after the Researcher issues
        a challenge. Produces a revised plan incorporating the critique.

        Args:
            current_plan:     The plan to be revised.
            critique:         Challenge dict from ResearcherAgent.challenge().
            pathology_report: Original pathology findings.
            research_summary: Original research brief.
            referee_update:   Optional Pathologist referee update.
            round_num:        Current debate round.

        Returns:
            Revised ManagementPlan with updated treatment and revision_notes.
        """
        log.info(f"OncologistAgent: revising plan (round {round_num})")

        challenge_text   = critique.get("challenge_text", "")
        flagged_issues   = critique.get("flagged_issues", [])
        recommendations  = critique.get("specific_recommendations", [])
        referee_note     = referee_update.get("referee_note", "") if referee_update else ""

        tissue = pathology_report.tissue_type.replace("_", " ").title()

        prompt = f"""You are a senior oncologist revising your management plan based on a
clinical challenge from your research colleague.

ORIGINAL PLAN:
  First-line treatment: {current_plan.treatment_plan.first_line}
  Immediate actions: {'; '.join(current_plan.immediate_actions[:4])}
  Further investigations: {'; '.join(current_plan.further_investigations[:4])}

RESEARCHER CHALLENGE (Round {round_num}):
  {challenge_text}
  Specific issues: {'; '.join(flagged_issues)}
  Recommendations: {'; '.join(recommendations)}
{f'PATHOLOGIST REFEREE UPDATE: {referee_note}' if referee_note else ''}

Revise your management plan to address these specific concerns.
Incorporate the recommended molecular tests and gate therapy on results.

Return a JSON object with exactly these fields:
{{
  "patient_summary": "Updated 2-3 sentence overview incorporating the challenge",
  "diagnosis": {{
    "primary": "{tissue}",
    "tnm_stage": "stage with updated caveats",
    "confidence": {current_plan.diagnosis.confidence}
  }},
  "immediate_actions": ["updated action 1", "updated action 2", "updated action 3"],
  "treatment_plan": {{
    "first_line": "updated regimen — gated on molecular results",
    "rationale": "revised rationale incorporating critique",
    "alternatives": ["alternative 1", "alternative 2"]
  }},
  "further_investigations": ["investigation 1", "investigation 2"],
  "multidisciplinary_referrals": ["referral 1", "referral 2"],
  "follow_up": "updated follow-up schedule",
  "confidence_score": 0.85,
  "board_consensus": "Revised consensus incorporating researcher challenge",
  "citations": ["citation 1", "citation 2"],
  "revision_notes": "✅ REVISED: One sentence describing what was changed and why"
}}

Return only the JSON."""

        try:
            if self.llm.ping():
                response = self.llm.generate_sync(
                    prompt=prompt,
                    system=ONCOLOGIST_SYSTEM,
                    temperature=0.15,
                    max_tokens=1000,
                )
                text = response.text.strip()
                match = re.search(r'\{.*\}', text, re.DOTALL)
                plan_dict = json.loads(match.group() if match else text)
            else:
                plan_dict = self._fallback_revision(current_plan, critique)
        except Exception as e:
            log.warning(f"OncologistAgent: revision LLM failed ({e}), using fallback")
            plan_dict = self._fallback_revision(current_plan, critique)

        revision_notes = plan_dict.pop("revision_notes", "✅ REVISED: Plan updated based on researcher challenge")

        # Build revised plan (carry over debate transcript from current plan)
        diag_raw = plan_dict.get("diagnosis", {})
        tx_raw   = plan_dict.get("treatment_plan", {})

        revised = ManagementPlan(
            case_id=current_plan.case_id,
            generated_at=current_plan.generated_at,
            patient_summary=plan_dict.get("patient_summary", current_plan.patient_summary),
            diagnosis=Diagnosis(
                primary=diag_raw.get("primary", current_plan.diagnosis.primary),
                tnm_stage=diag_raw.get("tnm_stage", current_plan.diagnosis.tnm_stage),
                confidence=float(diag_raw.get("confidence", current_plan.diagnosis.confidence)),
            ),
            immediate_actions=plan_dict.get("immediate_actions", current_plan.immediate_actions),
            treatment_plan=TreatmentPlan(
                first_line=tx_raw.get("first_line", current_plan.treatment_plan.first_line),
                rationale=tx_raw.get("rationale", current_plan.treatment_plan.rationale),
                alternatives=tx_raw.get("alternatives", current_plan.treatment_plan.alternatives),
            ),
            further_investigations=plan_dict.get("further_investigations", current_plan.further_investigations),
            multidisciplinary_referrals=plan_dict.get("multidisciplinary_referrals", current_plan.multidisciplinary_referrals),
            follow_up=plan_dict.get("follow_up", current_plan.follow_up),
            confidence_score=float(plan_dict.get("confidence_score", current_plan.confidence_score)),
            board_consensus=plan_dict.get("board_consensus", current_plan.board_consensus),
            disclaimer=current_plan.disclaimer,
            citations=plan_dict.get("citations", current_plan.citations),
            # Carry forward debate fields
            debate_transcript=current_plan.debate_transcript,
            revision_history=current_plan.revision_history,
            revision_notes=revision_notes,
            consensus_score=current_plan.consensus_score,
        )

        log.info(
            f"OncologistAgent: revision complete (round {round_num}) — "
            f"first_line='{revised.treatment_plan.first_line[:60]}'"
        )
        return revised

    def request_pathology_clarification(
        self,
        plan: ManagementPlan,
        pathology_report,   # ml.agents.pathologist.PathologyReport (avoid circular import)
    ) -> dict:
        """
        Generate specific morphological concerns when plan confidence is low.

        Called by board.run() as part of the cross-agent feedback loop:
          1. Oncologist detects low confidence in its own management plan.
          2. This method produces a structured critique of what morphological
             details need clarification from the Pathologist.
          3. board.run() passes those concerns to PathologistAgent.referee().

        The result is a named pipeline stage visible in the SSE timeline —
        the "backward gradient" from Oncologist → Pathologist.

        Args:
            plan:             ManagementPlan with a low confidence_score.
            pathology_report: PathologyReport from Agent 1.

        Returns:
            dict with keys:
              critique_text              (str)  — full explanation of uncertainties
              specific_concerns          (list[str]) — 2–4 morphological questions
              confidence_threshold_missed (float) — the actual confidence score
              triggered_by               (str)  — always "low_oncologist_confidence"
        """
        tissue = pathology_report.tissue_type.replace("_", " ").title()
        flags_str = (
            ", ".join(pathology_report.flags)
            if pathology_report.flags else "none"
        )
        confidence_pct = f"{plan.confidence_score:.0%}"

        prompt = f"""You are a senior consultant oncologist who has just produced a management plan
for a patient but has low diagnostic confidence ({confidence_pct}).

PATHOLOGY FINDINGS:
  Tissue classification : {tissue}
  GigaPath confidence   : {pathology_report.confidence:.0%}
  Pathology flags       : {flags_str}
  Pathology summary     : {pathology_report.summary[:300]}

MANAGEMENT PLAN CONFIDENCE: {confidence_pct}
DIAGNOSIS: {plan.diagnosis.primary}

You need to request targeted re-evaluation from the digital pathologist.
Identify 2–4 specific morphological features that, if confirmed or denied,
would most change or strengthen this diagnosis and treatment plan.

Return ONLY valid JSON with exactly these fields:
{{
  "critique_text": "The management plan confidence is low because...",
  "specific_concerns": [
    "Confirm presence of glandular structures consistent with adenocarcinoma",
    "Assess nuclear grade and mitotic index in high-abnormality patches"
  ]
}}

Be specific, clinically precise, and actionable. 2–4 concerns maximum."""

        try:
            if not self.llm.ping():
                log.warning(
                    "OncologistAgent.request_pathology_clarification: LLM unavailable "
                    "— using flag-based heuristic"
                )
                return self._fallback_clarification(plan, pathology_report)

            response = self.llm.generate_sync(
                prompt=prompt,
                system=(
                    "You are a senior oncologist requesting targeted pathology clarification. "
                    "Output valid JSON only — no markdown fences, no preamble."
                ),
                temperature=0.05,
                max_tokens=350,
            )
            text = response.text.strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            result = json.loads(match.group() if match else text)

            concerns = result.get("specific_concerns", [])
            log.info(
                f"OncologistAgent: pathology clarification requested — "
                f"{len(concerns)} concerns generated (confidence={confidence_pct})"
            )
            return {
                "critique_text":               result.get("critique_text", ""),
                "specific_concerns":           concerns[:4],
                "confidence_threshold_missed": plan.confidence_score,
                "triggered_by":                "low_oncologist_confidence",
            }

        except Exception as e:
            log.warning(
                f"OncologistAgent.request_pathology_clarification: failed ({e}) "
                "— using fallback"
            )
            return self._fallback_clarification(plan, pathology_report)

    def _fallback_clarification(self, plan: ManagementPlan, pathology_report) -> dict:
        """
        Heuristic clarification when the LLM is unavailable.

        Derives concerns from the existing pathology flags and tissue type
        so the backward feedback path always has something to work with.
        """
        tissue = pathology_report.tissue_type.replace("_", " ").title()
        concerns: list[str] = []

        if "high_abnormality_detected" in pathology_report.flags:
            concerns.append(
                f"Confirm nuclear atypia grade and mitotic figures in "
                f"high-abnormality patches ({tissue})"
            )
        if "heterogeneous_tissue" in pathology_report.flags:
            concerns.append(
                "Clarify dominant tissue subtype — patch-level heterogeneity "
                "may indicate mixed histology or sampling artefact"
            )
        if "high_diagnostic_uncertainty" in pathology_report.flags:
            concerns.append(
                "Re-assess classification confidence — MC Dropout flagged "
                "high uncertainty; consider IHC panel for confirmation"
            )

        # Always include a tissue-specific generic concern
        concerns.append(
            f"Verify glandular / architectural patterns consistent with "
            f"{tissue} vs differential diagnoses"
        )

        critique_text = (
            f"Management plan confidence is {plan.confidence_score:.0%} — below "
            f"the board's acceptance threshold.  The following morphological "
            f"features require targeted re-evaluation before finalising treatment."
        )

        return {
            "critique_text":               critique_text,
            "specific_concerns":           concerns[:4],
            "confidence_threshold_missed": plan.confidence_score,
            "triggered_by":                "low_oncologist_confidence",
        }

    def _fallback_revision(self, current_plan: ManagementPlan, critique: dict) -> dict:
        """Rule-based revision fallback when LLM unavailable."""
        recs = critique.get("specific_recommendations", [])
        updated_actions = list(current_plan.immediate_actions)
        updated_investigations = list(current_plan.further_investigations)

        # Add recommended tests to investigations
        for rec in recs:
            if rec not in updated_investigations:
                updated_investigations.insert(0, rec)

        return {
            "patient_summary": current_plan.patient_summary,
            "diagnosis": {
                "primary": current_plan.diagnosis.primary,
                "tnm_stage": current_plan.diagnosis.tnm_stage,
                "confidence": current_plan.diagnosis.confidence,
            },
            "immediate_actions": updated_actions,
            "treatment_plan": {
                "first_line": current_plan.treatment_plan.first_line + " (PENDING molecular results)",
                "rationale": current_plan.treatment_plan.rationale,
                "alternatives": current_plan.treatment_plan.alternatives,
            },
            "further_investigations": updated_investigations,
            "multidisciplinary_referrals": current_plan.multidisciplinary_referrals,
            "follow_up": current_plan.follow_up,
            "confidence_score": current_plan.confidence_score,
            "board_consensus": current_plan.board_consensus + " Revised per researcher challenge.",
            "citations": current_plan.citations,
            "revision_notes": "✅ REVISED: Added molecular testing requirements before targeted therapy.",
        }
