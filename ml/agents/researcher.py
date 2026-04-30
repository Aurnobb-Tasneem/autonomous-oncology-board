"""
ml/agents/researcher.py
=======================
Agent 2: Researcher — Evidence synthesis via RAG + Llama 3.3 70B.

Responsibilities:
  - Receive PathologyReport from Pathologist agent
  - Build a targeted query from tissue type + clinical context
  - Retrieve top-K evidence documents from OncologyRetriever
  - Use Llama 3.3 70B (via Ollama) to synthesise evidence into
    a structured clinical evidence bundle with citations
  - Return ResearchSummary JSON

Output schema (ResearchSummary):
  {
    "case_id": str,
    "tissue_type": str,
    "query": str,
    "key_findings": [str],        # bullet points of synthesised evidence
    "recommended_tests": [str],   # molecular/genomic tests to order
    "treatment_options": [        # ranked treatment protocols
      {
        "line": str,              # "First-line", "Second-line"
        "regimen": str,
        "evidence_level": str,    # "NCCN Category 1", "Phase III"
        "citation": str
      }
    ],
    "citations": [str],           # full reference list
    "evidence_quality": str,      # "High" / "Moderate" / "Low"
    "raw_evidence": dict          # the full EvidenceBundle
  }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from ml.models.llm_client import OllamaClient
from ml.rag.retriever import OncologyRetriever, EvidenceBundle
from ml.agents.pathologist import PathologyReport

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical oncology researcher with expertise in evidence-based medicine.
Your role is to synthesise retrieved oncology literature into a structured clinical evidence summary.
Always cite your sources. Be precise, factual, and concise.
Output valid JSON only — no markdown fences, no commentary outside the JSON."""


@dataclass
class TreatmentOption:
    line: str            # "First-line", "Second-line", "Adjuvant"
    regimen: str
    evidence_level: str  # "NCCN Category 1", "Phase III RCT", etc.
    citation: str


@dataclass
class ResearchSummary:
    case_id: str
    tissue_type: str
    query: str
    key_findings: list[str]
    recommended_tests: list[str]
    treatment_options: list[TreatmentOption]
    biomarker_requirements: list[dict]
    gated_treatments: list[dict]
    citations: list[str]
    evidence_quality: str
    raw_evidence: dict

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def format_for_oncologist(self) -> str:
        """Format as a readable brief for the Oncologist agent."""
        lines = [
            f"RESEARCH BRIEF — {self.tissue_type.replace('_', ' ').title()}",
            f"Evidence Quality: {self.evidence_quality}",
            "=" * 60,
            "\nKEY FINDINGS:",
        ]
        for finding in self.key_findings:
            lines.append(f"  • {finding}")

        lines.append("\nRECOMMENDED MOLECULAR TESTS:")
        for test in self.recommended_tests:
            lines.append(f"  • {test}")

        if self.biomarker_requirements:
            lines.append("\nBIOMARKER REQUIREMENTS (GATING):")
            for req in self.biomarker_requirements:
                biomarker = req.get("biomarker", "")
                status = req.get("status", "unknown")
                action = req.get("action", "")
                lines.append(f"  • {biomarker} — status: {status}. {action}")

        if self.gated_treatments:
            lines.append("\nGATED TREATMENTS:")
            for gate in self.gated_treatments:
                regimen = gate.get("regimen", "")
                rule = gate.get("gate", "")
                lines.append(f"  • {regimen} — {rule}")

        lines.append("\nTREATMENT OPTIONS:")
        for opt in self.treatment_options:
            lines.append(
                f"  [{opt.line}] {opt.regimen}\n"
                f"    Evidence: {opt.evidence_level} | {opt.citation}"
            )

        lines.append("\nCITATIONS:")
        for i, cite in enumerate(self.citations, 1):
            lines.append(f"  [{i}] {cite}")

        return "\n".join(lines)


class ResearcherAgent:
    """
    Agent 2 of the Autonomous Oncology Board.

    Takes the PathologyReport from Agent 1, retrieves relevant oncology
    evidence, and uses Llama 3.3 70B to synthesise it into a structured
    clinical research brief for Agent 3 (Oncologist).
    """

    def __init__(
        self,
        llm_client: Optional[OllamaClient] = None,
        retriever: Optional[OncologyRetriever] = None,
        top_k: int = 5,
    ):
        self.llm     = llm_client or OllamaClient()
        self.retriever = retriever or OncologyRetriever()
        self.top_k   = top_k
        log.info("ResearcherAgent: initialised")

    def _build_query(self, report: PathologyReport) -> str:
        """Build a targeted clinical query from the pathology report."""
        tissue = report.tissue_type.replace("_", " ")
        flags  = ", ".join(report.flags) if report.flags else "none"
        return (
            f"{tissue} diagnosis: treatment protocols, molecular testing requirements, "
            f"staging guidelines, prognosis. Flags: {flags}."
        )

    def _normalise_biomarker_status(self, status: str) -> str:
        """Normalise biomarker status values to positive/negative/unknown."""
        if not status:
            return "unknown"
        val = str(status).strip().lower()
        if val in {"pos", "positive", "mutant", "mutated"}:
            return "positive"
        if val in {"neg", "negative", "wildtype", "wt", "wild-type"}:
            return "negative"
        return "unknown"

    def _extract_biomarker_status(self, metadata: Optional[dict]) -> dict[str, str]:
        """Extract biomarker status map from metadata."""
        if not metadata:
            return {}

        status_map: dict[str, str] = {}

        # Preferred: metadata["biomarker_status"] = {"EGFR": "positive", ...}
        raw = metadata.get("biomarker_status") if isinstance(metadata, dict) else None
        if isinstance(raw, dict):
            for key, value in raw.items():
                if key:
                    status_map[str(key).upper()] = self._normalise_biomarker_status(value)

        # Fallbacks: egfr_status / alk_status
        for key in ("egfr_status", "alk_status"):
            if key in metadata:
                status_map[key.split("_")[0].upper()] = self._normalise_biomarker_status(metadata.get(key))

        return status_map

    def _apply_biomarker_statuses(self, requirements: list[dict], status_map: dict[str, str]) -> list[dict]:
        """Apply known biomarker statuses to the requirements list."""
        if not requirements or not status_map:
            return requirements

        for req in requirements:
            biomarker = str(req.get("biomarker", "")).strip()
            if not biomarker:
                continue
            key = biomarker.replace(" ", "").replace("-", "").upper()
            status = status_map.get(key)
            if status:
                req["status"] = status
                if status != "unknown":
                    req["action"] = "Status provided in metadata; gate therapy accordingly."

        return requirements

    def _synthesise_with_llm(
        self,
        report: PathologyReport,
        evidence: EvidenceBundle,
        biomarker_status: Optional[dict[str, str]] = None,
    ) -> dict:
        """
        Use Llama 3.3 70B to synthesise evidence into structured JSON.
        Falls back to rule-based extraction if LLM is unavailable.
        """
        tissue = report.tissue_type.replace("_", " ").title()
        evidence_text = evidence.format_for_llm()

        status_lines = []
        if biomarker_status:
            for key, value in biomarker_status.items():
            status_lines.append(f"- {key}: {value}")

        biomarker_status_block = "\n".join(status_lines) if status_lines else "none"

        prompt = f"""You are analysing a pathology case.

PATHOLOGY REPORT SUMMARY:
{report.summary}
Tissue type: {tissue}
Abnormality score: {sum(p.abnormality_score for p in report.patch_findings) / len(report.patch_findings):.2f}
Flags: {', '.join(report.flags) if report.flags else 'none'}

    KNOWN BIOMARKER STATUS (from metadata):
    {biomarker_status_block}

RETRIEVED EVIDENCE:
{evidence_text}

Based on the above, produce a JSON object with exactly these fields:
{{
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "recommended_tests": ["test 1", "test 2"],
  "treatment_options": [
    {{"line": "First-line", "regimen": "...", "evidence_level": "NCCN Category 1", "citation": "..."}},
    {{"line": "Second-line", "regimen": "...", "evidence_level": "Phase III", "citation": "..."}}
  ],
    "biomarker_requirements": [
        {{"biomarker": "EGFR", "status": "unknown", "action": "Order EGFR testing before EGFR TKI."}}
    ],
    "gated_treatments": [
        {{"regimen": "Osimertinib", "gate": "ONLY if EGFR-mutant; otherwise do not use."}}
    ],
  "citations": ["citation 1", "citation 2"],
  "evidence_quality": "High"
}}

Return only the JSON object. No markdown, no explanation."""

        try:
            if not self.llm.ping():
                log.warning("ResearcherAgent: LLM unavailable, using rule-based fallback")
                return self._rule_based_synthesis(evidence)

            response = self.llm.generate_sync(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=800,
            )
            # Parse JSON from response
            text = response.text.strip()
            # Extract JSON if wrapped in anything
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(text)

        except Exception as e:
            log.warning(f"ResearcherAgent: LLM synthesis failed ({e}), using fallback")
            return self._rule_based_synthesis(evidence)

    def _rule_based_synthesis(self, evidence: EvidenceBundle) -> dict:
        """
        Rule-based fallback when LLM is unavailable.
        Extracts treatment options directly from mock corpus.
        """
        docs = evidence.documents
        return {
            "key_findings": [
                d.content[:120] + "..." for d in docs[:3]
            ],
            "recommended_tests": self._infer_tests(evidence.tissue_type),
            "treatment_options": [
                {
                    "line": "First-line",
                    "regimen": docs[0].title if docs else "Standard chemotherapy",
                    "evidence_level": docs[0].source if docs else "NCCN",
                    "citation": docs[0].citation if docs else "",
                }
            ],
            "biomarker_requirements": self._default_biomarker_requirements(evidence.tissue_type),
            "gated_treatments": self._default_gated_treatments(evidence.tissue_type, [docs[0].title] if docs else []),
            "citations": [d.citation for d in docs if d.citation],
            "evidence_quality": "High" if len(docs) >= 3 else "Moderate",
        }

    def _infer_tests(self, tissue_type: str) -> list[str]:
        """Tissue-type specific molecular tests."""
        tests = {
            "lung_adenocarcinoma":         ["EGFR mutation", "ALK FISH", "ROS1", "BRAF V600E", "PD-L1 TPS", "KRAS G12C", "MET exon 14", "TMB"],
            "lung_squamous_cell_carcinoma": ["PD-L1 TPS", "TMB", "FGFR1 amplification", "EGFR (rare)"],
            "colon_adenocarcinoma":        ["MSI/MMR", "KRAS/NRAS/BRAF", "HER2", "NTRK fusion", "CEA"],
            "colon_benign_tissue":         ["Colonoscopy follow-up", "FIT annually"],
            "lung_benign_tissue":          ["CT follow-up per Fleischner", "PET if >8mm"],
        }
        return tests.get(tissue_type, ["Histopathology review", "IHC panel"])

    def _default_biomarker_requirements(self, tissue_type: str) -> list[dict]:
        """Default biomarker gating requirements when LLM is unavailable."""
        if tissue_type == "lung_adenocarcinoma":
            biomarkers = ["EGFR", "ALK", "ROS1", "BRAF", "KRAS", "MET", "RET", "NTRK", "PD-L1"]
        elif tissue_type == "lung_squamous_cell_carcinoma":
            biomarkers = ["PD-L1", "FGFR1", "EGFR (rare)"]
        elif tissue_type == "colon_adenocarcinoma":
            biomarkers = ["MSI/MMR", "KRAS/NRAS", "BRAF", "HER2", "NTRK"]
        else:
            biomarkers = ["IHC panel"]

        return [
            {
                "biomarker": b,
                "status": "unknown",
                "action": f"Order {b} testing before biomarker-linked therapy.",
            }
            for b in biomarkers
        ]

    def _default_gated_treatments(self, tissue_type: str, regimens: list[str]) -> list[dict]:
        """Infer simple treatment gating rules from regimen names."""
        rules = []
        for regimen in regimens:
            low = regimen.lower()
            if "osimertinib" in low:
                rules.append({"regimen": regimen, "gate": "ONLY if EGFR-mutant (exon 19 del or L858R)."})
            elif "alectinib" in low:
                rules.append({"regimen": regimen, "gate": "ONLY if ALK-positive."})
            elif "crizotinib" in low:
                rules.append({"regimen": regimen, "gate": "ONLY if ALK- or ROS1-positive."})
            elif "pembrolizumab" in low and "colon" in tissue_type:
                rules.append({"regimen": regimen, "gate": "ONLY if MSI-H/dMMR."})
            elif "pembrolizumab" in low and "lung" in tissue_type:
                rules.append({"regimen": regimen, "gate": "ONLY if PD-L1 high (>=50%) or per NCCN."})
            elif "cetuximab" in low or "panitumumab" in low:
                rules.append({"regimen": regimen, "gate": "ONLY if KRAS/NRAS wild-type."})
        return rules

    # ── Main ──────────────────────────────────────────────────────────────────
    def research(
        self,
        pathology_report: PathologyReport,
        metadata: Optional[dict] = None,
    ) -> ResearchSummary:
        """
        Run evidence retrieval and synthesis for a pathology case.

        Args:
            pathology_report: Output from PathologistAgent.

        Returns:
            ResearchSummary with citations and treatment options.
        """
        log.info(
            f"ResearcherAgent: researching case '{pathology_report.case_id}' "
            f"({pathology_report.tissue_type})"
        )

        # ── Step 1: Build query ──────────────────────────────────────────────
        query = self._build_query(pathology_report)

        # ── Step 2: Retrieve evidence ────────────────────────────────────────
        evidence = self.retriever.retrieve(
            query=query,
            tissue_type=pathology_report.tissue_type,
            top_k=self.top_k,
        )
        log.info(f"ResearcherAgent: retrieved {evidence.n_retrieved} documents")

        # ── Step 3: Synthesise with LLM ──────────────────────────────────────
        biomarker_status = self._extract_biomarker_status(metadata)
        synthesis = self._synthesise_with_llm(pathology_report, evidence, biomarker_status=biomarker_status)

        if not synthesis.get("biomarker_requirements"):
            synthesis["biomarker_requirements"] = self._default_biomarker_requirements(pathology_report.tissue_type)

        if not synthesis.get("gated_treatments"):
            regimen_list = [t.get("regimen", "") for t in synthesis.get("treatment_options", []) if t.get("regimen")]
            synthesis["gated_treatments"] = self._default_gated_treatments(pathology_report.tissue_type, regimen_list)

        # ── Step 4: Build ResearchSummary ────────────────────────────────────
        treatment_options = [
            TreatmentOption(
                line=t.get("line", ""),
                regimen=t.get("regimen", ""),
                evidence_level=t.get("evidence_level", ""),
                citation=t.get("citation", ""),
            )
            for t in synthesis.get("treatment_options", [])
        ]

        synthesis["biomarker_requirements"] = self._apply_biomarker_statuses(
            synthesis.get("biomarker_requirements", []),
            biomarker_status,
        )

        summary = ResearchSummary(
            case_id=pathology_report.case_id,
            tissue_type=pathology_report.tissue_type,
            query=query,
            key_findings=synthesis.get("key_findings", []),
            recommended_tests=synthesis.get("recommended_tests", []),
            treatment_options=treatment_options,
            biomarker_requirements=synthesis.get("biomarker_requirements", self._default_biomarker_requirements(pathology_report.tissue_type)),
            gated_treatments=synthesis.get("gated_treatments", []),
            citations=synthesis.get("citations", []),
            evidence_quality=synthesis.get("evidence_quality", "Moderate"),
            raw_evidence=evidence.to_dict(),
        )

        log.info(
            f"ResearcherAgent: done — {len(summary.treatment_options)} treatment options, "
            f"quality={summary.evidence_quality}"
        )
        return summary

    # ── Agent Debate: Challenge method ────────────────────────────────────────
    def challenge(
        self,
        draft_plan_dict: dict,
        pathology_report: PathologyReport,
        research_summary: ResearchSummary,
        round_num: int = 1,
    ) -> dict:
        """
        Challenge the Oncologist's draft plan using RAG evidence.

        Called during the Agent Debate loop. Searches for guideline
        contradictions, missing molecular tests, or evidence gaps in
        the proposed management plan.

        Args:
            draft_plan_dict:   The Oncologist's current plan as a dict.
            pathology_report:  Original pathology findings.
            research_summary:  Original research brief.
            round_num:         Current debate round number.

        Returns:
            dict with keys: challenge_text (str), flagged_issues (list),
            morphological_doubts (bool), specific_recommendations (list)
        """
        log.info(f"ResearcherAgent: challenging draft plan (round {round_num})")

        tissue = pathology_report.tissue_type.replace("_", " ").title()
        first_line = draft_plan_dict.get("treatment_plan", {}).get("first_line", "")
        investigations = draft_plan_dict.get("further_investigations", [])
        actions = draft_plan_dict.get("immediate_actions", [])

        # Retrieve specific biomarker/guideline evidence for the challenge
        challenge_query = (
            f"{tissue} NCCN guidelines molecular testing requirements biomarker "
            f"EGFR ALK ROS1 PD-L1 before targeted therapy contraindications"
        )
        evidence = self.retriever.retrieve(
            query=challenge_query,
            tissue_type=pathology_report.tissue_type,
            top_k=3,
        )
        evidence_text = evidence.format_for_llm() if hasattr(evidence, 'format_for_llm') else str(evidence)

        prompt = f"""You are a clinical oncology researcher reviewing a draft management plan
for a patient with {tissue}.

DRAFT MANAGEMENT PLAN:
  First-line treatment: {first_line}
  Immediate actions: {'; '.join(actions[:4])}
  Investigations ordered: {'; '.join(investigations[:4])}

EVIDENCE FROM ONCOLOGY CORPUS:
{evidence_text}

KEY GUIDELINES TO CHECK AGAINST:
- NCCN NSCLC: EGFR/ALK/ROS1/BRAF/KRAS/MET/RET/NTRK must be tested before TKI
- NCCN: PD-L1 ≥50% required for pembrolizumab monotherapy without chemo
- Osimertinib: ONLY for EGFR-mutant (exon 19 del or L858R)
- Alectinib: ONLY for ALK-positive NSCLC
- For colon: MSI-H required for pembrolizumab; KRAS/NRAS WT for anti-EGFR

Round {round_num} challenge: Identify any clinical concerns, missing tests,
or guideline violations in the draft plan.

Return a JSON object with exactly these fields:
{{
  "challenge_text": "⚠️ CHALLENGE: One clear sentence summarising the main concern",
  "flagged_issues": ["Specific issue 1", "Specific issue 2"],
  "morphological_doubts": false,
  "specific_recommendations": ["Add EGFR testing before osimertinib", "Gate TKI on molecular results"],
  "severity": "high"
}}

Return only the JSON. Be specific and cite the NCCN guideline violated."""

        try:
            if not self.llm.ping():
                return self._heuristic_challenge(tissue, first_line, investigations)

            response = self.llm.generate_sync(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=500,
            )
            text = response.text.strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            result = json.loads(match.group() if match else text)
            log.info(f"ResearcherAgent: challenge issued — '{result.get('challenge_text', '')[:80]}'")
            return result

        except Exception as e:
            log.warning(f"ResearcherAgent: challenge LLM failed ({e}), using heuristic")
            return self._heuristic_challenge(tissue, first_line, investigations)

    def _heuristic_challenge(
        self,
        tissue: str,
        first_line: str,
        investigations: list[str],
    ) -> dict:
        """Rule-based challenge when LLM is unavailable."""
        # Check for common missing items
        missing = []
        if "lung" in tissue.lower():
            mol_tests = ["EGFR", "ALK", "ROS1", "PD-L1"]
            missing = [t for t in mol_tests if not any(t in inv for inv in investigations)]
        elif "colon" in tissue.lower():
            mol_tests = ["MSI", "KRAS", "BRAF"]
            missing = [t for t in mol_tests if not any(t in inv for inv in investigations)]

        if missing:
            return {
                "challenge_text": f"⚠️ CHALLENGE: Molecular testing ({', '.join(missing)}) required before targeted therapy per NCCN guidelines.",
                "flagged_issues": [f"Missing {t} testing" for t in missing],
                "morphological_doubts": False,
                "specific_recommendations": [f"Order {t} testing before initiating {first_line}" for t in missing[:2]],
                "severity": "high",
            }
        return {
            "challenge_text": "✅ No major guideline violations detected in draft plan.",
            "flagged_issues": [],
            "morphological_doubts": False,
            "specific_recommendations": [],
            "severity": "low",
        }
