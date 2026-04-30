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

    def _synthesise_with_llm(
        self, report: PathologyReport, evidence: EvidenceBundle
    ) -> dict:
        """
        Use Llama 3.3 70B to synthesise evidence into structured JSON.
        Falls back to rule-based extraction if LLM is unavailable.
        """
        tissue = report.tissue_type.replace("_", " ").title()
        evidence_text = evidence.format_for_llm()

        prompt = f"""You are analysing a pathology case.

PATHOLOGY REPORT SUMMARY:
{report.summary}
Tissue type: {tissue}
Abnormality score: {sum(p.abnormality_score for p in report.patch_findings) / len(report.patch_findings):.2f}
Flags: {', '.join(report.flags) if report.flags else 'none'}

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

    # ── Main ──────────────────────────────────────────────────────────────────
    def research(
        self,
        pathology_report: PathologyReport,
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
        synthesis = self._synthesise_with_llm(pathology_report, evidence)

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

        summary = ResearchSummary(
            case_id=pathology_report.case_id,
            tissue_type=pathology_report.tissue_type,
            query=query,
            key_findings=synthesis.get("key_findings", []),
            recommended_tests=synthesis.get("recommended_tests", []),
            treatment_options=treatment_options,
            citations=synthesis.get("citations", []),
            evidence_quality=synthesis.get("evidence_quality", "Moderate"),
            raw_evidence=evidence.to_dict(),
        )

        log.info(
            f"ResearcherAgent: done — {len(summary.treatment_options)} treatment options, "
            f"quality={summary.evidence_quality}"
        )
        return summary
