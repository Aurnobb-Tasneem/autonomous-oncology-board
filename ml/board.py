"""
ml/board.py
===========
AOB Orchestration — Sequential State Machine.

Wires the three agents in a deterministic pipeline:
  Agent 1 (Pathologist) → Agent 2 (Researcher) → Agent 3 (Oncologist)

Per CLAUDE.md: "If CrewAI proves unstable on ROCm, fall back to a manual
Python state machine. Label it 'custom agentic framework' in the demo."

This IS that state machine. It's clean, debuggable, and works on ROCm.

Usage:
    board = AutonomousOncologyBoard()
    plan = board.run(case_id="case_001", images=[...])
    print(plan.format_report())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from ml.agents.pathologist import PathologistAgent, PathologyReport
from ml.agents.researcher import ResearcherAgent, ResearchSummary
from ml.agents.oncologist import OncologistAgent, ManagementPlan
from ml.models.llm_client import OllamaClient
from ml.rag.retriever import OncologyRetriever

log = logging.getLogger(__name__)


@dataclass
class BoardResult:
    """Complete output of one AOB run."""
    case_id: str
    pathology_report: PathologyReport
    research_summary: ResearchSummary
    management_plan: ManagementPlan
    total_time_s: float

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "total_time_s": self.total_time_s,
            "pathology_report": self.pathology_report.to_dict(),
            "research_summary": self.research_summary.to_dict(),
            "management_plan": self.management_plan.to_dict(),
        }


class AutonomousOncologyBoard:
    """
    The Autonomous Oncology Board — a three-agent sequential pipeline.

    Architecture (per CLAUDE.md §1):
      Pathologist  →  Researcher  →  Oncologist  →  ManagementPlan

    All agents share the same MI300X VRAM pool:
      - GigaPath (Pathologist) holds ~3 GB in the Docker container
      - Llama 3.3 70B (Researcher + Oncologist) holds ~40 GB via Ollama on HOST
      - Both visible in rocm-smi (192 GB unified HBM3)

    This design is physically impossible on a single NVIDIA H100 (80 GB VRAM).
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        ollama_host: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ):
        # Shared LLM client (Researcher + Oncologist both use it)
        self.llm = OllamaClient(
            host=ollama_host or None,
            model=ollama_model or None,
        )

        # Shared retriever
        self.retriever = OncologyRetriever()

        # Three agents
        self.pathologist = PathologistAgent(hf_token=hf_token)
        self.researcher  = ResearcherAgent(llm_client=self.llm, retriever=self.retriever)
        self.oncologist  = OncologistAgent(llm_client=self.llm)

        log.info("AutonomousOncologyBoard: all agents initialised")

    # ── Main pipeline ─────────────────────────────────────────────────────────
    def run(
        self,
        case_id: str,
        images: list[Image.Image],
        batch_size: int = 16,
    ) -> BoardResult:
        """
        Run the full AOB pipeline on a patient case.

        Args:
            case_id:    Unique case identifier.
            images:     List of histopathology patch images (PIL.Image).
            batch_size: GigaPath inference batch size.

        Returns:
            BoardResult containing all three agent outputs + final plan.
        """
        t0 = time.perf_counter()
        log.info(f"Board: starting case '{case_id}' — {len(images)} patches")

        # ── Agent 1: Pathologist ─────────────────────────────────────────────
        log.info("Board: [1/3] Pathologist agent running...")
        pathology = self.pathologist.analyse(case_id, images, batch_size=batch_size)
        log.info(f"Board: Pathologist done — {pathology.tissue_type} ({pathology.confidence:.0%})")

        # ── Agent 2: Researcher ──────────────────────────────────────────────
        log.info("Board: [2/3] Researcher agent running...")
        research = self.researcher.research(pathology)
        log.info(f"Board: Researcher done — {research.n_retrieved} docs, quality={research.evidence_quality}")

        # ── Agent 3: Oncologist ──────────────────────────────────────────────
        log.info("Board: [3/3] Oncologist agent running (Llama 3.3 70B)...")
        plan = self.oncologist.synthesise(pathology, research)
        log.info(f"Board: Oncologist done — confidence={plan.confidence_score:.0%}")

        total_time = round(time.perf_counter() - t0, 2)
        log.info(f"Board: ✅ case '{case_id}' complete in {total_time}s")

        return BoardResult(
            case_id=case_id,
            pathology_report=pathology,
            research_summary=research,
            management_plan=plan,
            total_time_s=total_time,
        )

    def run_from_paths(
        self,
        case_id: str,
        image_paths: list[Path | str],
        batch_size: int = 16,
    ) -> BoardResult:
        """Convenience wrapper — load images from file paths then run."""
        images = [Image.open(str(p)).convert("RGB") for p in image_paths]
        return self.run(case_id, images, batch_size=batch_size)
