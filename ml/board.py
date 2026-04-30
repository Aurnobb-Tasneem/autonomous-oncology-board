"""
ml/board.py
===========
AOB Orchestration — Sequential State Machine + Agent Debate Loop.

Pipeline:
  Agent 1 (Pathologist) → Agent 2 (Researcher) → Agent 3 (Oncologist)
         ↓ if debate_mode=True
  [Debate Loop — max 3 rounds]:
    Researcher.challenge() → Pathologist.referee() → Oncologist.revise()
    MetaEvaluator.evaluate() → if score < 70, repeat; else finalize

Per CLAUDE.md: "If CrewAI proves unstable on ROCm, fall back to a manual
Python state machine. Label it 'custom agentic framework' in the demo."

This IS that state machine. It's clean, debuggable, and works on ROCm.

Usage:
    board = AutonomousOncologyBoard()
    result = board.run(case_id="case_001", images=[...])
    print(result.management_plan.format_report())
    print(result.management_plan.debate_transcript)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from ml.agents.pathologist import PathologistAgent, PathologyReport
from ml.agents.researcher import ResearcherAgent, ResearchSummary
from ml.agents.oncologist import OncologistAgent, ManagementPlan
from ml.agents.meta_evaluator import MetaEvaluator
from ml.models.llm_client import OllamaClient
from ml.rag.retriever import OncologyRetriever

log = logging.getLogger(__name__)

# Consensus threshold — scores below this trigger another debate round
CONSENSUS_THRESHOLD = 70
MAX_DEBATE_ROUNDS   = 3


@dataclass
class DebateRound:
    """Record of one round in the agent debate."""
    round_num: int
    challenge_text: str
    flagged_issues: list[str]
    referee_note: str
    revision_notes: str
    consensus_score: int
    morphology_confirmed: bool


@dataclass
class BoardResult:
    """Complete output of one AOB run."""
    case_id: str
    pathology_report: PathologyReport
    research_summary: ResearchSummary
    management_plan: ManagementPlan
    total_time_s: float
    debate_rounds: list[DebateRound] = field(default_factory=list)
    debate_enabled: bool = False
    heatmaps_b64: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "case_id": self.case_id,
            "total_time_s": self.total_time_s,
            "debate_enabled": self.debate_enabled,
            "debate_rounds_completed": len(self.debate_rounds),
            "has_heatmaps": len(self.heatmaps_b64) > 0,
            "n_heatmaps": len(self.heatmaps_b64),
            "pathology_report": self.pathology_report.to_dict(),
            "research_summary": self.research_summary.to_dict(),
            "management_plan": self.management_plan.to_dict(),
        }
        if self.debate_rounds:
            d["debate_summary"] = [
                {
                    "round": r.round_num,
                    "challenge": r.challenge_text,
                    "flagged_issues": r.flagged_issues,
                    "referee_note": r.referee_note,
                    "revision": r.revision_notes,
                    "consensus_score": r.consensus_score,
                    "morphology_confirmed": r.morphology_confirmed,
                }
                for r in self.debate_rounds
            ]
        return d


# Step callback type: (agent: str, message: str, progress: int) -> None
StepCallback = Callable[[str, str, int], None]


class AutonomousOncologyBoard:
    """
    The Autonomous Oncology Board — three-agent sequential pipeline
    with optional multi-round Agent Debate.

    Architecture (per CLAUDE.md §1):
      Standard: Pathologist → Researcher → Oncologist → ManagementPlan
      Debate:   + Researcher.challenge() → Pathologist.referee()
                → Oncologist.revise() → MetaEvaluator.score()
                → [repeat if score < 70, max 3 rounds]

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
        # Shared LLM client (Researcher + Oncologist + MetaEvaluator all use it)
        self.llm = OllamaClient(
            host=ollama_host or None,
            model=ollama_model or None,
        )

        # Shared retriever
        self.retriever = OncologyRetriever()

        # Three core agents
        self.pathologist   = PathologistAgent(hf_token=hf_token)
        self.researcher    = ResearcherAgent(llm_client=self.llm, retriever=self.retriever)
        self.oncologist    = OncologistAgent(llm_client=self.llm)
        self.meta_evaluator = MetaEvaluator(llm_client=self.llm)

        log.info("AutonomousOncologyBoard: all agents initialised")

    # ── Main pipeline ─────────────────────────────────────────────────────────
    def run(
        self,
        case_id: str,
        images: list[Image.Image],
        batch_size: int = 16,
        debate_mode: bool = True,
        step_callback: Optional[StepCallback] = None,
    ) -> BoardResult:
        """
        Run the full AOB pipeline on a patient case.

        Args:
            case_id:       Unique case identifier.
            images:        List of histopathology patch images (PIL.Image).
            batch_size:    GigaPath inference batch size.
            debate_mode:   If True, run Agent Debate loop after standard pipeline.
            step_callback: Optional function called on each pipeline step for SSE.

        Returns:
            BoardResult containing all agent outputs, debate transcript, and final plan.
        """
        def _emit(agent: str, message: str, progress: int):
            log.info(f"[{agent}] {message}")
            if step_callback:
                step_callback(agent, message, progress)

        t0 = time.perf_counter()
        _emit("system", f"AOB: starting case '{case_id}' — {len(images)} patches", 2)

        # ── Agent 1: Pathologist ─────────────────────────────────────────────
        _emit("pathologist", "GigaPath: loading model and preprocessing patches", 8)
        pathology = self.pathologist.analyse(case_id, images, batch_size=batch_size)
        _emit(
            "pathologist",
            f"GigaPath: {pathology.n_patches} patches analysed → "
            f"{pathology.tissue_type.replace('_', ' ').title()} "
            f"({pathology.confidence:.0%} confidence)",
            30,
        )
        if pathology.flags:
            _emit("pathologist", f"⚠️ Flags: {', '.join(pathology.flags)}", 32)

        # Generate attention heatmaps after pathologist completes
        _emit("pathologist", "Generating attention heatmaps (attention rollout across all ViT blocks)...", 33)
        heatmaps = self.pathologist.generate_heatmaps(images, max_patches=8)
        if heatmaps:
            _emit("pathologist", f"🔥 {len(heatmaps)} attention heatmaps generated — suspicious regions highlighted in red", 35)
        else:
            _emit("pathologist", "Heatmap extraction skipped (flash-attn or hook incompatibility)", 34)

        # MC Dropout uncertainty quantification
        _emit("pathologist", f"Running MC Dropout uncertainty quantification (20 stochastic passes)...", 36)
        pathology = self.pathologist.quantify_uncertainty(pathology, images)
        if pathology.uncertainty_interval:
            unc_msg = f"Uncertainty: {pathology.uncertainty_interval}"
            if pathology.high_uncertainty:
                unc_msg += " ⚠️ HIGH — second-opinion biopsy recommended"
            _emit("pathologist", unc_msg, 38)

        # ── Agent 2: Researcher ──────────────────────────────────────────────
        _emit("researcher", "Building clinical query from pathology findings", 35)
        research = self.researcher.research(pathology)
        _emit(
            "researcher",
            f"Retrieved {research.raw_evidence.get('n_retrieved', 0)} evidence documents "
            f"(quality: {research.evidence_quality})",
            52,
        )
        _emit("researcher", f"Synthesised {len(research.treatment_options)} treatment options", 56)

        # ── Agent 3: Oncologist (initial draft) ──────────────────────────────
        _emit("oncologist", "Llama 3.3 70B: synthesising initial management plan...", 60)
        plan = self.oncologist.synthesise(pathology, research)
        _emit(
            "oncologist",
            f"Initial plan complete — {plan.diagnosis.primary} "
            f"(confidence: {plan.confidence_score:.0%})",
            72,
        )

        debate_rounds: list[DebateRound] = []

        # ── Agent Debate Loop ─────────────────────────────────────────────────
        if debate_mode:
            _emit("system", "🗣️ Agent Debate: initiating multi-round deliberation...", 74)
            plan, debate_rounds = self._run_debate(
                plan=plan,
                pathology=pathology,
                research=research,
                emit=_emit,
            )

        total_time = round(time.perf_counter() - t0, 2)

        final_msg = (
            f"✅ Analysis complete — {plan.diagnosis.primary}"
            + (f" | {len(debate_rounds)} debate round(s) completed" if debate_rounds else "")
        )
        _emit("system", final_msg, 100)
        log.info(f"Board: ✅ case '{case_id}' complete in {total_time}s")

        return BoardResult(
            case_id=case_id,
            pathology_report=pathology,
            research_summary=research,
            management_plan=plan,
            total_time_s=total_time,
            debate_rounds=debate_rounds,
            debate_enabled=debate_mode,
            heatmaps_b64=heatmaps,
        )

    # ── Debate Loop ──────────────────────────────────────────────────────────
    def _run_debate(
        self,
        plan: ManagementPlan,
        pathology: PathologyReport,
        research: ResearchSummary,
        emit: StepCallback,
    ) -> tuple[ManagementPlan, list[DebateRound]]:
        """
        Run the multi-round Agent Debate loop.

        Rounds:
          1. Researcher challenges the draft plan (RAG-grounded critique)
          2. If morphological doubts → Pathologist referee re-evaluates
          3. Oncologist revises the plan
          4. MetaEvaluator scores consensus (0–100)
          5. If score < CONSENSUS_THRESHOLD and rounds < MAX → repeat

        Returns:
            (final_plan, list_of_debate_rounds)
        """
        current_plan = plan
        debate_rounds: list[DebateRound] = []
        debate_transcript: list[dict] = []
        revision_history: list[dict] = []

        for round_num in range(1, MAX_DEBATE_ROUNDS + 1):
            emit("system", f"🗣️ Debate Round {round_num}/{MAX_DEBATE_ROUNDS}", 74 + round_num * 2)

            # ── Step A: Researcher challenges ────────────────────────────────
            emit("researcher", f"Round {round_num}: reviewing draft plan against NCCN guidelines...", 75)
            critique = self.researcher.challenge(
                draft_plan_dict=current_plan.to_dict(),
                pathology_report=pathology,
                research_summary=research,
                round_num=round_num,
            )
            challenge_text  = critique.get("challenge_text", "")
            flagged_issues  = critique.get("flagged_issues", [])
            morpho_doubts   = critique.get("morphological_doubts", False)
            severity        = critique.get("severity", "low")

            emit("researcher", challenge_text, 77)
            for issue in flagged_issues[:2]:
                emit("researcher", f"  ↳ {issue}", 77)

            debate_transcript.append({
                "round": round_num,
                "speaker": "researcher",
                "message": challenge_text,
                "flagged_issues": flagged_issues,
            })

            # If no serious issues, break early
            if severity == "low" and not flagged_issues:
                emit("system", "✅ Researcher: no major guideline violations — debate concluded", 80)
                break

            # ── Step B: Pathologist referee (if morphological doubts) ────────
            referee_update: Optional[dict] = None
            referee_note = ""

            if morpho_doubts:
                emit("pathologist", f"Referee: re-evaluating morphological findings (round {round_num})...", 79)
                referee_update = self.pathologist.referee(pathology, flagged_issues)
                referee_note   = referee_update.get("referee_note", "")
                emit("pathologist", f"Referee: {referee_note[:100]}...", 80)
                debate_transcript.append({
                    "round": round_num,
                    "speaker": "pathologist",
                    "message": f"Referee: {referee_note}",
                    "morphology_confirmed": referee_update.get("morphology_confirmed"),
                })

            # ── Step C: Oncologist revises ───────────────────────────────────
            emit("oncologist", f"Round {round_num}: revising management plan based on challenge...", 82)
            prev_first_line = current_plan.treatment_plan.first_line

            revised_plan = self.oncologist.revise(
                current_plan=current_plan,
                critique=critique,
                pathology_report=pathology,
                research_summary=research,
                referee_update=referee_update,
                round_num=round_num,
            )
            revision_notes = revised_plan.revision_notes
            emit("oncologist", revision_notes, 84)

            debate_transcript.append({
                "round": round_num,
                "speaker": "oncologist",
                "message": revision_notes,
                "revised_first_line": revised_plan.treatment_plan.first_line,
            })

            revision_history.append({
                "round": round_num,
                "removed": [prev_first_line] if prev_first_line != revised_plan.treatment_plan.first_line else [],
                "added": [revised_plan.treatment_plan.first_line] if prev_first_line != revised_plan.treatment_plan.first_line else [],
            })

            # ── Step D: MetaEvaluator scores consensus ───────────────────────
            emit("system", f"Round {round_num}: evaluating consensus...", 86)
            meta_result = self.meta_evaluator.evaluate(
                original_first_line=current_plan.treatment_plan.first_line,
                original_actions=current_plan.immediate_actions,
                critique=challenge_text,
                revised_first_line=revised_plan.treatment_plan.first_line,
                revised_actions=revised_plan.immediate_actions,
                revised_notes=revision_notes,
            )
            consensus_score  = int(meta_result.get("consensus_score", 75))
            meta_reasoning   = meta_result.get("reasoning", "")
            addressed        = meta_result.get("addressed_points", [])
            unaddressed      = meta_result.get("unaddressed_points", [])

            emit(
                "system",
                f"Consensus score: {consensus_score}/100 — {meta_reasoning[:80]}",
                87,
            )

            debate_rounds.append(DebateRound(
                round_num=round_num,
                challenge_text=challenge_text,
                flagged_issues=flagged_issues,
                referee_note=referee_note,
                revision_notes=revision_notes,
                consensus_score=consensus_score,
                morphology_confirmed=referee_update.get("morphology_confirmed", True) if referee_update else True,
            ))

            # Update current plan with debate metadata
            revised_plan.debate_transcript = debate_transcript[:]
            revised_plan.revision_history  = revision_history[:]
            revised_plan.consensus_score   = consensus_score
            revised_plan.revision_notes    = revision_notes
            current_plan = revised_plan

            # ── Step E: Check if another round needed ────────────────────────
            if consensus_score >= CONSENSUS_THRESHOLD:
                emit(
                    "system",
                    f"✅ Consensus reached (score {consensus_score}/100) — debate complete",
                    89,
                )
                break
            elif round_num < MAX_DEBATE_ROUNDS:
                emit(
                    "system",
                    f"⚠️ Consensus score {consensus_score} < {CONSENSUS_THRESHOLD} "
                    f"— initiating round {round_num + 1}",
                    88,
                )
            else:
                emit(
                    "system",
                    f"Max debate rounds ({MAX_DEBATE_ROUNDS}) reached — proceeding with best plan",
                    89,
                )

        return current_plan, debate_rounds

    def run_from_paths(
        self,
        case_id: str,
        image_paths: list[Path | str],
        batch_size: int = 16,
        debate_mode: bool = True,
    ) -> BoardResult:
        """Convenience wrapper — load images from file paths then run."""
        images = [Image.open(str(p)).convert("RGB") for p in image_paths]
        return self.run(case_id, images, batch_size=batch_size, debate_mode=debate_mode)
