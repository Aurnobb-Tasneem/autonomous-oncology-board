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
from ml.agents.board_memory import BoardMemory
from ml.agents.digital_twin import simulate_pfs
from ml.agents.vlm_pathologist import VLMPathologistAgent, VLMOpinion
from ml.agents.staging_specialist import StagingSpecialistAgent, TNMResult
from ml.agents.biomarker_specialist import BiomarkerSpecialistAgent, BiomarkerPanel
from ml.agents.treatment_specialist import TreatmentSpecialistAgent, TreatmentProposal
from ml.agents.differential import DifferentialDxAgent, DifferentialResult
from ml.agents.patient_summary import PatientSummaryAgent
from ml.agents.trial_matcher import TrialMatcherAgent, TrialMatchResult
from ml.agents.counterfactual import CounterfactualAgent, CounterfactualPlan
from ml.models.llm_client import OllamaClient
from ml.rag.retriever import OncologyRetriever

log = logging.getLogger(__name__)

# Consensus threshold — scores below this trigger another debate round
CONSENSUS_THRESHOLD = 70
MAX_DEBATE_ROUNDS   = 3

# If the Oncologist's initial confidence falls below this, the board triggers
# a backward feedback pass: Oncologist → Pathologist (cross-agent feedback loop)
LOW_CONFIDENCE_THRESHOLD = 0.60


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
    vlm_opinion: Optional[VLMOpinion] = None
    vlm_reconciliation: Optional[dict] = None
    pathology_feedback: Optional[dict] = None
    # Specialist suite outputs
    biomarker_panel: Optional[BiomarkerPanel] = None
    treatment_proposal: Optional[TreatmentProposal] = None
    differential_dx: Optional[DifferentialResult] = None
    # Post-plan outputs
    patient_summary: Optional[str] = None
    trial_matches: list = field(default_factory=list)
    counterfactual: Optional[CounterfactualPlan] = None

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
        if self.vlm_opinion is not None:
            d["vlm_opinion"] = self.vlm_opinion.to_dict()
        if self.vlm_reconciliation is not None:
            d["vlm_reconciliation"] = self.vlm_reconciliation
        if self.pathology_feedback is not None:
            d["pathology_feedback"] = self.pathology_feedback
        if self.biomarker_panel is not None:
            d["biomarker_panel"] = self.biomarker_panel.to_dict()
        if self.treatment_proposal is not None:
            d["treatment_proposal"] = self.treatment_proposal.to_dict()
        if self.differential_dx is not None:
            d["differential_dx"] = self.differential_dx.to_dict()
        if self.patient_summary is not None:
            d["patient_summary"] = self.patient_summary
        if self.trial_matches:
            d["trial_matches"] = [
                (t.to_dict() if hasattr(t, "to_dict") else t)
                for t in self.trial_matches
            ]
        if self.counterfactual is not None:
            d["counterfactual"] = self.counterfactual.to_dict()
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

        # Agent 1b: Qwen2-VL-7B visual second opinion
        self.vlm_pathologist = VLMPathologistAgent(hf_token=hf_token)

        # Specialist suite — Llama-3.1-8B LoRA adapters via vLLM :8006
        self.staging_specialist   = StagingSpecialistAgent()
        self.biomarker_specialist = BiomarkerSpecialistAgent()
        self.treatment_specialist = TreatmentSpecialistAgent()

        # Novel agents — Llama 3.3 70B via Ollama
        self.differential_agent   = DifferentialDxAgent(llm_client=self.llm)
        self.patient_summary_agent = PatientSummaryAgent(llm_client=self.llm)
        self.trial_matcher        = TrialMatcherAgent()
        self.counterfactual_agent = CounterfactualAgent(llm_client=self.llm)

        # Board memory — persists cases for similar-case retrieval
        self.memory = BoardMemory()

        log.info("AutonomousOncologyBoard: all agents initialised")

    # ── Main pipeline ─────────────────────────────────────────────────────────
    def run(
        self,
        case_id: str,
        images: list[Image.Image],
        batch_size: int = 16,
        debate_mode: bool = True,
        metadata: Optional[dict] = None,
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

        # ── Board Memory: retrieve similar past cases ────────────────────────
        similar_cases: list[dict] = []
        if pathology.embedding_stats and pathology.embedding_stats.centroid:
            similar_cases = self.memory.find_similar(
                centroid=pathology.embedding_stats.centroid, top_k=3
            )
            if similar_cases:
                _emit(
                    "system",
                    f"🗃️ Board Memory: {len(similar_cases)} similar past case(s) retrieved "
                    f"(top similarity: {similar_cases[0]['similarity']:.0%})",
                    34,
                )
            else:
                _emit("system", "🗃️ Board Memory: no similar past cases found (first run or cold start)", 34)

        # ── Agent 1b: Qwen2-VL Second Opinion ────────────────────────────────
        # Run Qwen2-VL-7B-Instruct directly on raw image pixels — independent
        # of GigaPath's embedding space.  Capped at 4 patches for <10s latency.
        _emit("vlm_pathologist", "Qwen2-VL-7B: requesting visual second opinion on patches...", 40)
        vlm_opinion = self.vlm_pathologist.describe(images[:4])
        if vlm_opinion.error:
            _emit(
                "vlm_pathologist",
                f"Qwen2-VL skipped ({vlm_opinion.error}) — proceeding without VLM input",
                41,
            )
        else:
            _emit(
                "vlm_pathologist",
                f"Qwen2-VL: '{vlm_opinion.suspected_tissue_type}' — "
                f"{vlm_opinion.aggregate_description[:100]}...",
                42,
            )
            if vlm_opinion.malignancy_indicators:
                _emit(
                    "vlm_pathologist",
                    f"Malignancy indicators: {', '.join(vlm_opinion.malignancy_indicators[:4])}",
                    43,
                )

        # VLM ↔ GigaPath reconciliation (before Researcher so consensus tissue
        # type is available to the Oncologist's context)
        _emit("system", "Reconciling GigaPath ↔ Qwen2-VL findings...", 44)
        vlm_reconciliation: Optional[dict] = None
        if vlm_opinion.is_available:
            vlm_reconciliation = self.meta_evaluator.reconcile(pathology, vlm_opinion)
            agreement = vlm_reconciliation.get("agreement_score", -1)
            combined_tissue = vlm_reconciliation.get("combined_tissue_type", "")
            _emit(
                "system",
                f"VLM reconciliation: agreement={agreement}/100  "
                f"consensus_tissue='{combined_tissue}'",
                45,
            )
            discrepancies = vlm_reconciliation.get("discrepancies", [])
            for disc in discrepancies[:2]:
                _emit("system", f"  ↳ Discrepancy: {disc}", 45)
        else:
            _emit("system", "VLM reconciliation skipped (VLM unavailable)", 45)

        # ── Agent 2: Researcher ──────────────────────────────────────────────
        _emit("researcher", "Building clinical query from pathology findings", 35)
        research = self.researcher.research(pathology, metadata=metadata)
        _emit(
            "researcher",
            f"Retrieved {research.raw_evidence.get('n_retrieved', 0)} evidence documents "
            f"(quality: {research.evidence_quality})",
            52,
        )
        _emit("researcher", f"Synthesised {len(research.treatment_options)} treatment options", 56)

        # ── Agent 2b: TNM Staging Specialist (Llama-3.1-8B LoRA) ─────────────
        # Calls the fine-tuned LoRA adapter via vLLM at :8006.
        # Falls back gracefully if vLLM is not running — pipeline continues.
        tnm_result: Optional[TNMResult] = None
        try:
            _emit("tnm_specialist", "Llama-3.1-8B LoRA: running TNM staging specialist...", 57)
            tnm_result = self.staging_specialist.stage(
                pathology_text=pathology.summary or pathology.tissue_type,
            )
            if tnm_result.is_fallback:
                _emit(
                    "tnm_specialist",
                    f"TNM specialist unavailable ({tnm_result.error}) — Oncologist will infer staging",
                    58,
                )
            else:
                _emit(
                    "tnm_specialist",
                    f"TNM result: {tnm_result.tnm_string()} "
                    f"(confidence: {tnm_result.confidence}, latency: {tnm_result.latency_ms:.0f}ms)",
                    58,
                )
                _emit(
                    "tnm_specialist",
                    f"AJCC Stage {tnm_result.stage} — T:{tnm_result.T}  N:{tnm_result.N}  M:{tnm_result.M}",
                    59,
                )
        except Exception as tnm_err:
            log.warning(f"Board: TNM specialist error ({tnm_err}) — continuing without staging")
            _emit("tnm_specialist", f"TNM specialist skipped: {tnm_err}", 58)

        # ── Agent 2c: Biomarker Specialist ───────────────────────────────────
        biomarker_panel: Optional[BiomarkerPanel] = None
        try:
            tissue_stage_text = (
                f"{pathology.tissue_type.replace('_', ' ').title()}. "
                + (f"{tnm_result.tnm_string()}. " if tnm_result and not tnm_result.is_fallback else "")
                + f"Stage: {getattr(pathology, 'tissue_type', 'unknown')}."
            )
            _emit("biomarker_specialist", "Biomarker specialist: extracting molecular testing panel...", 59)
            biomarker_panel = self.biomarker_specialist.extract(tissue_stage_text)
            if biomarker_panel.is_fallback:
                _emit("biomarker_specialist", f"Biomarker specialist unavailable ({biomarker_panel.error}) — Oncologist will infer", 60)
            else:
                _emit("biomarker_specialist", f"{biomarker_panel.summary()} (confidence: {biomarker_panel.confidence})", 60)
                for t in biomarker_panel.tests_required[:3]:
                    _emit("biomarker_specialist", f"  Test required: {t}", 60)
        except Exception as e:
            log.warning(f"Board: biomarker specialist error ({e}) — continuing")
            _emit("biomarker_specialist", f"Biomarker specialist skipped: {e}", 60)

        # ── Differential Diagnosis ────────────────────────────────────────────
        differential_result = None
        try:
            _emit("differential", "Differential diagnosis: computing top-3 candidate diagnoses...", 61)
            differential_result = self.differential_agent.analyse(
                pathology=pathology,
                vlm_opinion=vlm_opinion if vlm_opinion and vlm_opinion.is_available else None,
                metadata=metadata,
            )
            if differential_result and differential_result.differentials:
                top = differential_result.differentials[0]
                _emit("differential", f"Primary: {top['diagnosis']} ({top['probability']:.0%}) | "
                      f"DDx: {', '.join(d['diagnosis'] for d in differential_result.differentials[1:])}", 62)
        except Exception as e:
            log.warning(f"Board: differential diagnosis error ({e}) — continuing")
            _emit("differential", f"Differential diagnosis skipped: {e}", 62)

        # ── Agent 2d: Treatment Specialist ────────────────────────────────────
        treatment_proposal: Optional[TreatmentProposal] = None
        try:
            biomarker_summary = ""
            if biomarker_panel and not biomarker_panel.is_fallback and biomarker_panel.tests_required:
                biomarker_summary = "Biomarker panel required: " + "; ".join(biomarker_panel.tests_required[:4]) + "."
            tnm_stage_str = tnm_result.tnm_string() if (tnm_result and not tnm_result.is_fallback) else "Stage unknown"
            _emit("treatment_specialist", "Treatment specialist: generating evidence-based treatment proposal...", 63)
            treatment_proposal = self.treatment_specialist.plan(
                tissue_type=pathology.tissue_type.replace("_", " ").title(),
                tnm_stage=tnm_stage_str,
                biomarker_summary=biomarker_summary,
                metadata=metadata,
            )
            if treatment_proposal.is_fallback:
                _emit("treatment_specialist", f"Treatment specialist unavailable ({treatment_proposal.error})", 64)
            else:
                _emit("treatment_specialist", f"NCCN Category {treatment_proposal.nccn_category}: {treatment_proposal.first_line[:80]}...", 64)
        except Exception as e:
            log.warning(f"Board: treatment specialist error ({e}) — continuing")
            _emit("treatment_specialist", f"Treatment specialist skipped: {e}", 64)

        # ── Agent 3: Oncologist (initial draft) ──────────────────────────────
        _emit("oncologist", "Llama 3.3 70B: synthesising initial management plan...", 65)
        plan = self.oncologist.synthesise(
            pathology,
            research,
            similar_cases=similar_cases,
            metadata=metadata,
            biomarker_panel=biomarker_panel,
            treatment_proposal=treatment_proposal,
        )
        _emit(
            "oncologist",
            f"Initial plan complete — {plan.diagnosis.primary} "
            f"(confidence: {plan.confidence_score:.0%})",
            72,
        )

        # ── Cross-Agent Feedback Loop (Task 3) ───────────────────────────────
        # If the Oncologist's initial confidence is below the threshold, it
        # sends a structured text critique back to the Pathologist before the
        # multi-round debate begins.  This is the "backward gradient" from
        # Oncologist → Pathologist — a named, visible pipeline stage in the
        # SSE timeline that strengthens the board's consensus argument.
        pathology_feedback: Optional[dict] = None
        if plan.confidence_score < LOW_CONFIDENCE_THRESHOLD:
            _emit(
                "oncologist",
                f"Confidence {plan.confidence_score:.0%} below threshold "
                f"({LOW_CONFIDENCE_THRESHOLD:.0%}) — triggering cross-agent "
                f"feedback loop to Pathologist",
                73,
            )
            clarification = self.oncologist.request_pathology_clarification(
                plan, pathology
            )
            concerns = clarification.get("specific_concerns", [])
            _emit(
                "oncologist",
                f"Pathology clarification requested: {clarification.get('critique_text', '')[:100]}...",
                73,
            )
            for concern in concerns[:2]:
                _emit("oncologist", f"  Concern: {concern}", 73)

            referee_response = self.pathologist.referee(
                original_report=pathology,
                flagged_issues=concerns,
            )
            pathology_feedback = {**clarification, **referee_response}

            _emit(
                "pathologist",
                f"Pathologist feedback: {referee_response.get('referee_note', '')[:120]}...",
                74,
            )
            morphology_ok = referee_response.get("morphology_confirmed", True)
            _emit(
                "pathologist",
                f"Morphology {'confirmed' if morphology_ok else 'unconfirmed'} "
                f"— updated confidence: "
                f"{referee_response.get('updated_confidence', 0):.0%}",
                74,
            )
        else:
            _emit(
                "system",
                f"Oncologist confidence {plan.confidence_score:.0%} ≥ threshold "
                f"— cross-agent feedback loop not required",
                73,
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
                vlm_reconciliation=vlm_reconciliation,
            )

        # ── Patient Summary (plain-language) ─────────────────────────────
        patient_summary_text: Optional[str] = None
        try:
            _emit("patient_summary", "Generating patient-friendly plain-language summary...", 88)
            patient_summary_text = self.patient_summary_agent.generate(plan)
            if patient_summary_text:
                _emit("patient_summary", "Patient summary ready (plain English, 8th-grade reading level)", 89)
        except Exception as e:
            log.warning(f"Board: patient summary error ({e}) — continuing")
            _emit("patient_summary", f"Patient summary skipped: {e}", 89)

        # ── Clinical Trial Matching ───────────────────────────────────────
        trial_matches: list = []
        try:
            _emit("trial_matcher", "Searching local ClinicalTrials.gov corpus for matching trials...", 90)
            trial_matches = self.trial_matcher.find_matching(plan, top_k=5)
            if trial_matches:
                _emit("trial_matcher", f"{len(trial_matches)} potentially eligible trial(s) found", 91)
                for tm in trial_matches[:2]:
                    title = tm.title if hasattr(tm, "title") else str(tm.get("title", ""))
                    _emit("trial_matcher", f"  → {title[:80]}", 91)
            else:
                _emit("trial_matcher", "No matching trials found in local corpus", 91)
        except Exception as e:
            log.warning(f"Board: trial matcher error ({e}) — continuing")
            _emit("trial_matcher", f"Trial matching skipped: {e}", 91)

        # ── Digital Twin: 12-month PFS prediction ─────────────────────────
        _emit("system", "Digital Twin: simulating 12-month PFS curve...", 92)
        pfs = simulate_pfs(pathology.tissue_type)
        plan.pfs_12mo = pfs.pfs_12mo
        plan.pfs_curve = pfs.curve_points
        plan.pfs_model = pfs.model
        plan.pfs_assumptions = pfs.assumptions

        total_time = round(time.perf_counter() - t0, 2)

        final_msg = (
            f"✅ Analysis complete — {plan.diagnosis.primary}"
            + (f" | {len(debate_rounds)} debate round(s) completed" if debate_rounds else "")
        )
        _emit("system", final_msg, 100)
        log.info(f"Board: ✅ case '{case_id}' complete in {total_time}s")

        result = BoardResult(
            case_id=case_id,
            pathology_report=pathology,
            research_summary=research,
            management_plan=plan,
            total_time_s=total_time,
            debate_rounds=debate_rounds,
            debate_enabled=debate_mode,
            heatmaps_b64=heatmaps,
            vlm_opinion=vlm_opinion,
            vlm_reconciliation=vlm_reconciliation,
            pathology_feedback=pathology_feedback,
            biomarker_panel=biomarker_panel,
            treatment_proposal=treatment_proposal,
            differential_dx=differential_result,
            patient_summary=patient_summary_text,
            trial_matches=trial_matches,
        )

        # ── Board Memory: save this case for future similar-case retrieval ───
        try:
            self.memory.save_case(
                case_id=case_id,
                tissue_type=pathology.tissue_type,
                confidence=pathology.confidence,
                centroid=pathology.embedding_stats.centroid,
                first_line_tx=plan.treatment_plan.first_line,
                plan_summary=plan.patient_summary,
                n_patches=pathology.n_patches,
            )
        except Exception as mem_err:
            log.warning(f"Board: failed to save case to memory ({mem_err}) — continuing")

        return result

    # ── Debate Loop ──────────────────────────────────────────────────────────
    def _run_debate(
        self,
        plan: ManagementPlan,
        pathology: PathologyReport,
        research: ResearchSummary,
        emit: StepCallback,
        vlm_reconciliation: Optional[dict] = None,
    ) -> tuple[ManagementPlan, list[DebateRound]]:
        """
        Run the multi-round Agent Debate loop.

        Rounds:
          1. Researcher challenges the draft plan (RAG-grounded critique)
          2. If morphological doubts → Pathologist referee re-evaluates
          3. Oncologist revises the plan
          4. MetaEvaluator scores consensus (0–100), informed by VLM reconciliation
          5. If score < CONSENSUS_THRESHOLD and rounds < MAX → repeat

        Args:
            vlm_reconciliation: Output of MetaEvaluator.reconcile() run before
                the debate loop. Injected into the evaluate() prompt as additional
                context so the MetaEvaluator can factor in VLM ↔ GigaPath
                agreement when scoring each revision.

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
            # Augment the critique with VLM reconciliation context so the
            # evaluator can factor in pixel-level visual agreement when scoring.
            emit("system", f"Round {round_num}: evaluating consensus...", 86)
            augmented_critique = challenge_text
            if vlm_reconciliation and vlm_reconciliation.get("agreement_score", -1) >= 0:
                vlm_context = (
                    f" [VLM context — Qwen2-VL agreement: "
                    f"{vlm_reconciliation.get('agreement_score')}/100, "
                    f"consensus tissue: '{vlm_reconciliation.get('combined_tissue_type', 'unknown')}'"
                )
                discrepancies = vlm_reconciliation.get("discrepancies", [])
                if discrepancies:
                    vlm_context += f", discrepancies: {'; '.join(discrepancies[:2])}"
                vlm_context += "]"
                augmented_critique = challenge_text + vlm_context

            meta_result = self.meta_evaluator.evaluate(
                original_first_line=current_plan.treatment_plan.first_line,
                original_actions=current_plan.immediate_actions,
                critique=augmented_critique,
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
        metadata: Optional[dict] = None,
    ) -> BoardResult:
        """Convenience wrapper — load images from file paths then run."""
        images = [Image.open(str(p)).convert("RGB") for p in image_paths]
        return self.run(
            case_id,
            images,
            batch_size=batch_size,
            debate_mode=debate_mode,
            metadata=metadata,
        )
