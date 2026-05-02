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

    # ── VLM Reconciliation ────────────────────────────────────────────────────

    def reconcile(
        self,
        gigapath_report,   # ml.agents.pathologist.PathologyReport
        vlm_opinion,       # ml.agents.vlm_pathologist.VLMOpinion
    ) -> dict:
        """
        Reconcile GigaPath embedding-based classification with Qwen2-VL visual
        morphology output to produce a cross-model agreement assessment.

        This is called once per board run, before the debate loop, to surface
        any disagreements between the vision foundation model (GigaPath) and the
        vision-language model (Qwen2-VL) early — so the Oncologist can factor
        in both signals.

        Args:
            gigapath_report: PathologyReport from Agent 1 (GigaPath embeddings).
            vlm_opinion:     VLMOpinion from Agent 1b (Qwen2-VL pixel analysis).

        Returns:
            dict with keys:
              agreement_score     (int 0–100)
              agreement_summary   (str)
              discrepancies       (list[str])
              combined_tissue_type  (str)   — consensus tissue label
              combined_morphology   (list[str]) — merged feature list
              vlm_added_findings    (list[str]) — VLM-only observations
        """
        # If VLM was unavailable, return a neutral pass-through result
        if not vlm_opinion.is_available:
            log.info(
                "MetaEvaluator.reconcile: VLM opinion unavailable "
                f"({vlm_opinion.error}) — skipping reconciliation"
            )
            return {
                "agreement_score":      -1,
                "agreement_summary":    f"VLM unavailable: {vlm_opinion.error}",
                "discrepancies":        [],
                "combined_tissue_type": gigapath_report.tissue_type,
                "combined_morphology":  gigapath_report.flags,
                "vlm_added_findings":   [],
            }

        prompt = f"""You are a senior pathology peer reviewer comparing two independent analyses
of the same histopathology case.

ANALYSIS A — GigaPath (embedding-based, ViT-Giant foundation model):
  Tissue classification : {gigapath_report.tissue_type.replace("_", " ").title()}
  Confidence            : {gigapath_report.confidence:.0%}
  Flags                 : {', '.join(gigapath_report.flags) if gigapath_report.flags else 'none'}
  Summary               : {gigapath_report.summary[:300]}

ANALYSIS B — Qwen2-VL-7B-Instruct (pixel-based VLM, direct image reading):
  Suspected tissue type   : {vlm_opinion.suspected_tissue_type}
  Malignancy indicators   : {', '.join(vlm_opinion.malignancy_indicators) if vlm_opinion.malignancy_indicators else 'none identified'}
  Morphology description  : {vlm_opinion.aggregate_description[:400]}
  Patches analysed        : {vlm_opinion.n_patches_processed}

Your tasks:
1. Score agreement between the two analyses (0–100).
   100 = full agreement on tissue type and malignancy.
   0   = complete disagreement (e.g. one says malignant lung, other says benign colon).
2. Identify any discrepancies.
3. Propose a combined consensus tissue type.
4. List morphological features mentioned by ONLY Analysis B (VLM-added findings).

Return ONLY valid JSON with exactly these fields:
{{
  "agreement_score": 82,
  "agreement_summary": "Both models agree on lung adenocarcinoma with glandular features...",
  "discrepancies": ["GigaPath flagged high abnormality; VLM did not mention mitotic figures"],
  "combined_tissue_type": "lung adenocarcinoma",
  "combined_morphology": ["glandular patterns", "nuclear atypia", "irregular borders"],
  "vlm_added_findings": ["micropapillary architecture noted in patch 2"]
}}"""

        try:
            if not self.llm.ping():
                log.warning("MetaEvaluator.reconcile: LLM unavailable — using heuristic")
                return self._heuristic_reconcile(gigapath_report, vlm_opinion)

            response = self.llm.generate_sync(
                prompt=prompt,
                system=(
                    "You are a neutral pathology peer reviewer. "
                    "Output valid JSON only — no markdown fences, no preamble."
                ),
                temperature=0.05,
                max_tokens=500,
            )
            text = response.text.strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            result = json.loads(match.group() if match else text)

            score = int(result.get("agreement_score", 75))
            log.info(
                f"MetaEvaluator.reconcile: agreement_score={score}  "
                f"combined_tissue='{result.get('combined_tissue_type', 'unknown')}'"
            )
            return result

        except Exception as e:
            log.warning(f"MetaEvaluator.reconcile: failed ({e}), using heuristic")
            return self._heuristic_reconcile(gigapath_report, vlm_opinion)

    def _heuristic_reconcile(self, gigapath_report, vlm_opinion) -> dict:
        """
        Heuristic reconciliation when the LLM is unavailable.

        Checks whether GigaPath's tissue_type string appears in the VLM's
        aggregate description — a simple but reliable proxy for agreement.
        """
        gp_tissue = gigapath_report.tissue_type.replace("_", " ").lower()
        vlm_desc  = vlm_opinion.aggregate_description.lower()

        # Token-level partial match (e.g. "lung" in a longer description)
        gp_tokens = [t for t in gp_tissue.split() if len(t) > 3]
        token_hits = sum(1 for t in gp_tokens if t in vlm_desc)
        agreement  = 85 if token_hits >= len(gp_tokens) // 2 + 1 else 55

        return {
            "agreement_score":     agreement,
            "agreement_summary":   (
                f"Heuristic: GigaPath ({gp_tissue}) token overlap "
                f"{token_hits}/{len(gp_tokens)} with VLM description."
            ),
            "discrepancies":       [] if agreement >= 80 else [
                "Tissue type labels differ between GigaPath and VLM — manual review recommended."
            ],
            "combined_tissue_type":  gigapath_report.tissue_type,
            "combined_morphology":   gigapath_report.flags,
            "vlm_added_findings":    vlm_opinion.malignancy_indicators,
        }
