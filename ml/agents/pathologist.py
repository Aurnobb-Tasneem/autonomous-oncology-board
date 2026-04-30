"""
ml/agents/pathologist.py
========================
Agent 1: Pathologist — GigaPath patch inference engine.

Responsibilities:
  - Accept a batch of 224×224 histopathology image patches
  - Run them through Prov-GigaPath to extract 1536-dim embeddings
  - Classify tissue type (using prototype-based cosine similarity)
  - Detect morphological abnormalities via embedding distance thresholds
  - Return a structured PathologyReport JSON

Output schema (PathologyReport):
  {
    "case_id": str,
    "n_patches": int,
    "tissue_type": str,          # dominant classification
    "confidence": float,         # 0–1
    "patch_findings": [          # per-patch details
      {
        "patch_id": int,
        "tissue_class": str,
        "class_confidence": float,
        "abnormality_score": float,   # 0=normal, 1=highly abnormal
        "embedding_norm": float
      }
    ],
    "summary": str,              # 1–2 sentence natural language summary
    "flags": [str],              # e.g. ["high_abnormality_detected"]
    "embedding_stats": {
      "mean_norm": float,
      "std_norm": float,
      "centroid": [float]        # 1536-dim mean embedding (for RAG retrieval)
    }
  }
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

from ml.models.gigapath_loader import (
    load_gigapath,
    embed_patches,
    extract_attention_heatmap,
    build_transform,
    EMBEDDING_DIM,
)
from ml.agents.uncertainty import mc_dropout_inference, UncertaintyResult

log = logging.getLogger(__name__)

# ── LC25000 tissue classes ───────────────────────────────────────────────────
# LC25000 dataset: 5 classes, 5000 patches each
# https://arxiv.org/abs/1912.12378
TISSUE_CLASSES = [
    "colon_adenocarcinoma",
    "colon_benign_tissue",
    "lung_adenocarcinoma",
    "lung_benign_tissue",
    "lung_squamous_cell_carcinoma",
]

TISSUE_CLASS_LABELS = {
    "colon_adenocarcinoma":         "Colon Adenocarcinoma",
    "colon_benign_tissue":          "Colon Benign Tissue",
    "lung_adenocarcinoma":          "Lung Adenocarcinoma",
    "lung_benign_tissue":           "Lung Benign Tissue",
    "lung_squamous_cell_carcinoma": "Lung Squamous Cell Carcinoma",
}

# Abnormality threshold — patches with cosine distance > this from the
# nearest class prototype are flagged as anomalous.
ABNORMALITY_THRESHOLD = 0.35


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class PatchFinding:
    patch_id: int
    tissue_class: str
    class_confidence: float
    abnormality_score: float
    embedding_norm: float


@dataclass
class EmbeddingStats:
    mean_norm: float
    std_norm: float
    centroid: list[float]   # 1536-dim


@dataclass
class PathologyReport:
    case_id: str
    n_patches: int
    tissue_type: str
    confidence: float
    patch_findings: list[PatchFinding]
    summary: str
    flags: list[str]
    embedding_stats: EmbeddingStats
    processing_time_s: float
    # ── Attention heatmaps (populated on demand) ────────────────────────────
    heatmaps_b64: list[str] = field(default_factory=list)
    # ── MC Dropout uncertainty (populated if uncertainty_mode=True) ─────────
    uncertainty_interval: str = ""          # e.g. "91.2% ± 4.2%"
    uncertainty_std: float = 0.0
    high_uncertainty: bool = False
    uncertainty_class_probs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Pathologist Agent ────────────────────────────────────────────────────────
class PathologistAgent:
    """
    Agent 1 of the Autonomous Oncology Board.

    Uses Prov-GigaPath (ViT-Giant, 1.1B params) to analyse histopathology
    image patches and produce a structured pathology report.

    The classification uses prototype-based cosine similarity:
      1. Load class prototypes from a pre-computed cache (or random init for demo)
      2. For each patch embedding, compute cosine similarity to all prototypes
      3. Assign the class with highest similarity
      4. Abnormality score = 1 - max_similarity (distance from nearest known class)

    For the hackathon demo, prototypes are seeded from random noise since we
    don't have pre-computed class centroids from LC25000. In production, run
    scripts/calibrate_prototypes.py first to compute real centroids.
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        prototype_path: Optional[Path] = None,
    ):
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        self.prototype_path = prototype_path
        self._model = None
        self._device = None
        self._prototypes: Optional[torch.Tensor] = None  # shape: (n_classes, 1536)
        self._transform = build_transform(augment=False)

        log.info("PathologistAgent: initialising (model load deferred to first call)")

    def _ensure_model_loaded(self):
        """Lazy model loading — only loads on first inference call."""
        if self._model is None:
            self._model, self._device = load_gigapath(
                hf_token=self.hf_token,
                device=None,
            )
            self._prototypes = self._load_prototypes()

    def _load_prototypes(self) -> torch.Tensor:
        """
        Load class prototype embeddings.

        Tries to load from prototype_path first. Falls back to deterministic
        random prototypes (seeded for reproducibility) for demo mode.
        """
        if self.prototype_path and Path(self.prototype_path).exists():
            log.info(f"PathologistAgent: loading prototypes from {self.prototype_path}")
            return torch.load(self.prototype_path, map_location="cpu")

        log.warning(
            "PathologistAgent: no prototype file found — using seeded random prototypes. "
            "Run scripts/calibrate_prototypes.py on LC25000 for real class centroids."
        )
        # Deterministic random prototypes — normalised to unit sphere
        torch.manual_seed(42)
        proto = torch.randn(len(TISSUE_CLASSES), EMBEDDING_DIM)
        proto = F.normalize(proto, dim=1)
        return proto  # shape: (5, 1536)

    # ── Preprocessing ────────────────────────────────────────────────────────
    def preprocess_images(self, images: list[Image.Image]) -> torch.Tensor:
        """
        Convert a list of PIL images to a preprocessed tensor batch.

        Args:
            images: List of PIL.Image objects (any size — will be resized).

        Returns:
            Tensor of shape (N, 3, 224, 224) in FP16.
        """
        tensors = [self._transform(img.convert("RGB")) for img in images]
        batch = torch.stack(tensors)          # (N, 3, 224, 224)
        return batch.to(torch.float16)

    def preprocess_paths(self, image_paths: list[Path | str]) -> torch.Tensor:
        """Load images from file paths and preprocess them."""
        images = [Image.open(str(p)) for p in image_paths]
        return self.preprocess_images(images)

    # ── Classification ───────────────────────────────────────────────────────
    def _classify_embeddings(
        self, embeddings: torch.Tensor
    ) -> tuple[list[str], list[float], list[float]]:
        """
        Classify embeddings using cosine similarity to prototypes.

        Args:
            embeddings: Tensor of shape (N, 1536).

        Returns:
            (classes, confidences, abnormality_scores) — each of length N.
        """
        # Normalise embeddings to unit sphere for cosine similarity
        emb_norm  = F.normalize(embeddings, dim=1)    # (N, 1536)
        proto_norm = F.normalize(self._prototypes.float(), dim=1)  # (C, 1536)

        # Cosine similarity matrix: (N, C)
        sim = emb_norm.float() @ proto_norm.T

        # Best class per patch
        best_sim, best_idx = sim.max(dim=1)

        classes       = [TISSUE_CLASSES[i.item()] for i in best_idx]
        confidences   = best_sim.tolist()
        abnorm_scores = (1.0 - best_sim).clamp(0, 1).tolist()

        return classes, confidences, abnorm_scores

    # ── Main inference ───────────────────────────────────────────────────────
    def analyse(
        self,
        case_id: str,
        images: list[Image.Image],
        batch_size: int = 16,
    ) -> PathologyReport:
        """
        Run full pathology analysis on a set of image patches.

        Args:
            case_id:    Unique identifier for this case.
            images:     List of PIL.Image objects (224×224 patches recommended).
            batch_size: Mini-batch size for GigaPath inference.

        Returns:
            PathologyReport with per-patch findings and overall assessment.
        """
        self._ensure_model_loaded()
        t0 = time.perf_counter()

        log.info(f"PathologistAgent: analysing case '{case_id}' — {len(images)} patches")

        # ── Step 1: Preprocess and embed ────────────────────────────────────
        patch_tensor = self.preprocess_images(images)  # (N, 3, 224, 224)
        embeddings   = embed_patches(
            self._model, patch_tensor, self._device, batch_size=batch_size
        )  # (N, 1536)

        # ── Step 2: Classify patches ─────────────────────────────────────────
        classes, confidences, abnorm_scores = self._classify_embeddings(embeddings)

        # ── Step 3: Build per-patch findings ────────────────────────────────
        norms = embeddings.norm(dim=1).tolist()
        patch_findings = [
            PatchFinding(
                patch_id=i,
                tissue_class=cls,
                class_confidence=round(conf, 4),
                abnormality_score=round(absc, 4),
                embedding_norm=round(norm, 4),
            )
            for i, (cls, conf, absc, norm) in enumerate(
                zip(classes, confidences, abnorm_scores, norms)
            )
        ]

        # ── Step 4: Aggregate to case-level ─────────────────────────────────
        from collections import Counter
        class_votes = Counter(classes)
        dominant_class, dominant_count = class_votes.most_common(1)[0]
        overall_confidence = dominant_count / len(classes)

        mean_abnorm = sum(abnorm_scores) / len(abnorm_scores)
        flags: list[str] = []
        if mean_abnorm > ABNORMALITY_THRESHOLD:
            flags.append("high_abnormality_detected")
        if len(class_votes) > 2:
            flags.append("heterogeneous_tissue")

        # ── Step 5: Embedding statistics ────────────────────────────────────
        centroid  = embeddings.mean(dim=0)   # (1536,)
        emb_norms = embeddings.norm(dim=1)

        emb_stats = EmbeddingStats(
            mean_norm=round(emb_norms.mean().item(), 4),
            std_norm=round(emb_norms.std().item(), 4),
            centroid=centroid.tolist(),
        )

        # ── Step 6: Natural language summary ────────────────────────────────
        label = TISSUE_CLASS_LABELS.get(dominant_class, dominant_class)
        pct   = round(overall_confidence * 100, 1)
        summary = (
            f"Analysis of {len(images)} patches indicates {label} in {pct}% of tissue regions "
            f"(mean abnormality score: {mean_abnorm:.2f}). "
        )
        if flags:
            summary += f"Flags: {', '.join(flags)}."
        else:
            summary += "No critical flags detected."

        processing_time = round(time.perf_counter() - t0, 2)
        log.info(
            f"PathologistAgent: done in {processing_time}s — "
            f"{label} ({pct}%) — flags: {flags}"
        )

        return PathologyReport(
            case_id=case_id,
            n_patches=len(images),
            tissue_type=dominant_class,
            confidence=round(overall_confidence, 4),
            patch_findings=patch_findings,
            summary=summary,
            flags=flags,
            embedding_stats=emb_stats,
            processing_time_s=processing_time,
        )

    def analyse_from_paths(
        self,
        case_id: str,
        image_paths: list[Path | str],
        batch_size: int = 16,
    ) -> PathologyReport:
        """Convenience wrapper — load images from file paths then analyse."""
        images = [Image.open(str(p)).convert("RGB") for p in image_paths]
        return self.analyse(case_id, images, batch_size=batch_size)

    # ── MC Dropout Uncertainty ────────────────────────────────────────────────
    def quantify_uncertainty(
        self,
        report: PathologyReport,
        images: list[Image.Image],
        n_passes: int = 20,
    ) -> PathologyReport:
        """
        Run Monte Carlo Dropout uncertainty quantification on a completed report.

        Augments the PathologyReport with uncertainty_interval, uncertainty_std,
        high_uncertainty, and uncertainty_class_probs fields.

        When high_uncertainty=True, the Oncologist will auto-flag:
        "⚠️ High diagnostic uncertainty — recommend second-opinion biopsy"

        Args:
            report:   The PathologyReport from analyse().
            images:   The same images passed to analyse().
            n_passes: Number of MC dropout passes (default 20).

        Returns:
            Updated PathologyReport with uncertainty fields populated.
        """
        self._ensure_model_loaded()
        log.info(f"PathologistAgent: running MC Dropout ({n_passes} passes)")

        patch_tensor = self.preprocess_images(images)

        try:
            unc = mc_dropout_inference(
                model=self._model,
                patch_tensor=patch_tensor,
                device=self._device,
                prototypes=self._prototypes,
                tissue_classes=TISSUE_CLASSES,
                n_passes=n_passes,
            )

            # Update report fields (dataclasses are mutable)
            report.uncertainty_interval = unc.uncertainty_interval
            report.uncertainty_std       = unc.std_confidence
            report.high_uncertainty      = unc.high_uncertainty
            report.uncertainty_class_probs = unc.class_probabilities

            # Add uncertainty flag
            if unc.high_uncertainty and "high_diagnostic_uncertainty" not in report.flags:
                report.flags.append("high_diagnostic_uncertainty")
                report.summary += (
                    f" ⚠️ High diagnostic uncertainty detected "
                    f"({unc.uncertainty_interval}) — recommend second-opinion biopsy."
                )

            log.info(
                f"PathologistAgent: uncertainty = {unc.uncertainty_interval}, "
                f"high={unc.high_uncertainty}"
            )

        except Exception as e:
            log.warning(f"PathologistAgent: MC Dropout failed ({e}) — skipping uncertainty")
            report.uncertainty_interval = "N/A (uncertainty estimation failed)"

        return report


    def referee(
        self,
        original_report: PathologyReport,
        flagged_issues: list[str],
    ) -> dict:
        """
        Referee re-evaluation for the Agent Debate loop.

        Called when the Researcher raises morphological doubts about the
        Pathologist's original findings. Re-examines the original embeddings
        to provide updated confidence estimates on disputed tissue regions.

        Args:
            original_report: The original PathologyReport.
            flagged_issues:  List of issues raised by the Researcher.

        Returns:
            dict with referee_note (str), updated_confidence (float),
            morphology_confirmed (bool)
        """
        log.info(f"PathologistAgent: referee re-evaluation for '{original_report.case_id}'")

        # Analyse the existing findings — compute inter-patch consistency
        confidences  = [p.class_confidence for p in original_report.patch_findings]
        abnormalities = [p.abnormality_score for p in original_report.patch_findings]
        classes      = [p.tissue_class for p in original_report.patch_findings]

        mean_conf  = sum(confidences) / len(confidences) if confidences else 0
        mean_abnorm = sum(abnormalities) / len(abnormalities) if abnormalities else 0
        dominant   = original_report.tissue_type
        consistency = sum(1 for c in classes if c == dominant) / len(classes) if classes else 0

        # Build referee assessment
        morphology_confirmed = mean_conf > 0.55 and consistency > 0.5

        if morphology_confirmed:
            referee_note = (
                f"Referee confirms: Nuclear morphology consistent with "
                f"{TISSUE_CLASS_LABELS.get(dominant, dominant)} in "
                f"{consistency:.0%} of patches (mean confidence: {mean_conf:.0%}). "
                f"Mean abnormality score {mean_abnorm:.2f} supports malignant classification."
            )
        else:
            referee_note = (
                f"Referee: Mixed morphology detected — {consistency:.0%} patch agreement. "
                f"Tissue heterogeneity noted (mean confidence: {mean_conf:.0%}). "
                f"Consider additional sectioning or IHC confirmation."
            )

        log.info(f"PathologistAgent: referee — confirmed={morphology_confirmed}, "
                 f"consistency={consistency:.0%}")

        return {
            "referee_note": referee_note,
            "updated_confidence": round(mean_conf * consistency, 4),
            "morphology_confirmed": morphology_confirmed,
            "patch_consistency": round(consistency, 4),
            "mean_abnormality": round(mean_abnorm, 4),
        }

    # ── Attention Heatmaps ────────────────────────────────────────────────────
    def generate_heatmaps(
        self,
        images: list[Image.Image],
        max_patches: int = 8,
    ) -> list[str]:
        """
        Generate GigaPath attention heatmaps for a set of patches.

        Uses Attention Rollout across all ViT transformer blocks to
        highlight morphologically suspicious tissue regions in red.
        High-attention areas are labeled "⚠ SUSPICIOUS".

        Args:
            images:      List of PIL.Image patches.
            max_patches: Cap the number of patches to process (prevent OOM).

        Returns:
            List of base64-encoded PNG strings (one per patch).
            Returns empty list on failure (graceful degradation).
        """
        self._ensure_model_loaded()
        imgs = images[:max_patches]
        log.info(f"PathologistAgent: generating attention heatmaps for {len(imgs)} patches")

        patch_tensor = self.preprocess_images(imgs)

        heatmaps = extract_attention_heatmap(
            model=self._model,
            patch_tensor=patch_tensor,
            device=self._device,
        )
        log.info(f"PathologistAgent: {len(heatmaps)} heatmaps generated")
        return heatmaps
