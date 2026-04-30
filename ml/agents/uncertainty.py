"""
ml/agents/uncertainty.py
========================
Monte Carlo Dropout Uncertainty Quantification for GigaPath.

Instead of a single deterministic forward pass, this module runs N=20
stochastic passes with dropout enabled at inference time. The variance
across predictions gives a principled uncertainty estimate.

Output:
  uncertainty_interval: "91.2% ± 4.2%"   (mean ± std across MC passes)
  high_uncertainty: True/False             (triggers second-biopsy flag)

Clinical significance:
  High uncertainty (std > 0.10) → Oncologist auto-flags:
  "⚠️ High diagnostic uncertainty — recommend second-opinion biopsy"

Reference:
  Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016
  https://arxiv.org/abs/1506.02142
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# Uncertainty threshold above which we flag for second biopsy
HIGH_UNCERTAINTY_STD = 0.10
MC_PASSES = 20  # Number of stochastic forward passes


@dataclass
class UncertaintyResult:
    """MC Dropout uncertainty estimate for a single case."""
    mean_confidence: float          # Mean across MC passes
    std_confidence: float           # Std across MC passes (= uncertainty)
    confidence_interval_low: float  # mean - 1.96*std (95% CI lower)
    confidence_interval_high: float # mean + 1.96*std (95% CI upper)
    uncertainty_interval: str       # e.g. "91.2% ± 4.2%"
    high_uncertainty: bool          # True if std > HIGH_UNCERTAINTY_STD
    n_passes: int
    dominant_class: str
    class_probabilities: dict       # mean probability per tissue class


def enable_dropout(model: nn.Module) -> None:
    """
    Enable dropout layers at inference time for MC Dropout.

    Sets only Dropout modules to train() mode, leaving all other
    layers (BatchNorm, attention, etc.) in eval() mode.
    """
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def mc_dropout_inference(
    model: nn.Module,
    patch_tensor: torch.Tensor,
    device: torch.device,
    prototypes: torch.Tensor,
    tissue_classes: list[str],
    n_passes: int = MC_PASSES,
    batch_size: int = 8,
) -> UncertaintyResult:
    """
    Run Monte Carlo Dropout inference for uncertainty quantification.

    Args:
        model:          GigaPath model in eval() mode.
        patch_tensor:   Preprocessed patches (N, 3, 224, 224).
        device:         Device the model lives on.
        prototypes:     Class prototype embeddings (C, 1536).
        tissue_classes: List of tissue class names (length C).
        n_passes:       Number of stochastic forward passes.
        batch_size:     Mini-batch size per pass.

    Returns:
        UncertaintyResult with mean, std, CI, and dominant class.
    """
    log.info(f"MC Dropout: running {n_passes} stochastic passes on {len(patch_tensor)} patches")

    # Enable dropout for stochastic inference
    model.eval()
    enable_dropout(model)

    all_pass_confidences: list[float] = []  # dominant class confidence per pass
    all_pass_class_probs: list[dict] = []   # full class distribution per pass

    proto_norm = F.normalize(prototypes.float(), dim=1)  # (C, 1536)

    with torch.no_grad():
        for pass_idx in range(n_passes):
            # Embed all patches for this stochastic pass
            pass_embeddings = []
            patches = patch_tensor.to(device)
            for i in range(0, len(patches), batch_size):
                chunk = patches[i:i + batch_size]
                emb = model(chunk)
                pass_embeddings.append(emb.float().cpu())

            embeddings = torch.cat(pass_embeddings, dim=0)  # (N, 1536)

            # Cosine similarity → class probabilities via softmax
            emb_norm  = F.normalize(embeddings, dim=1)
            sim       = emb_norm @ proto_norm.T  # (N, C)
            probs     = F.softmax(sim * 10.0, dim=1)  # temperature-scaled

            # Aggregate patches → case-level distribution
            mean_probs = probs.mean(dim=0)  # (C,)

            # Dominant class and its confidence
            dominant_idx = mean_probs.argmax().item()
            dominant_conf = mean_probs[dominant_idx].item()

            all_pass_confidences.append(dominant_conf)
            all_pass_class_probs.append({
                cls: round(mean_probs[i].item(), 4)
                for i, cls in enumerate(tissue_classes)
            })

    # Restore eval mode fully (disable dropout)
    model.eval()

    # Aggregate across passes
    confs = torch.tensor(all_pass_confidences)
    mean_conf = float(confs.mean())
    std_conf  = float(confs.std())

    # Dominant class by mean confidence
    mean_class_probs: dict[str, float] = {}
    for cls in tissue_classes:
        mean_class_probs[cls] = round(
            sum(p[cls] for p in all_pass_class_probs) / n_passes, 4
        )
    dominant_class = max(mean_class_probs, key=mean_class_probs.get)

    # 95% confidence interval
    ci_low  = max(0.0, mean_conf - 1.96 * std_conf)
    ci_high = min(1.0, mean_conf + 1.96 * std_conf)

    high_uncertainty = std_conf > HIGH_UNCERTAINTY_STD

    interval_str = f"{mean_conf*100:.1f}% ± {std_conf*100:.1f}%"

    log.info(
        f"MC Dropout: {n_passes} passes complete — "
        f"mean={mean_conf:.0%}, std={std_conf:.1%}, "
        f"high_uncertainty={high_uncertainty}"
    )

    return UncertaintyResult(
        mean_confidence=round(mean_conf, 4),
        std_confidence=round(std_conf, 4),
        confidence_interval_low=round(ci_low, 4),
        confidence_interval_high=round(ci_high, 4),
        uncertainty_interval=interval_str,
        high_uncertainty=high_uncertainty,
        n_passes=n_passes,
        dominant_class=dominant_class,
        class_probabilities=mean_class_probs,
    )
