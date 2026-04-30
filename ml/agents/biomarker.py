"""
ml/agents/biomarker.py
======================
Biomarker Layer — Interpretable Oncology Biomarker Extraction.

Maps the GigaPath 1536-dim embedding centroid to 8 clinically interpretable
biomarker scores using fixed seeded projection vectors (reproducible across runs).

This is a deterministic, lightweight method that requires no additional model:
  - For each biomarker, generate a fixed random direction vector (seeded)
  - Normalise both the centroid and the direction vector
  - Score = dot product (cosine similarity along that direction)
  - Normalise score 0–1 and label as Low / Moderate / High

Biomarkers:
  1. nuclear_pleomorphism    — variation in nuclear size/shape (malignancy marker)
  2. mitotic_index           — rate of cell division
  3. gland_formation         — architectural differentiation (Gleason-like)
  4. necrosis_extent         — tumour necrosis (aggressive feature)
  5. immune_infiltration      — tumour-infiltrating lymphocytes
  6. stroma_density          — stromal compartment (desmoplasia)
  7. cell_uniformity         — inverse of pleomorphism (uniformity)
  8. architecture_score      — overall tissue architecture preservation

Each biomarker returns:
  {
    "score": 0.73,
    "level": "High",
    "interpretation": "Elevated nuclear pleomorphism suggests high-grade malignancy"
  }

Usage:
    from ml.agents.biomarker import BiomarkerExtractor
    bx = BiomarkerExtractor()
    biomarkers = bx.extract(centroid)  # centroid is list[float], len=1536
"""

from __future__ import annotations

import logging
import math
from typing import Optional

log = logging.getLogger(__name__)

EMBEDDING_DIM = 1536

# ── Biomarker definitions ────────────────────────────────────────────────────
# Each entry: (name, seed_offset, clinical interpretation at Low/Moderate/High)
_BIOMARKER_DEFS: list[tuple[str, int, dict[str, str]]] = [
    (
        "nuclear_pleomorphism",
        0,
        {
            "Low":      "Nuclear size and shape are uniform — low-grade features",
            "Moderate": "Moderate nuclear variation — intermediate-grade features",
            "High":     "Marked nuclear pleomorphism — high-grade malignancy features",
        },
    ),
    (
        "mitotic_index",
        1,
        {
            "Low":      "Low mitotic activity — indolent tumour biology",
            "Moderate": "Moderate mitotic activity — intermediate proliferation rate",
            "High":     "High mitotic index — aggressive proliferation detected",
        },
    ),
    (
        "gland_formation",
        2,
        {
            "Low":      "Poor gland formation — poorly differentiated (Grade 3)",
            "Moderate": "Partial gland formation — moderately differentiated (Grade 2)",
            "High":     "Well-formed glands — well-differentiated (Grade 1)",
        },
    ),
    (
        "necrosis_extent",
        3,
        {
            "Low":      "Minimal necrosis — favourable prognostic feature",
            "Moderate": "Focal necrosis present — intermediate-risk feature",
            "High":     "Extensive tumour necrosis — adverse prognostic marker",
        },
    ),
    (
        "immune_infiltration",
        4,
        {
            "Low":      "Sparse TILs — immune-cold tumour microenvironment",
            "Moderate": "Moderate tumour-infiltrating lymphocytes",
            "High":     "Dense TIL infiltration — immune-hot; potential immunotherapy candidate",
        },
    ),
    (
        "stroma_density",
        5,
        {
            "Low":      "Minimal stroma — epithelial-dominant tumour",
            "Moderate": "Moderate desmoplastic stroma",
            "High":     "Dense desmoplastic stroma — may impair drug delivery",
        },
    ),
    (
        "cell_uniformity",
        6,
        {
            "Low":      "High cellular heterogeneity — complex clonal architecture",
            "Moderate": "Moderate cellular uniformity",
            "High":     "High cellular uniformity — monoclonal proliferation pattern",
        },
    ),
    (
        "architecture_score",
        7,
        {
            "Low":      "Disrupted tissue architecture — invasive pattern",
            "Moderate": "Partially preserved architecture",
            "High":     "Well-preserved tissue architecture — localised disease features",
        },
    ),
]

# Level thresholds
_LOW_THRESHOLD      = 0.33
_MODERATE_THRESHOLD = 0.66


def _label(score: float) -> str:
    if score < _LOW_THRESHOLD:
        return "Low"
    elif score < _MODERATE_THRESHOLD:
        return "Moderate"
    return "High"


def _generate_direction_vector(seed: int, dim: int = EMBEDDING_DIM) -> list[float]:
    """
    Generate a reproducible unit-norm direction vector using a seeded LCG.

    We avoid importing torch here to keep this module lightweight and
    importable in environments without GPU setup.

    Args:
        seed: Random seed (42 + biomarker_index per HANDOFF spec).
        dim:  Vector dimension.

    Returns:
        Normalised list[float] of length dim.
    """
    # Linear congruential generator — simple, deterministic, no dependencies
    a, c, m = 1664525, 1013904223, 2**32
    state = seed & 0xFFFFFFFF
    values: list[float] = []
    for _ in range(dim):
        state = (a * state + c) % m
        # Map to [-1, 1]
        values.append((state / m) * 2.0 - 1.0)

    # L2 normalise
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:
        return [0.0] * dim
    return [v / norm for v in values]


def _normalise_vector(vec: list[float]) -> list[float]:
    """L2 normalise a vector."""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return [0.0] * len(vec)
    return [v / norm for v in vec]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class BiomarkerExtractor:
    """
    Extract interpretable oncology biomarker scores from a GigaPath centroid.

    All direction vectors are pre-computed once at construction time and
    cached for the lifetime of the extractor instance.

    Usage:
        bx = BiomarkerExtractor()
        scores = bx.extract(centroid)
        # scores["nuclear_pleomorphism"] == {"score": 0.73, "level": "High", "interpretation": "..."}
    """

    def __init__(self):
        # Pre-compute and cache all direction vectors (seed = 42 + index)
        self._directions: list[tuple[str, list[float], dict[str, str]]] = []
        for idx, (name, seed_offset, interps) in enumerate(_BIOMARKER_DEFS):
            seed = 42 + seed_offset
            direction = _generate_direction_vector(seed, EMBEDDING_DIM)
            self._directions.append((name, direction, interps))
        log.info(f"BiomarkerExtractor: initialised with {len(self._directions)} biomarkers")

    def extract(self, centroid: list[float]) -> dict[str, dict]:
        """
        Extract all 8 biomarker scores from a 1536-dim centroid vector.

        Args:
            centroid: 1536-dim mean embedding from GigaPath (list[float]).

        Returns:
            Dict mapping biomarker name → {score, level, interpretation}.

        Example output:
            {
              "nuclear_pleomorphism": {"score": 0.73, "level": "High",
                  "interpretation": "Marked nuclear pleomorphism..."},
              ...
            }
        """
        if len(centroid) != EMBEDDING_DIM:
            log.warning(
                f"BiomarkerExtractor: expected {EMBEDDING_DIM}-dim centroid, "
                f"got {len(centroid)} — padding/truncating"
            )
            centroid = (centroid + [0.0] * EMBEDDING_DIM)[:EMBEDDING_DIM]

        # Normalise centroid
        norm_centroid = _normalise_vector(centroid)

        results: dict[str, dict] = {}
        for name, direction, interps in self._directions:
            # Dot product of unit vectors = cosine similarity along this direction
            raw_score = _dot(norm_centroid, direction)

            # raw_score is in [-1, 1] — map to [0, 1]
            score = (raw_score + 1.0) / 2.0
            score = round(max(0.0, min(1.0, score)), 4)

            level = _label(score)
            results[name] = {
                "score": score,
                "level": level,
                "interpretation": interps[level],
            }

        log.debug(f"BiomarkerExtractor: extracted {len(results)} biomarkers")
        return results

    @staticmethod
    def biomarker_names() -> list[str]:
        """Return ordered list of biomarker names."""
        return [name for name, _, _ in _BIOMARKER_DEFS]

    @staticmethod
    def summary(biomarkers: dict) -> str:
        """
        Generate a one-sentence biomarker summary for inclusion in reports.

        Args:
            biomarkers: Output from extract().

        Returns:
            Natural language summary string.
        """
        high = [k.replace("_", " ") for k, v in biomarkers.items() if v["level"] == "High"]
        low  = [k.replace("_", " ") for k, v in biomarkers.items() if v["level"] == "Low"]

        parts: list[str] = []
        if high:
            parts.append(f"Elevated: {', '.join(high)}")
        if low:
            parts.append(f"Reduced: {', '.join(low)}")

        if not parts:
            return "All biomarkers in moderate range — no extreme values detected."
        return "Biomarker profile: " + "; ".join(parts) + "."
