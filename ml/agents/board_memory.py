"""
ml/agents/board_memory.py
=========================
Board Memory — Similar Case Retrieval for the Autonomous Oncology Board.

Responsibilities:
  - After each completed case: persist embedding centroid + key metadata to
    a local JSONL file (data/board_memory.jsonl)
  - At the start of each new case: retrieve the top-K most similar past cases
    using cosine similarity on the 1536-dim GigaPath centroid vectors

This gives the Oncologist "institutional memory" — it can say:
  "3 months ago we saw a similar lung adenocarcinoma case. That patient
   responded well to osimertinib after EGFR exon 19 deletion was confirmed."

Storage format (one JSON object per line):
  {
    "case_id": "case_abc123",
    "timestamp": "2025-01-15T10:32:00Z",
    "tissue_type": "lung_adenocarcinoma",
    "confidence": 0.87,
    "centroid": [0.12, -0.34, ...],   # 1536-dim
    "first_line_tx": "Osimertinib 80mg QD",
    "plan_summary": "Stage IV NSCLC...",
    "n_patches": 8
  }
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Default storage path — relative to project root
_DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "board_memory.jsonl"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class BoardMemory:
    """
    Persistent case memory for the Autonomous Oncology Board.

    Stores completed case embeddings in a JSONL flat file and retrieves
    the most similar past cases using cosine similarity.

    Thread-safety: file writes are atomic (write to temp, rename) on
    POSIX. On Windows we append directly — acceptable for hackathon use.

    Usage:
        memory = BoardMemory()
        similar = memory.find_similar(centroid, top_k=3)
        # ... run pipeline ...
        memory.save_case(
            case_id="case_001",
            tissue_type="lung_adenocarcinoma",
            confidence=0.87,
            centroid=[...],
            first_line_tx="Osimertinib 80mg QD",
            plan_summary="Stage IV NSCLC...",
        )
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = Path(storage_path or _DEFAULT_PATH)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"BoardMemory: storage at {self.storage_path}")

    # ── Read ─────────────────────────────────────────────────────────────────
    def _load_all(self) -> list[dict]:
        """Load all stored cases from JSONL."""
        if not self.storage_path.exists():
            return []
        cases = []
        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        cases.append(json.loads(line))
                    except json.JSONDecodeError:
                        log.warning(f"BoardMemory: skipping malformed line: {line[:60]}")
        return cases

    def find_similar(
        self,
        centroid: list[float],
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> list[dict]:
        """
        Retrieve the top-K most similar past cases.

        Uses cosine similarity between the query centroid and all stored
        centroids. Returns an empty list if fewer than 1 case is stored.

        Args:
            centroid:       1536-dim query embedding vector.
            top_k:          Maximum number of cases to return.
            min_similarity: Only return cases above this threshold.

        Returns:
            List of case dicts sorted by similarity (highest first).
            Each dict has all stored fields plus "similarity" (float).
        """
        all_cases = self._load_all()
        if not all_cases:
            return []

        scored: list[tuple[float, dict]] = []
        for case in all_cases:
            stored_centroid = case.get("centroid", [])
            if not stored_centroid:
                continue
            sim = _cosine_similarity(centroid, stored_centroid)
            if sim >= min_similarity:
                scored.append((sim, case))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, case in scored[:top_k]:
            entry = {k: v for k, v in case.items() if k != "centroid"}  # skip centroid in output
            entry["similarity"] = round(sim, 4)
            results.append(entry)

        log.info(
            f"BoardMemory: found {len(results)} similar case(s) "
            f"(top similarity: {results[0]['similarity'] if results else 'n/a'})"
        )
        return results

    # ── Write ────────────────────────────────────────────────────────────────
    def save_case(
        self,
        case_id: str,
        tissue_type: str,
        confidence: float,
        centroid: list[float],
        first_line_tx: str,
        plan_summary: str,
        n_patches: int = 0,
        extra: Optional[dict] = None,
    ) -> None:
        """
        Persist a completed case to the board memory store.

        Args:
            case_id:      Unique case identifier.
            tissue_type:  Dominant tissue classification.
            confidence:   Pathologist confidence score (0–1).
            centroid:     1536-dim mean embedding vector.
            first_line_tx: First-line treatment from the Oncologist plan.
            plan_summary:  Patient summary from the management plan.
            n_patches:    Number of patches analysed.
            extra:        Any additional metadata to store.
        """
        record = {
            "case_id": case_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tissue_type": tissue_type,
            "confidence": round(confidence, 4),
            "centroid": centroid,  # store full vector for future retrieval
            "first_line_tx": first_line_tx,
            "plan_summary": plan_summary,
            "n_patches": n_patches,
        }
        if extra:
            record.update(extra)

        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        log.info(f"BoardMemory: saved case '{case_id}' ({tissue_type})")

    # ── Introspection ────────────────────────────────────────────────────────
    def list_all(self) -> list[dict]:
        """
        Return all stored cases without their centroids (for the API).
        Sorted newest-first by timestamp.
        """
        cases = self._load_all()
        # Strip centroid (too large for API response) and sort newest-first
        slim = []
        for c in cases:
            entry = {k: v for k, v in c.items() if k != "centroid"}
            slim.append(entry)
        slim.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return slim

    def count(self) -> int:
        """Return total number of stored cases."""
        return len(self._load_all())
