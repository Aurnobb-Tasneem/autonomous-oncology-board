"""
ml/agents/trial_matcher.py
===========================
Clinical Trial Matching Agent.

Retrieves top-K matching oncology clinical trials from a pre-indexed local
ClinicalTrials.gov corpus (static JSON snapshot — no live API calls per
CLAUDE.md §12).

The 500-trial snapshot lives at aob/ml/rag/trials/trials_snapshot.json.
Trials are indexed in Qdrant under collection "oncology_trials".

Matching strategy:
    1. Extract tissue_type + TNM stage + biomarker panel from ManagementPlan
    2. Build a semantic query from these fields
    3. Retrieve top-K trials by cosine similarity from Qdrant
    4. Re-rank by eligibility criteria overlap (rule-in/rule-out check)
    5. Return list of TrialMatch with eligibility_score and eligibility_flags

Usage:
    agent = TrialMatcherAgent()
    matches = agent.find_matching(management_plan, top_k=5)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_TRIALS_SNAPSHOT_PATH = Path(
    os.getenv("TRIALS_SNAPSHOT_PATH",
              str(Path(__file__).parent.parent / "rag" / "trials" / "trials_snapshot.json"))
)

QDRANT_COLLECTION = "oncology_trials"
QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")

_trials_cache: Optional[list[dict]] = None


@dataclass
class TrialMatch:
    """A matching clinical trial with eligibility assessment."""
    trial_id:          str
    title:             str
    phase:             str             # "Phase I", "Phase II", "Phase III"
    cancer_type:       str
    biomarker_focus:   str             # e.g. "EGFR+", "MSI-H", "KRAS G12C"
    eligibility_score: float           # 0–1 cosine similarity
    eligibility_flags: dict[str, str]  # {"age": "eligible", "ecog": "check", ...}
    nct_id:            str
    study_status:      str             # "Recruiting", "Active", etc.
    brief_summary:     str
    inclusion_snippet: str
    exclusion_snippet: str
    contact_info:      str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_potentially_eligible(self) -> bool:
        return self.eligibility_score >= 0.4

    def badge(self) -> str:
        if self.eligibility_score >= 0.7:
            return "likely eligible"
        elif self.eligibility_score >= 0.4:
            return "potentially eligible"
        else:
            return "criteria mismatch"


def _load_snapshot() -> list[dict]:
    """Load the trials snapshot JSON (cached in memory)."""
    global _trials_cache
    if _trials_cache is not None:
        return _trials_cache

    if not _TRIALS_SNAPSHOT_PATH.exists():
        log.warning(f"Trials snapshot not found at {_TRIALS_SNAPSHOT_PATH}")
        return []

    try:
        with open(_TRIALS_SNAPSHOT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _trials_cache = data if isinstance(data, list) else data.get("trials", [])
        log.info(f"Loaded {len(_trials_cache)} trials from snapshot")
        return _trials_cache
    except Exception as e:
        log.warning(f"Failed to load trials snapshot: {e}")
        return []


def _simple_keyword_match(query: str, trial: dict) -> float:
    """
    Fast keyword-based matching score when Qdrant is unavailable.
    Returns 0–1 float.
    """
    query_lower = query.lower()
    score = 0.0

    target_text = " ".join([
        trial.get("cancer_type", ""),
        trial.get("biomarker_focus", ""),
        trial.get("brief_summary", ""),
        trial.get("inclusion_snippet", ""),
        trial.get("title", ""),
    ]).lower()

    # Keyword matching
    keywords = query_lower.split()
    for kw in keywords:
        if len(kw) > 3 and kw in target_text:
            score += 1.0

    # Normalise by number of query keywords
    if keywords:
        score = min(1.0, score / len(keywords))

    return round(score, 3)


def _build_query_from_plan(plan) -> str:
    """Build a semantic search query from a ManagementPlan."""
    plan_dict = plan.to_dict() if hasattr(plan, "to_dict") else {}
    diag      = plan_dict.get("diagnosis", {})
    tx        = plan_dict.get("treatment_plan", {})

    parts = []
    if diag.get("primary"):
        parts.append(diag["primary"])
    if diag.get("tnm_stage"):
        parts.append(diag["tnm_stage"])
    if tx.get("first_line"):
        # Extract key drug/mechanism mentions
        fl = tx["first_line"]
        for token in ["EGFR", "ALK", "KRAS", "PD-L1", "HER2", "MSI-H", "BRAF",
                      "pembrolizumab", "osimertinib", "alectinib", "bevacizumab",
                      "oxaliplatin", "FOLFOX", "adenocarcinoma", "squamous"]:
            if token.lower() in fl.lower():
                parts.append(token)

    return " ".join(parts)


class TrialMatcherAgent:
    """
    Clinical Trial Matching Agent.

    Uses Qdrant for semantic search over a pre-indexed corpus of 500
    curated oncology trials, with keyword fallback if Qdrant is unavailable.
    """

    def __init__(
        self,
        trials_path: Optional[Path] = None,
        qdrant_url:  str = QDRANT_URL,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.trials_path     = trials_path or _TRIALS_SNAPSHOT_PATH
        self.qdrant_url      = qdrant_url
        self.embedding_model = embedding_model
        self._qdrant         = None
        self._embedder       = None
        self._trials         = _load_snapshot()

    def _get_qdrant(self):
        if self._qdrant is not None:
            return self._qdrant
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=self.qdrant_url, timeout=3)
            client.get_collections()  # test connection
            self._qdrant = client
            log.info("TrialMatcherAgent: Qdrant connected")
        except Exception as e:
            log.warning(f"TrialMatcherAgent: Qdrant unavailable ({e}) — using keyword matching")
            self._qdrant = None
        return self._qdrant

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model)
        except ImportError:
            pass
        return self._embedder

    def _semantic_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Search Qdrant for top-K trial IDs by semantic similarity.
        Returns list of (trial_id, score) tuples.
        """
        client   = self._get_qdrant()
        embedder = self._get_embedder()

        if client is None or embedder is None:
            return []

        try:
            query_vec = embedder.encode(query).tolist()
            results   = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_vec,
                limit=top_k,
            )
            return [(r.id, r.score) for r in results]
        except Exception as e:
            log.warning(f"TrialMatcherAgent Qdrant search failed: {e}")
            return []

    def _keyword_rank(self, query: str, trials: list[dict], top_k: int) -> list[tuple[dict, float]]:
        """Keyword-based ranking fallback."""
        scored = []
        for trial in trials:
            score = _simple_keyword_match(query, trial)
            if score > 0:
                scored.append((trial, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _assess_eligibility(self, plan_dict: dict, trial: dict) -> dict[str, str]:
        """
        Quick rule-based eligibility flag assessment.
        Returns dict of {criterion: status} where status is "eligible"|"check"|"excluded".
        """
        flags = {}

        # Age check
        meta = plan_dict.get("metadata", {}) or {}
        age  = meta.get("age")
        if age and trial.get("min_age"):
            try:
                flags["age"] = "eligible" if int(age) >= int(trial["min_age"]) else "check"
            except (ValueError, TypeError):
                flags["age"] = "check"

        # ECOG PS
        ecog = meta.get("ecog_ps")
        if ecog is not None:
            max_ecog = trial.get("max_ecog_ps", 2)
            try:
                flags["ecog_ps"] = "eligible" if int(ecog) <= int(max_ecog) else "excluded"
            except (ValueError, TypeError):
                flags["ecog_ps"] = "check"

        # Prior therapy lines
        trial_phase = trial.get("phase", "")
        if "III" in trial_phase:
            flags["prior_therapy"] = "check"

        # Biomarker match
        biomarker_focus = trial.get("biomarker_focus", "").lower()
        diag            = plan_dict.get("diagnosis", {})
        primary_lower   = str(diag.get("primary", "")).lower()
        if biomarker_focus and biomarker_focus in primary_lower:
            flags["biomarker_match"] = "eligible"

        return flags

    def find_matching(self, management_plan, top_k: int = 5) -> list[TrialMatch]:
        """
        Find top-K matching clinical trials for a completed ManagementPlan.

        Args:
            management_plan: ManagementPlan from oncologist.py.
            top_k:           Maximum number of trials to return.

        Returns:
            List of TrialMatch objects, sorted by eligibility_score descending.
        """
        if not self._trials:
            log.warning("TrialMatcherAgent: no trials loaded — returning empty list")
            return []

        plan_dict = management_plan.to_dict() if hasattr(management_plan, "to_dict") else {}
        query     = _build_query_from_plan(management_plan)

        if not query.strip():
            log.warning("TrialMatcherAgent: empty query — no trials matched")
            return []

        # Try Qdrant semantic search first
        semantic_results = self._semantic_search(query, top_k * 2)
        trial_id_map     = {t.get("nct_id", t.get("trial_id", "")): t for t in self._trials}

        if semantic_results:
            # Map IDs back to trial dicts
            scored_trials = []
            for trial_id, score in semantic_results:
                trial = trial_id_map.get(str(trial_id))
                if trial:
                    scored_trials.append((trial, score))
        else:
            # Keyword fallback
            scored_trials = self._keyword_rank(query, self._trials, top_k * 2)

        # Build TrialMatch objects
        matches: list[TrialMatch] = []
        for trial, score in scored_trials[:top_k]:
            eligibility_flags = self._assess_eligibility(plan_dict, trial)
            matches.append(TrialMatch(
                trial_id=trial.get("trial_id", trial.get("nct_id", "")),
                title=trial.get("title", "Unknown trial"),
                phase=trial.get("phase", "Unknown"),
                cancer_type=trial.get("cancer_type", ""),
                biomarker_focus=trial.get("biomarker_focus", ""),
                eligibility_score=float(score),
                eligibility_flags=eligibility_flags,
                nct_id=trial.get("nct_id", ""),
                study_status=trial.get("study_status", "Unknown"),
                brief_summary=trial.get("brief_summary", ""),
                inclusion_snippet=trial.get("inclusion_snippet", ""),
                exclusion_snippet=trial.get("exclusion_snippet", ""),
                contact_info=trial.get("contact_info", ""),
            ))

        matches.sort(key=lambda m: m.eligibility_score, reverse=True)
        log.info(f"TrialMatcherAgent: {len(matches)} trials matched for query: {query[:60]}")
        return matches

    def index_trials(self, force: bool = False) -> int:
        """
        Index the trials snapshot into Qdrant under QDRANT_COLLECTION.
        Run once after loading the trials snapshot.

        Returns:
            Number of trials indexed.
        """
        client   = self._get_qdrant()
        embedder = self._get_embedder()

        if client is None or embedder is None:
            log.warning("TrialMatcherAgent: cannot index — Qdrant or embedder unavailable")
            return 0

        if not self._trials:
            return 0

        try:
            from qdrant_client.models import Distance, VectorParams, PointStruct

            # Create collection if it doesn't exist
            collections = [c.name for c in client.get_collections().collections]
            if QDRANT_COLLECTION not in collections or force:
                dim = embedder.get_sentence_embedding_dimension()
                client.recreate_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )

            # Build and upsert vectors
            points = []
            for i, trial in enumerate(self._trials):
                text = " ".join([
                    trial.get("title", ""),
                    trial.get("cancer_type", ""),
                    trial.get("biomarker_focus", ""),
                    trial.get("brief_summary", ""),
                    trial.get("inclusion_snippet", ""),
                ])
                vec = embedder.encode(text).tolist()
                points.append(PointStruct(
                    id=i,
                    vector=vec,
                    payload={"nct_id": trial.get("nct_id", str(i))},
                ))

            client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            log.info(f"TrialMatcherAgent: indexed {len(points)} trials into Qdrant/{QDRANT_COLLECTION}")
            return len(points)

        except Exception as e:
            log.error(f"TrialMatcherAgent indexing failed: {e}")
            return 0
