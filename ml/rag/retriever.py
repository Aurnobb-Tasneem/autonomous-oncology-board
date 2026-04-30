"""
ml/rag/retriever.py
===================
Agent 2: Researcher — RAG retrieval pipeline over local oncology corpus.

Uses Qdrant (in-process) + sentence-transformers to retrieve relevant
oncology literature, NCCN guidelines, and TCGA study excerpts.

Workflow:
  1. Take a query (from PathologyReport summary + tissue type)
  2. Embed query with sentence-transformers (MiniLM or similar)
  3. Retrieve top-K documents from Qdrant
  4. Return EvidenceBundle with cited passages

Collection name: "oncology_corpus"
Vector dim: 384 (all-MiniLM-L6-v2) or 768 (mpnet)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Qdrant config ────────────────────────────────────────────────────────────
QDRANT_PATH       = Path(os.getenv("QDRANT_PATH", "/workspace/aob/data/qdrant"))
COLLECTION_NAME   = "oncology_corpus"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, fast
VECTOR_DIM        = 384
TOP_K_DEFAULT     = 5


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class EvidenceDocument:
    doc_id: str
    title: str
    content: str          # relevant passage
    source: str           # e.g. "NCCN Guidelines 2024", "TCGA-LUAD study"
    relevance_score: float
    citation: str         # formatted citation string


@dataclass
class EvidenceBundle:
    query: str
    tissue_type: str
    documents: list[EvidenceDocument]
    n_retrieved: int

    def to_dict(self) -> dict:
        return asdict(self)

    def format_for_llm(self) -> str:
        """
        Format the evidence bundle as a readable string for the LLM prompt.
        """
        lines = [
            f"RETRIEVED EVIDENCE for: {self.query}",
            f"Tissue context: {self.tissue_type}",
            "=" * 60,
        ]
        for i, doc in enumerate(self.documents, 1):
            lines += [
                f"\n[{i}] {doc.title}",
                f"Source: {doc.source}  (relevance: {doc.relevance_score:.2f})",
                f"Citation: {doc.citation}",
                f"Content: {doc.content}",
            ]
        return "\n".join(lines)


# ── Retriever ────────────────────────────────────────────────────────────────
class OncologyRetriever:
    """
    RAG retriever for the Autonomous Oncology Board.

    Wraps a Qdrant local collection indexed by corpus_indexer.py.
    Falls back to a curated mock corpus if Qdrant is not initialised yet.
    """

    def __init__(
        self,
        qdrant_path: Path = QDRANT_PATH,
        collection: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.qdrant_path     = qdrant_path
        self.collection      = collection
        self.embedding_model = embedding_model
        self._client         = None
        self._embedder       = None

    def _ensure_ready(self):
        """Lazy initialisation of Qdrant client and sentence-transformer."""
        if self._client is not None:
            return

        # ── Qdrant client ────────────────────────────────────────────────────
        try:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(path=str(self.qdrant_path))
            cols = [c.name for c in self._client.get_collections().collections]
            if self.collection not in cols:
                log.warning(
                    f"Retriever: collection '{self.collection}' not found in Qdrant. "
                    "Run scripts/index_corpus.py first. Falling back to mock corpus."
                )
                self._client = None  # triggers fallback
        except Exception as e:
            log.warning(f"Retriever: Qdrant init failed ({e}). Using mock corpus fallback.")
            self._client = None

        # ── Sentence-transformer embedder ────────────────────────────────────
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model)
            log.info(f"Retriever: embedder loaded ({self.embedding_model})")
        except Exception as e:
            log.warning(f"Retriever: embedder load failed ({e}). Using fallback.")
            self._embedder = None

    # ── Query ────────────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        tissue_type: str,
        top_k: int = TOP_K_DEFAULT,
    ) -> EvidenceBundle:
        """
        Retrieve the top-K most relevant oncology evidence documents.

        Args:
            query:       The search query (e.g. pathology summary + clinical question).
            tissue_type: Tissue classification from PathologistAgent (for filtering).
            top_k:       Number of documents to retrieve.

        Returns:
            EvidenceBundle with ranked, cited documents.
        """
        self._ensure_ready()

        if self._client is not None and self._embedder is not None:
            docs = self._qdrant_retrieve(query, tissue_type, top_k)
        else:
            log.info("Retriever: using mock corpus (Qdrant not ready)")
            docs = self._mock_retrieve(query, tissue_type, top_k)

        return EvidenceBundle(
            query=query,
            tissue_type=tissue_type,
            documents=docs,
            n_retrieved=len(docs),
        )

    def _qdrant_retrieve(
        self, query: str, tissue_type: str, top_k: int
    ) -> list[EvidenceDocument]:
        """Live retrieval from Qdrant vector database."""
        query_vector = self._embedder.encode(query).tolist()

        results = self._client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

        docs = []
        for hit in results:
            payload = hit.payload or {}
            docs.append(EvidenceDocument(
                doc_id=str(hit.id),
                title=payload.get("title", "Untitled"),
                content=payload.get("content", ""),
                source=payload.get("source", "Unknown"),
                relevance_score=round(hit.score, 4),
                citation=payload.get("citation", ""),
            ))
        return docs

    def _mock_retrieve(
        self, query: str, tissue_type: str, top_k: int
    ) -> list[EvidenceDocument]:
        """
        Hardcoded mock evidence corpus for demo/testing.
        Covers the 5 LC25000 tissue classes with real oncology facts.
        """
        MOCK_CORPUS: dict[str, list[dict]] = {
            "lung_adenocarcinoma": [
                {
                    "title": "EGFR-Targeted Therapy in Lung Adenocarcinoma",
                    "content": (
                        "First-line osimertinib (80mg/day) is recommended for patients with "
                        "EGFR exon 19 deletions or exon 21 L858R mutations. FLAURA trial showed "
                        "median PFS of 18.9 months vs 10.2 months for standard EGFR-TKI."
                    ),
                    "source": "NCCN Guidelines NSCLC 2024",
                    "citation": "NCCN Clinical Practice Guidelines in Oncology: NSCLC v4.2024",
                    "relevance_score": 0.94,
                },
                {
                    "title": "Pembrolizumab Monotherapy in PD-L1 High Lung Cancer",
                    "content": (
                        "Pembrolizumab (200mg Q3W) is first-line standard of care for "
                        "advanced NSCLC with PD-L1 TPS ≥50% and no EGFR/ALK alterations. "
                        "KEYNOTE-024 demonstrated superior OS vs chemotherapy."
                    ),
                    "source": "KEYNOTE-024 Trial, NEJM 2016",
                    "citation": "Reck M, et al. N Engl J Med 2016;375:1823-1833.",
                    "relevance_score": 0.91,
                },
                {
                    "title": "ALK Rearrangement Testing in Lung Adenocarcinoma",
                    "content": (
                        "All advanced lung adenocarcinoma patients should undergo ALK testing. "
                        "Alectinib (600mg BID) is preferred first-line for ALK-positive NSCLC "
                        "over crizotinib. ALEX trial: mPFS 34.8 vs 10.9 months."
                    ),
                    "source": "ALEX Trial, NEJM 2017",
                    "citation": "Peters S, et al. N Engl J Med 2017;377:829-838.",
                    "relevance_score": 0.88,
                },
            ],
            "lung_squamous_cell_carcinoma": [
                {
                    "title": "First-Line Treatment for Squamous NSCLC",
                    "content": (
                        "Pembrolizumab + carboplatin + paclitaxel/nab-paclitaxel is preferred "
                        "first-line for metastatic squamous NSCLC (KEYNOTE-789). "
                        "EGFR/ALK testing still recommended to exclude rare cases."
                    ),
                    "source": "NCCN Guidelines NSCLC 2024",
                    "citation": "NCCN Clinical Practice Guidelines in Oncology: NSCLC v4.2024",
                    "relevance_score": 0.93,
                },
                {
                    "title": "Necitumumab in Squamous Cell Lung Cancer",
                    "content": (
                        "Necitumumab + gemcitabine/cisplatin improved OS in metastatic squamous "
                        "NSCLC (SQUIRE trial). However, pembrolizumab-based regimens are now "
                        "preferred. EGFR expression does not predict necitumumab benefit."
                    ),
                    "source": "SQUIRE Trial, Lancet Oncol 2015",
                    "citation": "Thatcher N, et al. Lancet Oncol 2015;16:763-774.",
                    "relevance_score": 0.82,
                },
            ],
            "colon_adenocarcinoma": [
                {
                    "title": "Adjuvant Chemotherapy for Stage III Colon Cancer",
                    "content": (
                        "FOLFOX (oxaliplatin + leucovorin + 5-FU) is standard adjuvant therapy "
                        "for stage III colon cancer. MOSAIC trial demonstrated significant "
                        "reduction in recurrence risk vs 5-FU/LV alone (HR 0.77)."
                    ),
                    "source": "MOSAIC Trial, NEJM 2004",
                    "citation": "André T, et al. N Engl J Med 2004;350:2343-2351.",
                    "relevance_score": 0.95,
                },
                {
                    "title": "MSI-High Colorectal Cancer Immunotherapy",
                    "content": (
                        "Pembrolizumab monotherapy demonstrated superior PFS vs chemotherapy "
                        "in MSI-H/dMMR metastatic CRC (KEYNOTE-177). MSI testing is mandatory "
                        "for all newly diagnosed CRC per NCCN guidelines."
                    ),
                    "source": "KEYNOTE-177 Trial, NEJM 2020",
                    "citation": "André T, et al. N Engl J Med 2020;383:2207-2218.",
                    "relevance_score": 0.93,
                },
                {
                    "title": "Bevacizumab in Metastatic Colorectal Cancer",
                    "content": (
                        "Bevacizumab added to FOLFIRI or FOLFOX significantly improved OS in "
                        "mCRC (NO16966 trial). Not indicated for adjuvant stage III treatment. "
                        "Requires monitoring for hypertension and proteinuria."
                    ),
                    "source": "NO16966 Trial, JCO 2008",
                    "citation": "Saltz LB, et al. J Clin Oncol 2008;26:2013-2019.",
                    "relevance_score": 0.89,
                },
            ],
            "colon_benign_tissue": [
                {
                    "title": "Colorectal Cancer Screening Guidelines",
                    "content": (
                        "Average-risk adults should begin CRC screening at age 45. "
                        "Colonoscopy every 10 years is preferred. Alternatives include "
                        "annual FIT, stool DNA testing (every 1-3 years), or CT colonography."
                    ),
                    "source": "USPSTF Guidelines 2021",
                    "citation": "US Preventive Services Task Force. JAMA 2021;325:1965-1977.",
                    "relevance_score": 0.72,
                },
            ],
            "lung_benign_tissue": [
                {
                    "title": "Pulmonary Nodule Management Guidelines",
                    "content": (
                        "Solid pulmonary nodules <6mm in low-risk patients require no routine "
                        "follow-up. Nodules 6-8mm: CT at 6-12 months. Nodules >8mm: "
                        "consider PET/CT or tissue sampling per Fleischner Society guidelines."
                    ),
                    "source": "Fleischner Society Guidelines 2017",
                    "citation": "MacMahon H, et al. Radiology 2017;284:228-243.",
                    "relevance_score": 0.76,
                },
            ],
        }

        # Match tissue type — fall back to a generic set if unknown
        candidates = MOCK_CORPUS.get(tissue_type, [])
        if not candidates:
            # Return generic oncology references
            candidates = [
                v[0] for v in MOCK_CORPUS.values() if v
            ]

        # Sort by relevance_score and take top_k
        candidates = sorted(candidates, key=lambda d: d["relevance_score"], reverse=True)[:top_k]

        return [
            EvidenceDocument(
                doc_id=f"mock_{i}",
                title=d["title"],
                content=d["content"],
                source=d["source"],
                relevance_score=d["relevance_score"],
                citation=d["citation"],
            )
            for i, d in enumerate(candidates)
        ]
