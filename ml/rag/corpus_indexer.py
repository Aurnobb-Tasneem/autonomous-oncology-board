"""
ml/rag/corpus_indexer.py
=========================
One-time script: index the local oncology corpus into Qdrant.

Run this ONCE on the MI300X instance before starting the backend:
  python scripts/index_corpus.py

Input:  ml/rag/corpus/ directory — PDF and .txt files
Output: Qdrant collection at data/qdrant/oncology_corpus

The indexer:
  1. Reads all .txt / .md / .pdf files in corpus/
  2. Chunks them into 512-token passages with 50-token overlap
  3. Embeds each chunk with sentence-transformers/all-MiniLM-L6-v2
  4. Upserts into Qdrant with metadata (title, source, citation)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CORPUS_DIR      = Path(__file__).parent / "corpus"
QDRANT_PATH     = Path(os.getenv("QDRANT_PATH", "/workspace/aob/data/qdrant"))
COLLECTION_NAME = "oncology_corpus"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM      = 384
CHUNK_SIZE      = 512    # characters per chunk (approximate)
CHUNK_OVERLAP   = 80     # character overlap between consecutive chunks
BATCH_SIZE      = 64     # documents per upsert batch


# ── Text chunking ─────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 50]  # skip tiny fragments


# ── Document loader ───────────────────────────────────────────────────────────
def load_corpus_documents(corpus_dir: Path) -> Iterator[dict]:
    """
    Load documents from the corpus directory.

    Expected format — each .txt or .md file:
      Line 1: TITLE: <document title>
      Line 2: SOURCE: <source name>
      Line 3: CITATION: <full citation>
      Line 4+: document content

    PDF support requires pypdf (optional dependency).
    """
    if not corpus_dir.exists():
        log.warning(f"Corpus directory not found: {corpus_dir}")
        log.info("Creating example corpus with seed documents...")
        corpus_dir.mkdir(parents=True, exist_ok=True)
        _create_seed_corpus(corpus_dir)

    for fpath in sorted(corpus_dir.glob("**/*")):
        if fpath.suffix.lower() in (".txt", ".md"):
            yield from _load_text_document(fpath)
        elif fpath.suffix.lower() == ".pdf":
            yield from _load_pdf_document(fpath)


def _load_text_document(fpath: Path) -> Iterator[dict]:
    """Parse a structured .txt corpus document."""
    try:
        text = fpath.read_text(encoding="utf-8")
        lines = text.splitlines()

        # Extract metadata from header lines
        title, source, citation = fpath.stem, str(fpath), ""
        content_start = 0

        for i, line in enumerate(lines[:5]):
            if line.startswith("TITLE:"):
                title = line.removeprefix("TITLE:").strip()
                content_start = i + 1
            elif line.startswith("SOURCE:"):
                source = line.removeprefix("SOURCE:").strip()
                content_start = i + 1
            elif line.startswith("CITATION:"):
                citation = line.removeprefix("CITATION:").strip()
                content_start = i + 1

        content = "\n".join(lines[content_start:]).strip()
        for chunk in chunk_text(content):
            yield {
                "id": str(uuid.uuid4()),
                "title": title,
                "content": chunk,
                "source": source,
                "citation": citation,
                "file": str(fpath.name),
            }
    except Exception as e:
        log.warning(f"Failed to load {fpath}: {e}")


def _load_pdf_document(fpath: Path) -> Iterator[dict]:
    """Load and chunk a PDF file (requires pypdf)."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(fpath))
        full_text = " ".join(page.extract_text() or "" for page in reader.pages)
        for chunk in chunk_text(full_text):
            yield {
                "id": str(uuid.uuid4()),
                "title": fpath.stem,
                "content": chunk,
                "source": str(fpath.name),
                "citation": fpath.stem,
                "file": str(fpath.name),
            }
    except ImportError:
        log.warning("pypdf not installed — skipping PDF. pip install pypdf")
    except Exception as e:
        log.warning(f"Failed to load PDF {fpath}: {e}")


def _create_seed_corpus(corpus_dir: Path):
    """Create seed corpus documents for demo when no real corpus is available."""
    seed_docs = [
        {
            "filename": "nccn_nsclc_2024.txt",
            "content": """TITLE: NCCN Clinical Practice Guidelines — Non-Small Cell Lung Cancer
SOURCE: NCCN Guidelines NSCLC 2024
CITATION: National Comprehensive Cancer Network. NSCLC Guidelines v4.2024.

Non-small cell lung cancer (NSCLC) comprises approximately 85% of all lung cancers.
The two major subtypes are adenocarcinoma and squamous cell carcinoma.

For stage IV adenocarcinoma, molecular testing is mandatory: EGFR, ALK, ROS1, BRAF, MET, RET, NTRK, PD-L1.
First-line osimertinib for EGFR-mutant disease. Alectinib for ALK-positive disease.
Pembrolizumab monotherapy for PD-L1 TPS ≥50% without driver mutations.

For squamous cell carcinoma, pembrolizumab + carboplatin + paclitaxel is first-line standard.
Molecular testing still recommended to exclude rare EGFR mutations.

TNM Staging: Stage I (T1-2, N0), Stage II (T1-2 N1, T3 N0), Stage III (locally advanced), Stage IV (metastatic).
Median overall survival: Stage IV NSCLC approx 12-16 months with immunotherapy combinations.
""",
        },
        {
            "filename": "nccn_colorectal_2024.txt",
            "content": """TITLE: NCCN Clinical Practice Guidelines — Colon Cancer
SOURCE: NCCN Guidelines Colon Cancer 2024
CITATION: National Comprehensive Cancer Network. Colon Cancer Guidelines v3.2024.

Colorectal cancer is the third most common cancer worldwide.
Adenocarcinoma accounts for >95% of colorectal cancers.

Staging: TNM system. Stage I (T1-2 N0), Stage II (T3-4 N0), Stage III (any T N1-2), Stage IV (metastatic).

Adjuvant therapy: FOLFOX recommended for stage III colon cancer. No benefit for stage II low-risk.
MSI/MMR testing mandatory for all newly diagnosed CRC.
MSI-High: pembrolizumab first-line for metastatic disease (KEYNOTE-177).
KRAS/NRAS/BRAF testing required before anti-EGFR therapy.

Metastatic CRC: FOLFOX or FOLFIRI backbone +/- bevacizumab or anti-EGFR (if RAS/RAF wild-type).
Median OS: approximately 30 months with modern triplet chemotherapy + targeted agents.
""",
        },
        {
            "filename": "tcga_lung_adenocarcinoma.txt",
            "content": """TITLE: TCGA Comprehensive Molecular Profiling of Lung Adenocarcinoma
SOURCE: TCGA Network, Nature 2014
CITATION: Cancer Genome Atlas Research Network. Nature 2014;511:543-550.

Analysis of 230 lung adenocarcinomas identified major driver alterations:
- KRAS mutations: 33%
- EGFR mutations: 14%
- ALK fusions: 4%
- ROS1 fusions: 2%
- MET amplification: 2%
- BRAF mutations: 7%

Histological subtypes correlated with molecular alterations.
Lepidic predominant showed better prognosis vs solid/micropapillary.
STK11/KEAP1 co-mutations associated with resistance to immunotherapy.
Tumor mutational burden correlates with smoking history.

Copy number alterations: frequent amplification of NKX2-1 (43%), MYC (26%).
TP53 mutations present in 46% of cases.
""",
        },
    ]

    for doc in seed_docs:
        fpath = corpus_dir / doc["filename"]
        fpath.write_text(doc["content"], encoding="utf-8")
        log.info(f"Created seed corpus document: {fpath.name}")


# ── Qdrant indexing ───────────────────────────────────────────────────────────
def index_corpus(
    corpus_dir: Path = CORPUS_DIR,
    qdrant_path: Path = QDRANT_PATH,
    collection: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
    force_reindex: bool = False,
) -> int:
    """
    Index all corpus documents into Qdrant.

    Args:
        corpus_dir:      Directory containing .txt, .md, .pdf files.
        qdrant_path:     Where to persist Qdrant data.
        collection:      Qdrant collection name.
        embedding_model: sentence-transformers model name.
        force_reindex:   If True, drop and recreate the collection.

    Returns:
        Total number of vectors indexed.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sentence_transformers import SentenceTransformer

    log.info(f"Indexer: starting — corpus={corpus_dir}, qdrant={qdrant_path}")

    # ── Qdrant setup ─────────────────────────────────────────────────────────
    qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(qdrant_path))

    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        if force_reindex:
            log.info(f"Indexer: dropping existing collection '{collection}'")
            client.delete_collection(collection)
        else:
            info = client.get_collection(collection)
            n = info.points_count
            log.info(f"Indexer: collection '{collection}' already exists ({n} vectors). Use force_reindex=True to rebuild.")
            return n

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    log.info(f"Indexer: created collection '{collection}' (dim={VECTOR_DIM})")

    # ── Load embedder ────────────────────────────────────────────────────────
    log.info(f"Indexer: loading embedder ({embedding_model}) ...")
    embedder = SentenceTransformer(embedding_model)

    # ── Index documents ──────────────────────────────────────────────────────
    documents = list(load_corpus_documents(corpus_dir))
    log.info(f"Indexer: {len(documents)} chunks to index")

    total = 0
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        texts = [d["content"] for d in batch]
        vectors = embedder.encode(texts, show_progress_bar=False).tolist()

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={k: v for k, v in doc.items() if k != "id"},
            )
            for doc, vec in zip(batch, vectors)
        ]
        client.upsert(collection_name=collection, points=points)
        total += len(points)
        log.info(f"Indexer: {total}/{len(documents)} vectors indexed")

    log.info(f"Indexer: ✅ done — {total} vectors in '{collection}'")
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    n = index_corpus()
    print(f"\n✅ Indexed {n} document chunks into Qdrant.")
