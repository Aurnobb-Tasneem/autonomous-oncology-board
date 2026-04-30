#!/usr/bin/env python3
"""
scripts/index_corpus.py
========================
One-time script: index the oncology corpus into Qdrant.

Run this inside the Docker container BEFORE starting the API:
  python scripts/index_corpus.py [--force]

Arguments:
  --force    Drop and recreate the collection (re-index from scratch)
"""
import argparse
import logging
import sys
from pathlib import Path

# Make sure the package root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from ml.rag.corpus_indexer import index_corpus, CORPUS_DIR, QDRANT_PATH, COLLECTION_NAME


def main():
    parser = argparse.ArgumentParser(description="Index oncology corpus into Qdrant.")
    parser.add_argument("--force", action="store_true", help="Re-index from scratch")
    args = parser.parse_args()

    print(f"Corpus dir : {CORPUS_DIR}")
    print(f"Qdrant path: {QDRANT_PATH}")
    print(f"Collection : {COLLECTION_NAME}")
    print()

    n = index_corpus(force_reindex=args.force)
    print(f"\n✅ Done — {n} vectors indexed into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
