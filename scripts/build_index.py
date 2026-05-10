"""
scripts/build_index.py
One-time script to build the FAISS index from the raw catalog JSON.

Usage:
    python -m scripts.build_index

Run this once before starting the server, or whenever the catalog changes.
The index is saved to data/faiss_index/ and loaded at server startup.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval import CatalogRetriever
from app.catalog import load_catalog, get_catalog_summary


def main():
    print("=" * 60)
    print("SHL Assessment Recommender — Index Builder")
    print("=" * 60)

    # Print catalog summary before building
    try:
        catalog = load_catalog()
        summary = get_catalog_summary(catalog)
        print(f"\nCatalog loaded: {summary['total_items']} items")
        print(f"  By type: {summary['by_type']}")
        print(f"  Adaptive: {summary['adaptive_count']}")
        print(f"  Remote:   {summary['remote_count']}\n")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Place your catalog JSON at data/catalog_raw.json and retry.")
        sys.exit(1)

    # Build and save FAISS index
    r = CatalogRetriever()
    r.build_and_save()

    print("\n✓ Index build complete. You can now start the server.")
    print("  Run: uvicorn app.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
