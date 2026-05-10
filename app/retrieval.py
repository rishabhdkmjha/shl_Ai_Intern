"""
retrieval.py
Builds and queries a FAISS index over the SHL catalog.

Design decisions:
- Uses cosine similarity (inner product on L2-normalised vectors).
- Over-retrieves (k=20) then post-filters so hard filters never empty results.
- Model: all-MiniLM-L6-v2 — 80 MB, fast cold start, good semantic quality.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

import faiss
from transformers import AutoTokenizer, AutoModel
import torch

from app.catalog import load_catalog

INDEX_DIR = Path("data/faiss_index")
INDEX_FILE = INDEX_DIR / "catalog.index"
ITEMS_FILE = INDEX_DIR / "catalog_items.json"

MODEL_NAME = "all-MiniLM-L6-v2"

# Map full job level strings to normalised slugs for filter matching
LEVEL_ALIASES: dict[str, list[str]] = {
    "entry": ["Entry-Level", "Entry Level"],
    "graduate": ["Graduate"],
    "mid": ["Mid-Professional", "Professional Individual Contributor"],
    "manager": ["Manager", "Front Line Manager", "Supervisor"],
    "director": ["Director", "Executive"],
    "general": ["General Population"],
}


def _normalise_level(raw: str) -> Optional[str]:
    """Map a user-supplied seniority string to a canonical slug."""
    raw_lower = raw.lower()
    for slug, aliases in LEVEL_ALIASES.items():
        if slug in raw_lower:
            return slug
        if any(a.lower() in raw_lower for a in aliases):
            return slug
    return None


class CatalogRetriever:
    """
    Wraps the FAISS index and exposes a single .search() method.
    Instantiate once at app startup; reuse across requests.
    """

    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.items: list[dict] = []

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def build_and_save(self, catalog_path: Optional[Path] = None) -> None:
        """
        Build FAISS index from the catalog and persist it to disk.
        Run once with: python -m scripts.build_index
        """
        self._ensure_model()
        self.items = load_catalog(catalog_path)

        texts = [item["searchable_text"] for item in self.items]
        print(f"[build] Encoding {len(texts)} catalog items …")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,  # Required for cosine via IndexFlatIP
            batch_size=64,
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.index.add(embeddings.astype(np.float32))

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_FILE))
        with open(ITEMS_FILE, "w", encoding="utf-8") as fh:
            json.dump(self.items, fh, ensure_ascii=False, indent=2)

        print(f"[build] Index saved to {INDEX_DIR}/ ({len(self.items)} items)")

    def load(self) -> None:
        """
        Load pre-built index from disk.
        Called once inside the FastAPI lifespan handler.
        """
        if not INDEX_FILE.exists() or not ITEMS_FILE.exists():
            raise RuntimeError(
                "FAISS index not found. Run: python -m scripts.build_index"
            )

        self._ensure_model()
        self.index = faiss.read_index(str(INDEX_FILE))
        with open(ITEMS_FILE, encoding="utf-8") as fh:
            self.items = json.load(fh)

        print(f"[retriever] Loaded index with {len(self.items)} items.")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 10,
        job_level: Optional[str] = None,
        type_filter: Optional[str] = None,
        require_remote: bool = False,
    ) -> list[dict]:
        """
        Semantic search over the catalog with optional hard filters.

        Strategy: retrieve 3×k candidates semantically, apply filters
        (gracefully degrading if a filter would empty results), then
        return the top k.

        Args:
            query:          Natural language query built from conversation.
            k:              Maximum items to return (1-10).
            job_level:      Raw seniority string from the user ("manager",
                            "mid-level", etc.). Normalised internally.
            type_filter:    Single-letter test_type code ("K", "P", …).
            require_remote: If True, filter to remote-available items only.

        Returns:
            List of catalog item dicts, each with an added '_score' field.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call .load() first.")

        oversample = min(k * 3, len(self.items))
        query_vec = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, oversample)

        candidates: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = self.items[idx].copy()
            item["_score"] = float(score)
            candidates.append(item)

        # --- Hard filter: remote ---
        if require_remote:
            filtered = [c for c in candidates if c["remote"] == "yes"]
            if filtered:
                candidates = filtered

        # --- Soft filter: job level ---
        if job_level:
            slug = _normalise_level(job_level)
            if slug:
                target_levels = LEVEL_ALIASES[slug]
                filtered = [
                    c for c in candidates
                    if any(tl in c["job_levels"] for tl in target_levels)
                ]
                if filtered:  # only apply if it doesn't empty results
                    candidates = filtered

        # --- Soft filter: test type ---
        if type_filter:
            filtered = [c for c in candidates if c["test_type"] == type_filter]
            if filtered:
                candidates = filtered

        return candidates[:k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self.model is None:
            print(f"[retriever] Loading embedding model '{MODEL_NAME}' …")
            self.model = SentenceTransformer(MODEL_NAME)

    def get_item_by_name(self, name: str) -> Optional[dict]:
        """Exact (case-insensitive) name lookup. Used for compare queries."""
        name_lower = name.lower().strip()
        for item in self.items:
            if item["name"].lower().strip() == name_lower:
                return item
        # Partial match fallback
        for item in self.items:
            if name_lower in item["name"].lower():
                return item
        return None
