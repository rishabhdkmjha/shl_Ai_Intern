import json
import numpy as np
from pathlib import Path
from typing import Optional
import faiss
from app.catalog import load_catalog

INDEX_DIR = Path("data/faiss_index")
INDEX_FILE = INDEX_DIR / "catalog.index"
ITEMS_FILE = INDEX_DIR / "catalog_items.json"

LEVEL_ALIASES: dict[str, list[str]] = {
    "entry": ["Entry-Level", "Entry Level"],
    "graduate": ["Graduate"],
    "mid": ["Mid-Professional", "Professional Individual Contributor"],
    "manager": ["Manager", "Front Line Manager", "Supervisor"],
    "director": ["Director", "Executive"],
    "general": ["General Population"],
}

def _normalise_level(raw: str) -> Optional[str]:
    raw_lower = raw.lower()
    for slug, aliases in LEVEL_ALIASES.items():
        if slug in raw_lower:
            return slug
        if any(a.lower() in raw_lower for a in aliases):
            return slug
    return None

def _get_embedding(text: str) -> np.ndarray:
    """Lightweight embedding using TF-IDF style hashing - no torch needed."""
    import hashlib
    dim = 384
    vec = np.zeros(dim, dtype=np.float32)
    words = text.lower().split()
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

class CatalogRetriever:
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.items: list[dict] = []

    def build_and_save(self, catalog_path: Optional[Path] = None) -> None:
        self.items = load_catalog(catalog_path)
        texts = [item["searchable_text"] for item in self.items]
        print(f"[build] Encoding {len(texts)} catalog items ...")
        embeddings = np.array([_get_embedding(t) for t in texts])
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_FILE))
        with open(ITEMS_FILE, "w", encoding="utf-8") as fh:
            json.dump(self.items, fh, ensure_ascii=False, indent=2)
        print(f"[build] Index saved ({len(self.items)} items)")

    def load(self) -> None:
        if not INDEX_FILE.exists() or not ITEMS_FILE.exists():
            raise RuntimeError("FAISS index not found. Run: python -m scripts.build_index")
        self.index = faiss.read_index(str(INDEX_FILE))
        with open(ITEMS_FILE, encoding="utf-8") as fh:
            self.items = json.load(fh)
        print(f"[retriever] Loaded index with {len(self.items)} items.")

    def search(self, query: str, k: int = 10, job_level=None, type_filter=None, require_remote=False) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Index not loaded.")
        oversample = min(k * 3, len(self.items))
        query_vec = _get_embedding(query).reshape(1, -1)
        scores, indices = self.index.search(query_vec, oversample)
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = self.items[idx].copy()
            item["_score"] = float(score)
            candidates.append(item)
        if require_remote:
            filtered = [c for c in candidates if c["remote"] == "yes"]
            if filtered:
                candidates = filtered
        if job_level:
            slug = _normalise_level(job_level)
            if slug:
                target_levels = LEVEL_ALIASES[slug]
                filtered = [c for c in candidates if any(tl in c["job_levels"] for tl in target_levels)]
                if filtered:
                    candidates = filtered
        if type_filter:
            filtered = [c for c in candidates if c["test_type"] == type_filter]
            if filtered:
                candidates = filtered
        return candidates[:k]

    def get_item_by_name(self, name: str) -> Optional[dict]:
        name_lower = name.lower().strip()
        for item in self.items:
            if item["name"].lower().strip() == name_lower:
                return item
        for item in self.items:
            if name_lower in item["name"].lower():
                return item
        return None
