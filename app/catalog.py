"""
catalog.py
Loads the raw SHL catalog JSON, enriches each entry with a searchable
text blob and a single-letter test_type code, and returns a clean list
of dicts that the retriever and agent can consume directly.
"""

import json
from pathlib import Path
from typing import Optional

CATALOG_PATH = Path("data/catalog_raw.json")

# Map raw key strings → single-letter test_type codes used in API responses
KEY_TO_TYPE: dict[str, str] = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Simulations": "S",
    "Competencies": "C",
    "Biodata & Situational Judgment": "B",
    "Assessment Exercises": "E",
    "Development & 360": "D",
}

# Priority order when an item has multiple keys (most specific wins)
TYPE_PRIORITY = ["S", "A", "K", "P", "C", "B", "E", "D"]


def derive_primary_type(keys: list[str]) -> str:
    """
    Pick the most specific test_type when an item has multiple keys.
    Falls back to 'K' if nothing matches.
    """
    type_set = {KEY_TO_TYPE.get(k, "K") for k in keys}
    for p in TYPE_PRIORITY:
        if p in type_set:
            return p
    return "K"


def build_searchable_text(item: dict) -> str:
    """
    Construct a rich text blob for embedding.
    This is the single most important function for Recall@10 quality —
    every field that a recruiter might implicitly reference is included.
    """
    parts: list[str] = []

    # Product name is the strongest signal
    parts.append(f"Assessment: {item['name']}")

    # Description contains the real semantic content
    desc = item.get("description", "").strip()
    if desc:
        parts.append(desc)

    # Job levels let us match seniority queries ("entry-level developer")
    levels = item.get("job_levels", [])
    if levels:
        parts.append("Suitable for: " + ", ".join(levels))

    # Keys let us match category queries ("personality test", "simulation")
    keys = item.get("keys", [])
    if keys:
        parts.append("Test type / category: " + ", ".join(keys))

    # Duration helps answer "quick screening" style queries
    duration = item.get("duration", "")
    if duration and duration not in ("", "Variable", "Untimed", "N/A", "-", "TBC"):
        parts.append(f"Duration: approximately {duration} minutes")

    # Adaptive flag is relevant for "adaptive testing" queries
    if item.get("adaptive") == "yes":
        parts.append("This assessment uses adaptive testing.")

    # Remote availability
    if item.get("remote") == "yes":
        parts.append("Available for remote online testing.")

    # Languages (relevant for multinational hiring queries)
    langs = item.get("languages", [])
    if langs:
        parts.append("Available in: " + ", ".join(langs[:5]))  # cap at 5

    return " | ".join(parts)


def load_catalog(path: Optional[Path] = None) -> list[dict]:
    """
    Load raw catalog JSON, enrich each item, return processed list.
    Each dict contains everything the agent and retriever need at runtime.

    Args:
        path: Override the default catalog file path (useful for testing).

    Returns:
        List of enriched catalog item dicts.
    """
    catalog_path = path or CATALOG_PATH

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog file not found at {catalog_path}. "
            "Place catalog_raw.json in the data/ directory."
        )

    raw: list[dict] = json.loads(catalog_path.read_text(encoding="utf-8"))
    processed: list[dict] = []

    for item in raw:
        # Skip malformed entries
        if not item.get("name") or not item.get("link"):
            continue

        # Skip entries with status != "ok" if the field exists
        if item.get("status") and item["status"] != "ok":
            continue

        enriched = {
            "entity_id": str(item.get("entity_id", "")),
            "name": item["name"].strip(),
            "url": item["link"].strip(),
            "description": item.get("description", "").strip(),
            "job_levels": item.get("job_levels", []),
            "keys": item.get("keys", []),
            "duration": item.get("duration", ""),
            "remote": item.get("remote", "no"),
            "adaptive": item.get("adaptive", "no"),
            "languages": item.get("languages", []),
            # Derived fields
            "test_type": derive_primary_type(item.get("keys", [])),
            "searchable_text": build_searchable_text(item),
        }
        processed.append(enriched)

    return processed


def get_catalog_summary(catalog: list[dict]) -> dict:
    """
    Return a brief summary of catalog contents.
    Useful for debugging and the approach document.
    """
    type_counts: dict[str, int] = {}
    for item in catalog:
        t = item["test_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "total_items": len(catalog),
        "by_type": type_counts,
        "adaptive_count": sum(1 for i in catalog if i["adaptive"] == "yes"),
        "remote_count": sum(1 for i in catalog if i["remote"] == "yes"),
    }
