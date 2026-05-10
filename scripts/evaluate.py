"""
scripts/evaluate.py
Evaluates the running /chat endpoint against the 10 sample conversation
traces. Reports Recall@10 per trace and mean Recall@10 overall.

Usage:
    python -m scripts.evaluate --base-url http://localhost:8000
    python -m scripts.evaluate --base-url https://your-deployed-url.com
    python -m scripts.evaluate --traces-dir traces/

Trace file format (JSON):
{
  "persona": "...",
  "facts": {...},
  "expected_assessments": ["Assessment Name 1", "Assessment Name 2", ...]
}
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def recall_at_k(recommended: list[str], relevant: list[str], k: int = 10) -> float:
    """
    Recall@K = |relevant ∩ top-k recommended| / |relevant|
    Name matching is case-insensitive and strips whitespace.
    """
    if not relevant:
        return 1.0  # vacuously true
    top_k = [r.lower().strip() for r in recommended[:k]]
    relevant_lower = [r.lower().strip() for r in relevant]
    hits = sum(1 for r in relevant_lower if r in top_k)
    return hits / len(relevant_lower)


def run_conversation(base_url: str, messages: list[dict], max_turns: int = 8) -> list[str]:
    """
    Send the conversation to /chat and extract final recommendations.
    Returns list of recommended assessment names.
    """
    url = base_url.rstrip("/") + "/chat"
    recommended_names: list[str] = []

    for _ in range(max_turns):
        payload = {"messages": messages}
        try:
            resp = requests.post(url, json=payload, timeout=35)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"    [ERROR] Request failed: {e}")
            break

        data = resp.json()
        reply = data.get("reply", "")
        recs = data.get("recommendations", [])
        end = data.get("end_of_conversation", False)

        # Append assistant reply to history
        messages.append({"role": "assistant", "content": reply})

        if recs:
            recommended_names = [r["name"] for r in recs]

        if end or recs:
            break

        # Simulate user ending conversation if assistant is done
        # (In real eval, an LLM simulates the user — here we just stop)
        messages.append({"role": "user", "content": "That looks good, thank you."})

    return recommended_names


def load_traces(traces_dir: Path) -> list[dict]:
    """Load all .json trace files from the given directory."""
    traces = []
    for path in sorted(traces_dir.glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            trace = json.load(fh)
            trace["_file"] = path.name
            traces.append(trace)
    return traces


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Recall@10 on sample traces")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--traces-dir", default="traces")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    traces_dir = Path(args.traces_dir)
    if not traces_dir.exists():
        print(f"Traces directory not found: {traces_dir}")
        print("Create traces/ and add .json trace files.")
        sys.exit(1)

    traces = load_traces(traces_dir)
    if not traces:
        print(f"No .json files found in {traces_dir}")
        sys.exit(1)

    # Health check
    try:
        hc = requests.get(args.base_url.rstrip("/") + "/health", timeout=10)
        hc.raise_for_status()
        print(f"✓ Service healthy at {args.base_url}\n")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        sys.exit(1)

    print(f"{'Trace':<35} {'Recommended':>12} {'Relevant':>10} {'Recall@'+str(args.k):>10}")
    print("-" * 70)

    recalls: list[float] = []

    for trace in traces:
        fname = trace["_file"]
        expected = trace.get("expected_assessments", [])

        # Build initial messages from persona + first user message
        persona = trace.get("persona", "")
        first_message = trace.get("first_message", persona[:200] if persona else "I need an assessment.")
        messages = [{"role": "user", "content": first_message}]

        recommended = run_conversation(args.base_url, messages)

        r_at_k = recall_at_k(recommended, expected, k=args.k)
        recalls.append(r_at_k)

        print(f"{fname:<35} {len(recommended):>12} {len(expected):>10} {r_at_k:>10.2f}")

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    print("-" * 70)
    print(f"{'Mean Recall@'+str(args.k):<57} {mean_recall:>10.4f}")
    print(f"\nEvaluated {len(traces)} traces.")


if __name__ == "__main__":
    main()
