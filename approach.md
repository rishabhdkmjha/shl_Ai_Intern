# SHL Assessment Recommender — Approach Document

## System Overview

A stateless FastAPI service backed by FAISS semantic retrieval and Claude (Sonnet)
for natural-language generation. The agent guides a recruiter from a vague intent to
a grounded shortlist of 1–10 SHL assessments through multi-turn dialogue.

---

## Design Choices

### 1. Data Strategy

The raw catalog JSON is enriched into a `searchable_text` blob per item that
concatenates: assessment name, full description, job levels, test categories,
duration, adaptive flag, and remote availability. This single rich string is what
gets embedded — not just the name or description in isolation. The rationale: a
query like "quick personality test for entry-level call centre staff" contains
multiple implicit signals (personality → type filter, entry-level → job level
filter, quick → duration preference) that all need to be present in the indexed
text for semantic retrieval to work.

Each item also carries a `test_type` code (K/P/A/S/C/B/E/D) derived from its
`keys` field using a priority ordering. This single character feeds the API schema
directly and avoids re-deriving it at response time.

### 2. Retrieval Setup

- **Model**: `all-MiniLM-L6-v2` (80 MB, ~50 ms encode on CPU). Chosen for cold-start
  friendliness on free-tier platforms (Render, Railway) while staying within the
  30-second timeout. Larger models (bge-large-en-v1.5) improved Recall@10 by ~4%
  in offline experiments but caused cold-start timeouts.
- **Index**: `faiss.IndexFlatIP` (exact cosine similarity on L2-normalised vectors).
  No approximate index (HNSW) needed at ~400 items.
- **Over-retrieval**: Retrieve 3×k (default k=10, so fetch 30 candidates), apply
  post-retrieval hard filters (job level, test type, remote), then truncate to k.
  This prevents filters from emptying results when semantic ranking and hard
  constraints pull in different directions.

### 3. Prompt Design

The system prompt is injected fresh on every turn with the retrieved catalog slice
formatted as a compact bullet list (name, type code, URL, level tags, 150-char
description snippet). The LLM sees only the retrieved items, not the full catalog.
This keeps the context window small, prevents the LLM from hallucinating catalog
items it was trained on but that are not in the current slice, and ensures every
URL in the output came from our data.

Recommendations are requested inside `<RECOMMENDATIONS>[...]</RECOMMENDATIONS>` XML
tags rather than via function/tool calling. Reason: in testing, tool call JSON was
occasionally malformed on the first turn; tag-delimited regex parsing is
deterministic and does not depend on the model correctly invoking a tool schema.

### 4. Agent (Decider) Design

Intent classification runs **before** the LLM on every turn. A rule-based
`decide_action()` function checks for injection patterns, comparison signals,
refinement signals, and sufficient context signals in that priority order. The LLM
only handles natural-language generation within the intent the Decider has already
fixed. Benefits:

- Guardrails are deterministic — injection can never reach the generation step.
- Clarify-vs-recommend is predictable regardless of LLM temperature.
- No wasted tokens on "should I ask a question?" deliberation.

The "sufficient context" heuristic requires at least one role/skill signal AND one
seniority/type signal, OR detects a job-description block, OR fires after 3 user
turns regardless (prevents the agent from asking questions indefinitely).

### 5. Schema Compliance

A post-processing step validates every item in the LLM's `<RECOMMENDATIONS>` JSON
against the retrieved catalog URL allowlist. Items with URLs not in the allowlist
are looked up by name; if still not found, they are silently dropped. This is the
last line of defence against hallucinated URLs. The `end_of_conversation` flag is
set in the API layer (not by the LLM) — it's True when recommendations are present
and the intent was RECOMMEND or REFINE.

---

## Evaluation Approach

Offline Recall@10 was measured by replaying each of the 10 public traces against
the live `/chat` endpoint using a scripted user (`scripts/evaluate.py`). The
simulated user submits the trace's `first_message`, then accepts the first
shortlist the agent returns.

**Key findings from iteration:**
- Using only the description for embedding gave Recall@10 ≈ 0.48.
- Adding job levels and keys to `searchable_text` lifted it to ≈ 0.67.
- The over-retrieve-then-filter pattern added ~0.05 over exact-k retrieval.

**Failure modes caught during testing:**
- LLM recommended on turn 1 for vague queries → fixed by Decider CLARIFY check.
- Hallucinated OPQ32 URL with slightly wrong path → fixed by URL allowlist check.
- Agent looped on clarification for JD-paste queries → fixed by JD detection in
  `_has_enough_context`.

---

## Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| API | FastAPI + Pydantic v2 | Schema validation, async-ready |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 | Fast CPU inference |
| Vector store | FAISS (faiss-cpu) | Zero infra, deterministic |
| LLM | Anthropic Claude Sonnet | Strong instruction-following |
| Deployment | Render / Railway / Fly.io | Free tier, cold-start < 2 min |

AI tools used: Claude (Anthropic) for code review and prompt iteration drafts.
All design decisions were made and understood independently.
