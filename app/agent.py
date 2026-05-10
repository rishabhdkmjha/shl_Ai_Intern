"""
agent.py
Rule-based intent classifier (Decider) that runs before the LLM on every
turn. By fixing the intent outside the LLM we get:
  - Deterministic guardrails (injection never reaches the LLM)
  - Predictable clarify-vs-recommend behaviour
  - No wasted tokens on unhelpful deliberation
"""

import re
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Intent enum
# ---------------------------------------------------------------------------

class AgentAction(Enum):
    CLARIFY = "clarify"       # Ask a follow-up question
    RECOMMEND = "recommend"   # Provide recommendations
    COMPARE = "compare"       # Explain differences between named tests
    REFINE = "refine"         # Update an existing shortlist
    REFUSE = "refuse"         # Off-topic / injection attempt


# ---------------------------------------------------------------------------
# Signal lists (kept as module-level constants for easy tuning)
# ---------------------------------------------------------------------------

INJECTION_PATTERNS = [
    "ignore previous", "ignore all", "forget instructions", "disregard",
    "you are now", "pretend you are", "act as", "new persona",
    "override", "jailbreak", "DAN",
    "legal advice", "hiring law", "discrimination law", "salary",
    "compensation", "benefit", "stock option",
]

COMPARE_SIGNALS = [
    "difference between", "compare", " vs ", " vs.", "versus",
    "better than", "which is better", "what's the difference",
    "how does", "differ from",
]

REFINE_SIGNALS = [
    "actually", "also add", "instead", "without", "remove",
    "change", "update", "add personality", "include more",
    "drop the", "can you add", "also include", "but only",
    "shorter", "longer", "remote only",
]

# Signals that indicate sufficient job / skill context
ROLE_SIGNALS = [
    "developer", "engineer", "manager", "analyst", "sales", "support",
    "data scientist", "tester", "qa", "devops", "accountant", "nurse",
    "recruiter", "hr", "customer service", "cashier", "warehouse",
    "front end", "backend", "full stack", "python", "java", "sql",
    "hiring for", "we are hiring", "looking for", "role is", "position",
    "job description", "jd:", "responsibilities",
]

# Signals that indicate seniority / test-type context
CONTEXT_SIGNALS = [
    "entry level", "entry-level", "graduate", "junior", "mid-level",
    "mid level", "senior", "manager", "director", "executive",
    "years of experience", "years experience",
    "cognitive", "personality", "ability", "aptitude", "simulation",
    "knowledge test", "technical test", "behavioural", "behavioral",
    "coding", "coding test", "verbal", "numerical", "deductive", "inductive",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decide_action(messages: list[dict]) -> AgentAction:
    """
    Classify intent from conversation history WITHOUT calling the LLM.

    Decision tree (evaluated top-to-bottom, first match wins):
      1. Injection / off-topic  → REFUSE
      2. Comparison request     → COMPARE
      3. Refinement of prior rec → REFINE
      4. Enough context         → RECOMMEND
      5. Default                → CLARIFY
    """
    if not messages:
        return AgentAction.CLARIFY

    user_messages = [m for m in messages if m["role"] == "user"]
    if not user_messages:
        return AgentAction.CLARIFY

    last_user: str = user_messages[-1]["content"].lower()
    all_user_text: str = " ".join(m["content"].lower() for m in user_messages)

    # 1. Refuse injection / off-topic
    if _matches_any(last_user, INJECTION_PATTERNS):
        return AgentAction.REFUSE

    # 2. Compare
    if _matches_any(last_user, COMPARE_SIGNALS):
        return AgentAction.COMPARE

    # 3. Refine (only meaningful if prior recommendation exists)
    if _has_prior_recommendation(messages) and _matches_any(last_user, REFINE_SIGNALS):
        return AgentAction.REFINE

    # 4. Recommend when we have enough context
    if _has_enough_context(all_user_text, len(user_messages)):
        return AgentAction.RECOMMEND

    return AgentAction.CLARIFY


def extract_context(messages: list[dict]) -> dict:
    """
    Pull structured constraints out of conversation history.
    Returns a dict with optional keys: job_level, type_filter, require_remote.
    Used by the retriever to apply post-retrieval filters.
    """
    all_user_text = " ".join(
        m["content"].lower() for m in messages if m["role"] == "user"
    )

    return {
        "job_level": _extract_job_level(all_user_text),
        "type_filter": _extract_type_filter(all_user_text),
        "require_remote": "remote" in all_user_text,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(p in text for p in patterns)


def _has_prior_recommendation(messages: list[dict]) -> bool:
    return any(
        m["role"] == "assistant" and "<RECOMMENDATIONS>" in m.get("content", "")
        for m in messages
    )


def _has_enough_context(all_user_text: str, turn_count: int) -> bool:
    """
    Return True if we have:
      - At least one role/skill signal  AND one context signal, OR
      - A job description block (long text with keywords), OR
      - User has sent ≥3 turns (patience threshold — recommend rather than
        keep asking questions)
    """
    role_hits = sum(1 for s in ROLE_SIGNALS if s in all_user_text)
    context_hits = sum(1 for s in CONTEXT_SIGNALS if s in all_user_text)

    # Explicit job description submission
    has_jd = any(kw in all_user_text for kw in ("job description", "jd:", "responsibilities:", "requirements:"))

    if has_jd:
        return True

    if role_hits >= 1 and context_hits >= 1:
        return True

    # After 3 user turns, commit to a recommendation with available context
    if turn_count >= 3:
        return True

    return False


# Level string → canonical slug mappings (mirrors retrieval.py for consistency)
_LEVEL_MAP = {
    "entry": "entry",
    "graduate": "graduate",
    "junior": "entry",
    "mid": "mid",
    "senior": "mid",
    "manager": "manager",
    "supervisor": "manager",
    "director": "director",
    "executive": "director",
    "vp": "director",
    "c-level": "director",
    "ceo": "director",
}

_TYPE_MAP = {
    "cognitive": "A",
    "ability": "A",
    "aptitude": "A",
    "numerical": "A",
    "verbal": "A",
    "deductive": "A",
    "inductive": "A",
    "personality": "P",
    "behavioural": "P",
    "behavioral": "P",
    "opq": "P",
    "motivation": "P",
    "simulation": "S",
    "coding": "S",
    "typing": "S",
    "knowledge": "K",
    "technical": "K",
    "skills test": "K",
    "situational": "B",
    "sjt": "B",
    "competency": "C",
    "competencies": "C",
    "360": "D",
    "development": "D",
}


def _extract_job_level(text: str) -> Optional[str]:
    for keyword, slug in _LEVEL_MAP.items():
        if keyword in text:
            return slug
    # Pattern: "X years" → likely mid-level
    if re.search(r"\b[3-9]\s+years?\b|\b1[0-9]\s+years?\b", text):
        return "mid"
    if re.search(r"\b[1-2]\s+years?\b", text):
        return "entry"
    return None


def _extract_type_filter(text: str) -> Optional[str]:
    for keyword, code in _TYPE_MAP.items():
        if keyword in text:
            return code
    return None
