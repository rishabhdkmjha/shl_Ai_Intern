"""
main.py
FastAPI application exposing:
  GET  /health  →  {"status": "ok"}
  POST /chat    →  {"reply": str, "recommendations": [...], "end_of_conversation": bool}

Design notes:
- Stateless: full conversation history sent on every request.
- Index loaded once at startup via lifespan handler.
- LLM call wrapped in try/except so timeouts return a safe fallback.
- Schema compliance enforced by Pydantic response_model.
"""

import json
import os
import re
from contextlib import asynccontextmanager
from typing import Optional

from groq import Groq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from app.agent import AgentAction, decide_action, extract_context
from app.prompts import REFUSE_PROMPT, SYSTEM_PROMPT, build_query_from_history, format_catalog_context
from app.retrieval import CatalogRetriever

# ---------------------------------------------------------------------------
# Globals (initialised in lifespan)
# ---------------------------------------------------------------------------

retriever = CatalogRetriever()
llm_client: Optional[Groq] = None

MAX_TURNS = 8
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Lifespan: load index once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client
    retriever.load()
    llm_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("[startup] Retriever and Groq LLM client ready.")
    yield
    print("[shutdown] Goodbye.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("messages list cannot be empty")
        return v


class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[RecommendationItem]
    end_of_conversation: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Readiness probe. Returns 200 when the service is up."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Stateless conversational endpoint.
    Full conversation history must be provided on every call.
    """
    messages = [m.model_dump() for m in request.messages]

    # --- Turn cap ---
    user_turns = sum(1 for m in messages if m["role"] == "user")
    if user_turns > MAX_TURNS:
        return ChatResponse(
            reply=(
                "We've reached the maximum conversation length. "
                "Please start a new session to continue."
            ),
            recommendations=[],
            end_of_conversation=True,
        )

    # --- Decide intent ---
    action = decide_action(messages)

    # --- Build retrieval query and fetch catalog context ---
    catalog_items: list[dict] = []

    if action in (AgentAction.RECOMMEND, AgentAction.COMPARE, AgentAction.REFINE):
        context = extract_context(messages)
        query = build_query_from_history(messages)
        catalog_items = retriever.search(
            query=query,
            k=20,
            job_level=context.get("job_level"),
            type_filter=None,  # let LLM pick across types for better Recall@10
            require_remote=context.get("require_remote", False),
        )
    elif action == AgentAction.CLARIFY:
        # Light retrieval so the LLM can mention relevant areas without recommending
        query = build_query_from_history(messages)
        catalog_items = retriever.search(query=query, k=5)

    catalog_context = format_catalog_context(catalog_items)

    # --- Build system prompt ---
    if action == AgentAction.REFUSE:
        system = REFUSE_PROMPT
    else:
        system = SYSTEM_PROMPT.replace("{catalog_context}", catalog_context)

    # --- Call Groq ---
    try:
        groq_messages = [{"role": "system", "content": system}] + messages
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            messages=groq_messages,
            temperature=0.3,
        )
        raw_reply: str = response.choices[0].message.content
    except Exception as exc:
        print(f"[LLM ERROR] {exc}")
        return ChatResponse(
            reply="Sorry, there was an error contacting the AI service. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )

    # --- Parse structured output ---
    recommendations, clean_reply = _extract_recommendations(raw_reply, catalog_items)

    # end_of_conversation: True when the agent has committed to a shortlist
    end_of_conversation = (
        len(recommendations) > 0
        and action in (AgentAction.RECOMMEND, AgentAction.REFINE)
    )

    return ChatResponse(
        reply=clean_reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_recommendations(
    raw: str,
    catalog_items: list[dict],
) -> tuple[list[RecommendationItem], str]:
    """
    Parse <RECOMMENDATIONS>[...]</RECOMMENDATIONS> from raw LLM output.

    Validation layer:
      - Only accept items whose URL appears in the retrieved catalog slice
        (prevents hallucinated URLs from slipping through).
      - If JSON is malformed, return empty list rather than crashing.

    Returns:
        (recommendations, clean_reply_text)
    """
    pattern = r"<RECOMMENDATIONS>(.*?)</RECOMMENDATIONS>"
    match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)

    if not match:
        return [], raw.strip()

    block = match.group(1).strip()
    clean_reply = raw.replace(match.group(0), "").strip()

    # Parse JSON
    try:
        data = json.loads(block)
        if not isinstance(data, list):
            return [], clean_reply
    except json.JSONDecodeError:
        # Try to salvage by removing trailing commas (common LLM mistake)
        try:
            cleaned = re.sub(r",\s*]", "]", block)
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return [], clean_reply

    # Build allowlist of valid URLs from retrieved catalog
    valid_urls = {item["url"] for item in catalog_items}

    items: list[RecommendationItem] = []
    seen_urls: set[str] = set()

    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "").strip()
        url = entry.get("url", "").strip()
        test_type = entry.get("test_type", "K").strip().upper()

        if not name or not url:
            continue

        # Deduplicate
        if url in seen_urls:
            continue
        seen_urls.add(url)

        # Allow if URL came from catalog (strict) OR if name matches catalog item
        # (handles minor URL formatting differences)
        if url not in valid_urls:
            # Try name-based lookup as fallback
            found = retriever.get_item_by_name(name)
            if found:
                url = found["url"]
                test_type = found["test_type"]
            else:
                # Cannot verify — skip to prevent hallucination
                continue

        # Validate test_type code
        valid_types = {"K", "P", "A", "S", "C", "B", "E", "D"}
        if test_type not in valid_types:
            test_type = "K"

        items.append(RecommendationItem(name=name, url=url, test_type=test_type))

        if len(items) >= 10:  # Hard cap per spec
            break

    return items, clean_reply