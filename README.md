# SHL Assessment Recommender

Conversational agent that helps hiring managers find the right SHL assessments
through multi-turn dialogue. Built for the SHL Labs AI Intern take-home assignment.

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo>
cd shl-recommender
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your catalog

Place the SHL catalog JSON at:
```
data/catalog_raw.json
```

### 3. Set your API key

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Build the FAISS index

```bash
python -m scripts.build_index
```

This creates `data/faiss_index/` with the FAISS index and metadata.
Run once; re-run when the catalog changes.

### 5. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /chat`

**Request**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer, mid-level"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "About 4 years experience"}
  ]
}
```

**Response**
```json
{
  "reply": "Here are assessments that fit a mid-level Java developer...",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "Occupational Personality Questionnaire OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": true
}
```

`recommendations` is an empty array when the agent is still clarifying.  
`end_of_conversation` is `true` when the agent has committed to a shortlist.

**test_type codes**

| Code | Meaning |
|------|---------|
| K | Knowledge & Skills |
| P | Personality & Behavior |
| A | Ability & Aptitude |
| S | Simulation |
| C | Competencies |
| B | Situational Judgment |
| E | Assessment Exercises |
| D | 360 / Development |

---

## Evaluation

Run Recall@10 against sample traces:

```bash
# Start the server first, then:
python -m scripts.evaluate --base-url http://localhost:8000 --traces-dir traces/
```

---

## Docker

```bash
# Build (requires data/catalog_raw.json)
docker build -t shl-recommender .

# Run
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... shl-recommender
```

---

## Project Structure

```
shl-recommender/
├── app/
│   ├── main.py          # FastAPI app, /health, /chat
│   ├── agent.py         # Decider logic (rule-based intent classification)
│   ├── retrieval.py     # FAISS index build and search
│   ├── catalog.py       # Catalog loading and preprocessing
│   └── prompts.py       # System prompt and context formatting
├── data/
│   ├── catalog_raw.json       # SHL catalog (you provide this)
│   └── faiss_index/           # Built by build_index.py
├── scripts/
│   ├── build_index.py   # One-time index builder
│   └── evaluate.py      # Recall@10 evaluator
├── traces/              # Sample conversation traces (.json)
├── Dockerfile
├── requirements.txt
├── approach.md          # 2-page design document
└── README.md
```
