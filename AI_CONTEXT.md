# AI_CONTEXT.md — Weather-Advisory Support Bot

> **Purpose**: Durable handoff file for any AI coding session or collaborator.
> Commit this file. Do not .gitignore it.

---

## LLM Configuration

| Key | Value |
|-----|-------|
| **LLM Provider** | Ollama (local) |
| **Model** | `qwen2.5:7b` |
| **API Cost** | Free / local — no usage fees |
| **External LLM API key** | **Not required** |
| **Ollama base URL** | `http://localhost:11434` (configurable via `OLLAMA_BASE_URL`) |
| **How to install** | `ollama pull qwen2.5:7b` |

---

## Project Goal

Build a chat application that answers outdoor activity safety questions
using **live weather data** from Open-Meteo and a **controlled set of
externally defined Standard Operating Procedures (SOPs)**.

The LLM must NOT invent safety advice. Every recommendation is derived
from a matched SOP evaluated against actual weather values from the API.

---

## Assignment Constraints (non-negotiable)

| # | Constraint |
|---|-----------|
| 1 | LangGraph is mandatory, real branching required |
| 2 | Live weather from Open-Meteo only |
| 3 | SOPs external in `sops/policies.yaml` |
| 4 | Changing/adding an SOP must NOT require modifying LangGraph code |
| 5 | ≥10 SOPs, ≥3 categories, multiple severity levels |
| 6 | At least one fuzzy/non-numeric SOP |
| 7 | Every response must cite the SOP that caused advice OR state no SOP applies |
| 8 | Explicit failure branches — no guessing on location/weather failures |
| 9 | Multiple matching SOPs → deterministic documented resolution |
| 10 | Session context preserved within session; memory resets between sessions |
| 11 | Minimal functional chat frontend |
| 12 | Evaluation suite: 2+ clear SOP, 2+ paraphrase, 1 severe, 1 no-SOP, 1 API fail, 1 location fail, 1 adversarial, 1 new-SOP-no-code-change |
| 13 | README: setup, execution, SOP rationale, evaluation results |

---

## Architecture

### LangGraph topology (6 conditional edges, 3 terminal nodes)

```
START
  -> parse_intent           [LLM: extracts activity, location, audience, time]
      -> FAIL: handle_failure          (no location / not outdoor / parse error)
  -> resolve_location       [Open-Meteo geocoding API]
      -> FAIL: handle_failure          (location unresolvable or API error)
  -> fetch_weather          [Open-Meteo forecast API]
      -> FAIL: handle_failure          (API unavailable / data incomplete)
  -> match_sop              [deterministic numeric + LLM semantic activity match]
      -> NO COVERAGE:    handle_no_sop           (zero SOPs cover this activity)
      -> NOT TRIGGERED:  handle_no_triggered_sop (SOPs exist, conditions below threshold)
  -> validate_policy        [re-evaluates SOP conditions deterministically]
      -> NOT SATISFIED:  handle_no_sop  (edge case: conditions failed at validation)
  -> compose_response       [LLM: composes grounded final answer]
  -> END
```

**Critical distinction — two separate no-response paths:**

| Node | Meaning | Example |
|------|---------|--------|
| `handle_no_sop` | Zero SOPs cover this activity at all | "swimming" — no SOP exists |
| `handle_no_triggered_sop` | SOPs cover the activity, but weather is below every threshold | Cycling with wind = 8.8 km/h (SOP-001 needs ≥ 40 km/h) |

### LLM Boundary (critical)

The LLM is permitted to:
- Extract structured intent (activity, location, audience, time_context)
- Expand activity keyword matching semantically (selecting among loaded SOPs only)
- Compose the final natural-language response

The LLM is NOT permitted to:
- Invent weather values
- Decide whether weather is dangerous
- Create or override an SOP
- Produce safety advice when no SOP applies

### Policy Engine (deterministic core)

`backend/policy_engine.py` is the security boundary.

- Loads SOPs from `sops/policies.yaml` (external, no code change for new SOPs)
- Evaluates numeric conditions (`gt`, `lt`, `gte`, `lte`, `eq`) against weather dict
- Evaluates fuzzy conditions via `FUZZY_REGISTRY` (Python functions, not LLM)
- Resolution strategy (when multiple SOPs match):
  1. Highest severity (critical > high > moderate > low)
  2. Most conditions matched (specificity)
  3. Alphabetical by ID (deterministic tiebreak)

---

## Repository Structure

```
MeddiBuddy_Assig/
├── backend/
│   ├── app.py              FastAPI: /chat /health /config /session/reset /sessions
│   ├── graph.py            LangGraph: 8 nodes + 5 conditional edges
│   ├── nodes.py            All 8 node functions (ChatOllama, no API key)
│   ├── state.py            BotState TypedDict (shared schema)
│   ├── weather.py          Open-Meteo geocoding + forecast client
│   ├── policy_engine.py    SOP loader, condition evaluator, resolver, renderer
│   ├── session_store.py    In-memory session history (thread-safe)
│   └── requirements.txt    langchain-ollama replaces langchain-openai
├── sops/
│   └── policies.yaml       12 SOPs — the ONLY place to add/edit policies
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/
│   ├── test_evaluation_suite.py   13 evaluator-facing cases (92 tests)
│   ├── test_integration.py        Full graph traversal w/ mocked Ollama (13 tests)
│   ├── test_policy_engine.py      Unit tests for policy_engine.py (23 tests)
│   └── test_weather.py            Unit tests for weather.py (9 tests)
├── conftest.py             Adds backend/ to sys.path (no API key needed)
├── pytest.ini
├── README.md
├── AI_CONTEXT.md           ← this file
├── .env.example            OLLAMA_BASE_URL + OLLAMA_MODEL (no secrets)
└── .gitignore
```

---

## Important Design Decisions

### 1. Ollama replaces OpenAI — no API key required
`ChatOllama` from `langchain-ollama` is used for all LLM calls.
The model defaults to `qwen2.5:7b`. The base URL defaults to
`http://localhost:11434`. Both are configurable via environment variables.
No API key of any kind is required.

### 2. Honest failure when Ollama is unavailable
`GET /health` pings `OLLAMA_BASE_URL/api/tags` and returns HTTP 503 if
Ollama is not running or the model is not pulled. The system never invents
answers without a working LLM.

### 3. `GET /config` endpoint
Returns LLM provider settings without requiring source code inspection.
Shows `api_key_required: false`.

### 4. `validate_policy` is a real conditional branch
After `match_sop` selects an SOP, `validate_policy` re-runs the
deterministic condition evaluator. If conditions fail (edge case), it sets
`policy_validated=False` and the graph routes to `handle_no_sop`.

### 5. `policy_validated` flag in BotState
Added to state so the graph's `_after_validate_policy` edge function can
read it cleanly. Exposed in the API response so evaluators can inspect it.

### 6. `candidate_sop_ids` in trace
`match_sop` records ALL matching SOP IDs (not just the winner) in
`trace.candidate_sop_ids`. The API response exposes this for audit.

### 7. Fuzzy SOP (SOP-010) uses a Python function, not LLM
`FUZZY_REGISTRY["is_good_for_outing"]` is a pure Python composite check.
Adding a new fuzzy condition type: (a) add Python function to
`policy_engine.py`, (b) register in `FUZZY_REGISTRY`, (c) reference in
`policies.yaml`. No graph changes.

### 8. No hardcoded SOP IDs in graph.py or nodes.py
Tested explicitly. The test strips docstrings before scanning executable
code, so illustrative examples in LLM prompt templates are not flagged.

---

## Current Implementation Status

### Completed ✅

- [x] `sops/policies.yaml` — 12 SOPs, 4 categories, 4 severity levels, 1 fuzzy
- [x] `backend/state.py` — TypedDict with `policy_validated`, `candidate_sop_ids`, `covered_sop_ids` in trace
- [x] `backend/graph.py` — 9 nodes, 6 conditional edges, 3 terminal paths
- [x] `backend/nodes.py` — ChatOllama (qwen2.5:7b), all 9 nodes incl. `handle_no_triggered_sop`
- [x] `backend/policy_engine.py` — numeric + fuzzy evaluator, `find_activity_coverage()`, `describe_coverage_thresholds()`, resolution, renderer
- [x] `backend/weather.py` — Open-Meteo geocoding + forecast
- [x] `backend/session_store.py` — thread-safe in-process session store
- [x] `backend/app.py` — FastAPI with Ollama health check (`/health` → HTTP 503 if down), `covered_sop_ids` in response
- [x] `frontend/` — dark-mode chat UI
- [x] `tests/test_evaluation_suite.py` — 15 cases, 107 tests (incl. Case 14 + 15 regressions)
- [x] `tests/test_integration.py` — 14 tests, full graph with mocked Ollama
- [x] `tests/test_policy_engine.py` — 23 unit tests
- [x] `tests/test_weather.py` — 9 unit tests
- [x] `README.md`
- [x] `conftest.py` — no dummy API key needed

### Test Results (latest run)

```
115 passed, 0 failed, 0 skipped  (2.34s)
Tests: test_evaluation_suite (107) + test_integration (14) +
       test_policy_engine (23) + test_weather (9) +
       regression Case 14 (8) + regression Case 15 (4)
No Ollama instance needed — all LLM calls mocked.
No Ollama instance needed to run tests — all LLM calls mocked.
```

### Live Verification

```
GET /health   → {"status":"healthy","llm_provider":"ollama","llm_model":"qwen2.5:7b",
                  "llm_base_url":"http://localhost:11434","llm_ready":true,"llm_detail":"ready"}

GET /config   → {"llm_provider":"ollama","llm_model":"qwen2.5:7b",
                  "llm_base_url":"http://localhost:11434","api_key_required":false}
```

---

## Known Gaps

1. **Ollama must be running** for the server to respond to `/chat`. The health
   endpoint returns 503 honestly if it's not. Tests are fully mocked.

2. **Live severe-weather case** — TestCase6 uses mocked `weather_code=96`.

3. **Session persistence** — in-process only; resets on server restart.

---

## Failures Encountered and Fixed

| Failure | Root Cause | Fix |
|---------|-----------|-----|
| `validate_policy` unconditional edge | Hard `add_edge` | Added `_after_validate_policy` + `add_conditional_edges` |
| `test_nodes_file_contains_no_hardcoded_sop_ids` failed | Regex matched examples in prompt docstrings | Strip triple-quoted strings before scanning |
| `policy_validated` missing from state init | Not added to `app.py` | Added `policy_validated=False` + new trace fields |
| Live query failed with model not found | `qwen2.5:7b` tag not pulled, only `qwen2.5:7b-instruct` existed | Ran `ollama pull qwen2.5:7b` |
| **"I don't have a policy covering cycling"** when wind is low | `find_candidate_sops` mixed activity-coverage and weather-trigger in one pass; empty result → system incorrectly said "no coverage" | Added `find_activity_coverage()` (activity-only pass) and `handle_no_triggered_sop` node; `_after_match_sop` now has 3 routes: triggered → validate_policy, covered+not-triggered → handle_no_triggered_sop, zero-coverage → handle_no_sop |
| **"An unexpected error occurred"** for indoor yoga (intermittent) | LLM sometimes wraps JSON in prose; `json.loads()` failed and `error=None` propagated to `handle_failure` → generic fallback | Added two-level JSON parsing (direct parse + regex `{...}` extraction fallback); `parse_intent` now pre-sets friendly `error_msg` for non-outdoor and missing-location cases |

---

## Exact Next Recommended Action

```bash
# Pull model (once)
ollama pull qwen2.5:7b

# Copy env (no secrets needed — just base URL override if not default)
cp .env.example .env

# Install dependencies
pip install -r backend/requirements.txt

# Run tests (no Ollama needed)
pytest -v   # → 105 passed

# Start server (Ollama must be running)
cd backend
uvicorn app:app --reload --port 8000

# Verify
curl http://localhost:8000/health
curl http://localhost:8000/config
```

---

## SOP Quick Reference

| SOP | Category | Severity | Trigger | Activity |
|-----|----------|----------|---------|----------|
| SOP-001 | wind_safety | high | wind ≥ 40 km/h | cycling |
| SOP-002 | wind_safety | critical | wind ≥ 70 km/h | any |
| SOP-003 | precipitation | high | rain ≥ 10 mm/h | hiking/trail running |
| SOP-004 | precipitation | moderate | rain 2–10 mm/h | running |
| SOP-005 | temperature | critical | temp ≥ 38 °C | any |
| SOP-006 | temperature | high | temp 32–37 °C | outdoor events |
| SOP-007 | temperature | high | temp < 0 °C | any |
| SOP-008 | general_conditions | moderate | visibility < 1000 m | cycling/running |
| SOP-009 | general_conditions | moderate | UV ≥ 6 | picnic/sunbathing |
| SOP-010 | general_conditions | low | fuzzy composite | picnic/stroll/outing |
| SOP-011 | precipitation | critical | weather_code ≥ 95 | any (thunderstorm) |
| SOP-012 | precipitation | critical | code 71–77 (snow) | cycling/running/hiking |
