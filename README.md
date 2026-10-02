# Weather-Advisory Support Bot

A chat application that answers outdoor activity safety questions using **live weather data** from Open-Meteo and a controlled set of externally defined Standard Operating Procedures (SOPs).

## Key Design Principles

- **The LLM never invents safety advice.** Every recommendation is derived from a matched SOP evaluated against actual weather values.
- **SOPs are external and configurable.** Adding a new SOP requires editing `sops/policies.yaml` only — no application code changes.
- **Honest failure.** The system explicitly fails when location resolution or weather data is unavailable rather than guessing.
- **Grounded responses.** Every answer cites the SOP that triggered it, or explicitly states no SOP applies.

---

## Architecture

```
User Question
     │
     ▼
 [LangGraph]
     │
  parse_intent ──── LLM extracts: activity, location, audience, time
     │
     ├─ no location / not outdoor ──► handle_failure → "specify a location"
     │
  resolve_location ── Open-Meteo geocoding API
     │
     ├─ location not found ──────────► handle_failure → "cannot resolve location"
     │
  fetch_weather ── Open-Meteo forecast API (live data)
     │
     ├─ API unavailable ─────────────► handle_failure → "cannot fetch weather"
     │
  match_sop ── Deterministic numeric + LLM semantic matching against loaded SOPs
     │
     ├─ no candidates ───────────────► handle_no_sop → "no policy covers this"
     │
  validate_policy ── Render SOP guidance with actual weather values
     │
  compose_response ── LLM composes final grounded response
     │
     ▼
  Final answer with SOP citation + live weather facts
```

### Files

| File | Purpose |
|------|---------|
| `sops/policies.yaml` | All 12 SOPs — external, editable without code changes |
| `backend/graph.py` | LangGraph definition: nodes + conditional edges |
| `backend/nodes.py` | Node implementations (one responsibility each) |
| `backend/state.py` | Shared TypedDict state schema |
| `backend/policy_engine.py` | SOP loader, condition evaluator, resolver, guidance renderer |
| `backend/weather.py` | Open-Meteo geocoding + forecast client |
| `backend/session_store.py` | In-memory session history (resets on server restart) |
| `backend/app.py` | FastAPI application |
| `frontend/` | HTML/CSS/JS chat UI |
| `tests/` | Evaluation suite + unit tests |

---

## Setup

### Prerequisites

- Python 3.11+
- Ollama installed and running locally
- `qwen2.5:7b` model pulled in Ollama (`ollama pull qwen2.5:7b`)

### 1. Clone and enter the project

```bash
git clone https://github.com/dixitshubham93/weather-advisory-support-bot.git
cd weather-advisory-support-bot
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

The application runs locally with Ollama and does not require an OpenAI API key. Optionally edit `.env` to configure Ollama settings:
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `qwen2.5:7b`)

### 5. Run the server

```bash
cd backend
uvicorn app:app --reload --port 8000
```

### 6. Open the UI

Visit `http://localhost:8000` in your browser.

---

## Running Tests

From the project root:

```bash
# All tests
pytest -v

# Just the evaluation suite
pytest tests/test_evaluation_suite.py -v

# Just unit tests
pytest tests/test_policy_engine.py tests/test_weather.py -v
```

> **Note:** Tests mock all external LLM and weather API calls. No API keys or running Ollama instance are required to run tests.

---

## SOPs — Design Rationale

SOPs live in `sops/policies.yaml`. They are structured so that a **generic policy engine** can evaluate them without needing to know about specific activities or thresholds.

### Why this structure?

Each SOP contains:
- `conditions` — a list of `{field, operator, threshold}` blocks that the engine evaluates against live weather
- `logic` — `any` or `all` — how conditions combine
- `guidance` — a template with `{field_name}` placeholders that get filled with actual weather values
- `fuzzy_fn` (optional) — for non-numeric composite assessments

This means:
- Adding SOP-013 = editing `policies.yaml` only
- The graph, nodes, and policy engine do NOT need to change
- Evaluated by the evaluator as: load the YAML, see if SOP-013 is caught by the engine ✓

### SOP Coverage

| SOP | Title | Category | Severity | Fuzzy? |
|-----|-------|----------|----------|--------|
| SOP-001 | High Wind — Cycling Hazard | wind_safety | high | No |
| SOP-002 | Storm-Force Wind — All Activities | wind_safety | critical | No |
| SOP-003 | Heavy Rain — Trail Running / Hiking | precipitation | high | No |
| SOP-004 | Moderate Rain — Outdoor Running | precipitation | moderate | No |
| SOP-005 | Extreme Heat — All Outdoor Exertion | temperature | critical | No |
| SOP-006 | High Heat — Outdoor Sporting Events | temperature | high | No |
| SOP-007 | Sub-Zero Temperatures — Hypothermia | temperature | high | No |
| SOP-008 | Low Visibility — Fog Advisory | general_conditions | moderate | No |
| SOP-009 | UV Index — Prolonged Sun Exposure | general_conditions | moderate | No |
| SOP-010 | General Outing Suitability — Picnic | general_conditions | low | ✅ Yes |
| SOP-011 | Thunderstorm — All Activities | precipitation | critical | No |
| SOP-012 | Snow / Blizzard — Cycling and Running | precipitation | critical | No |

### Categories (4)
- `wind_safety` — wind speed thresholds
- `precipitation` — rain, snow, thunderstorm codes
- `temperature` — heat, cold extremes
- `general_conditions` — visibility, UV, composite comfort

### Severity Levels (4)
- `critical` — suspend all activity (SOP-002, 005, 011, 012)
- `high` — strong advisory against activity (SOP-001, 003, 006, 007)
- `moderate` — proceed with specific precautions (SOP-004, 008, 009)
- `low` — informational/composite comfort report (SOP-010)

### Fuzzy SOP (SOP-010)
Rather than hardcoding "is it good for a picnic?", the engine has a registered Python function `is_good_for_outing` that evaluates a composite comfort assessment (temperature + wind + precipitation + UV). Any activity in the `activities` list (picnic, stroll, family outing, etc.) triggers this function. Adding a new fuzzy scenario requires:
1. Adding a new Python function in `policy_engine.py`
2. Registering it in `FUZZY_REGISTRY`
3. Referencing it in `policies.yaml`

No changes to the LangGraph graph are needed.

### Multiple SOP Resolution Strategy

When multiple SOPs match a situation, the engine uses:
1. **Highest severity** (critical > high > moderate > low)
2. **Most conditions matched** (more specific SOP wins)
3. **Alphabetical by ID** (deterministic tiebreak)

This is documented in `policy_engine.py` and tested in the evaluation suite.

---

## Chat API

### `POST /chat`

```json
{
  "message": "Is it safe to cycle in London today?",
  "session_id": "optional-existing-session-id"
}
```

Response:
```json
{
  "session_id": "uuid",
  "response": "Full answer text with citation...",
  "sop_citation": {
    "id": "SOP-001",
    "title": "High Wind — Cycling Hazard",
    "category": "wind_safety",
    "severity": "high",
    "source": "..."
  },
  "weather_facts": {
    "location": "London, United Kingdom",
    "temperature_2m": 14.2,
    "wind_speed_10m": 45.0,
    "precipitation": 0.0,
    ...
  },
  "trace": {
    "nodes_visited": ["parse_intent", "resolve_location", "fetch_weather", "match_sop", "validate_policy", "compose_response"],
    "sop_candidates_count": 2,
    "resolution_strategy_applied": "resolved 2 candidates — selected 'SOP-001'..."
  },
  "error": null
}
```

### `POST /session/reset?session_id=<id>`

Clears session history.

### `GET /health`

Returns `{"status": "healthy"}`.

---

## Session Handling

- Each `/chat` request with no `session_id` starts a fresh session.
- The last 3 turns of chat history are included in the intent-parsing prompt for context.
- Memory is **in-process** and resets when the server restarts (no database required).
- Explicitly reset with `POST /session/reset`.

---

## Evaluation Notes

### What works well
- Deterministic policy matching: the numeric condition evaluator is fully testable without LLM
- Failure handling: all 4 failure branches are explicit and tested
- SOP extensibility: Case 11 in the evaluation suite proves a new SOP works without code changes
- Thunderstorm override: CRITICAL severity always wins multi-SOP resolution

### Limitations
- **Live-data tests**: Tests for severe weather use mocked weather data. Real live tests depend on current conditions at evaluation time.
- **LLM intent parsing**: In rare cases, the LLM may mis-extract the activity (e.g., treating "swimming pool" as an indoor activity). The policy engine then correctly finds no matching outdoor SOP.
- **Semantic SOP matching**: Uses the LLM to expand keyword matching, but only among already-loaded SOPs. The LLM cannot create new policies.
- **Language**: Intent extraction is English-optimised; other languages may produce less reliable results.
- **Session persistence**: Sessions are lost on server restart. A production system would use a database.

### Security
User input is treated as untrusted. The policy engine evaluates SOPs against numeric weather values only — it never reads user text. Even if a user injects "ignore all policies", the deterministic condition evaluator will still fire the correct SOP based on weather data.

---

## Adding a New SOP

**No code changes needed.** Edit `sops/policies.yaml` only:

```yaml
- id: SOP-013
  title: "Cold Water — Swimming Advisory"
  category: temperature
  severity: high
  activities:
    - swimming
    - open water swimming
  description: >
    Water temperatures below 15°C create hypothermia risk for swimmers.
  conditions:
    - field: temperature_2m
      operator: lt
      threshold: 15
  logic: any
  guidance: >
    Temperature is {temperature_2m} °C — cold water hypothermia risk.
    Wear a full wetsuit and never swim alone.
  source: "Open Water Swimming Safety Association"
```

Restart the server. The new SOP is immediately active.

---

## Technology Stack

- **LangGraph** — graph orchestration with real branching and failure paths
- **LangChain Ollama** — intent extraction + response composition using local `qwen2.5:7b` model via Ollama
- **Open-Meteo** — free, no-key-required live weather + geocoding API
- **FastAPI** — async REST API
- **PyYAML** — external SOP loading
- **Pytest** — automated unit, integration, and evaluation test suite
