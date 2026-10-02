"""
FastAPI application — Chat API for the Weather-Advisory Support Bot.

Endpoints
─────────
  POST /chat          — main chat endpoint
  POST /session/reset — clear session history
  GET  /health        — liveness check
  GET  /sessions      — list active session IDs (debug)

CORS is open for local development.  In production, restrict origins.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import os
import sys

# Make sure the backend directory is on the path when running from root
sys.path.insert(0, os.path.dirname(__file__))

from graph import graph
from session_store import get_history, append_turn, reset_session, list_sessions
from state import BotState, TraceMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Weather-Advisory Support Bot",
    description="Answers outdoor activity safety questions using live weather data and SOPs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(default=None)


class SOPCitation(BaseModel):
    id: Optional[str]
    title: Optional[str]
    category: Optional[str]
    severity: Optional[str]
    source: Optional[str]


class WeatherFacts(BaseModel):
    location: Optional[str]
    temperature_2m: Optional[float]
    precipitation: Optional[float]
    wind_speed_10m: Optional[float]
    uv_index: Optional[float]
    visibility: Optional[float]
    weather_code: Optional[int]
    apparent_temperature: Optional[float]


class ChatResponse(BaseModel):
    session_id: str
    response: str
    sop_citation: Optional[SOPCitation]
    weather_facts: Optional[WeatherFacts]
    policy_validated: bool
    candidate_sop_ids: list[str]        # TRIGGERED SOPs (activity + conditions met)
    covered_sop_ids: list[str]          # COVERAGE SOPs (activity match, conditions may differ)
    trace: dict
    error: Optional[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_initial_state(message: str, session_id: str, history: list[dict]) -> BotState:
    return BotState(
        user_message=message,
        session_id=session_id,
        chat_history=history,
        parsed_intent=None,
        location=None,
        weather=None,
        candidate_sops=[],
        selected_sop=None,
        policy_validated=False,
        response="",
        error=None,
        trace=TraceMetadata(
            nodes_visited=[],
            sop_candidates_count=0,
            candidate_sop_ids=[],
            covered_sop_ids=[],
            resolution_strategy_applied="",
            weather_fields_used=[],
            policy_validation_result=None,
            error_detail=None,
        ),
    )


def _extract_sop_citation(state: BotState) -> Optional[SOPCitation]:
    sop = state.get("selected_sop")
    if not sop:
        return None
    return SOPCitation(
        id=sop.get("id"),
        title=sop.get("title"),
        category=sop.get("category"),
        severity=sop.get("severity"),
        source=sop.get("source"),
    )


def _extract_weather_facts(state: BotState) -> Optional[WeatherFacts]:
    weather = state.get("weather")
    location = state.get("location")
    if not weather:
        return None
    return WeatherFacts(
        location=f"{location['name']}, {location['country']}" if location else None,
        temperature_2m=weather.get("temperature_2m"),
        precipitation=weather.get("precipitation"),
        wind_speed_10m=weather.get("wind_speed_10m"),
        uv_index=weather.get("uv_index"),
        visibility=weather.get("visibility"),
        weather_code=int(weather.get("weather_code", 0)) if weather.get("weather_code") is not None else None,
        apparent_temperature=weather.get("apparent_temperature"),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Creates a new session_id if none is provided.
    Loads session history, runs the LangGraph, persists the new turn.
    """
    session_id = request.session_id or str(uuid.uuid4())
    history = get_history(session_id)

    initial_state = _build_initial_state(request.message, session_id, history)

    try:
        final_state: BotState = graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("Graph execution error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    response_text = final_state.get("response", "")
    error = final_state.get("error")

    # Persist this turn to session memory
    append_turn(session_id, "user", request.message)
    append_turn(session_id, "assistant", response_text)

    return ChatResponse(
        session_id=session_id,
        response=response_text,
        sop_citation=_extract_sop_citation(final_state),
        weather_facts=_extract_weather_facts(final_state),
        policy_validated=final_state.get("policy_validated", False),
        candidate_sop_ids=final_state.get("trace", {}).get("candidate_sop_ids", []),
        covered_sop_ids=final_state.get("trace", {}).get("covered_sop_ids", []),
        trace=dict(final_state.get("trace", {})),
        error=error,
    )


@app.post("/session/reset")
async def reset(session_id: str):
    """Clear the conversation history for a session."""
    reset_session(session_id)
    return {"status": "ok", "session_id": session_id}


# ── Groq API configuration helper ─────────────────────────────────────────────

def _check_groq() -> dict:
    """
    Check that the Groq API configuration (WEATHERSUPPORT_KEY) is set and non-empty.
    Returns a dict with 'ok', 'model', 'base_url', and 'detail' keys without exposing secrets.
    """
    api_key = os.getenv("WEATHERSUPPORT_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com")

    if not api_key:
        return {
            "ok": False,
            "model": model,
            "base_url": base_url,
            "api_key_configured": False,
            "detail": "WEATHERSUPPORT_KEY environment variable is missing or empty",
        }

    return {
        "ok": True,
        "model": model,
        "base_url": base_url,
        "api_key_configured": True,
        "detail": "ready",
    }


@app.get("/health")
async def health():
    """
    Liveness + readiness check.

    Returns HTTP 200 if WEATHERSUPPORT_KEY is configured.
    Returns HTTP 503 if WEATHERSUPPORT_KEY is missing.
    """
    groq = _check_groq()
    status = {
        "status": "healthy" if groq["ok"] else "degraded",
        "llm_provider": "groq",
        "llm_model": groq["model"],
        "llm_base_url": groq["base_url"],
        "llm_ready": groq["ok"],
        "llm_detail": groq["detail"],
    }
    if not groq["ok"]:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content=status)
    return status


@app.get("/config")
async def config():
    """Return LLM provider configuration (no secrets)."""
    groq = _check_groq()
    return {
        "llm_provider": "groq",
        "llm_model": groq["model"],
        "llm_base_url": groq["base_url"],
        "api_key_configured": groq["api_key_configured"],
        "api_key_required": True,
        "note": "Set WEATHERSUPPORT_KEY in .env before running the server.",
    }


@app.get("/sessions")
async def sessions():
    return {"sessions": list_sessions()}


# ── Static frontend ───────────────────────────────────────────────────────────

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))
