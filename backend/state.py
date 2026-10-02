"""
State schema for the Weather-Advisory Support Bot LangGraph.

Keeping the schema as a single TypedDict ensures every node reads/writes
from the same structure and the graph remains easy to trace.
"""
from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict


class ParsedIntent(TypedDict):
    activity: str           # e.g. "cycling", "hiking"
    location_name: str      # raw location string from user
    audience: str           # e.g. "general", "children", "athletes"
    time_context: str       # e.g. "today", "this afternoon"
    is_outdoor_query: bool


class LocationInfo(TypedDict):
    name: str               # resolved display name
    country: str
    latitude: float
    longitude: float
    timezone: str


class WeatherData(TypedDict):
    temperature_2m: float       # °C
    precipitation: float        # mm/h (current hour)
    wind_speed_10m: float       # km/h
    weather_code: int           # WMO code
    uv_index: float
    visibility: float           # metres (if available)
    relative_humidity_2m: float
    apparent_temperature: float
    raw: dict                   # full API response snapshot


class SOPRecord(TypedDict):
    id: str
    title: str
    category: str
    severity: str
    activities: list[str]
    description: str
    conditions: list[dict]
    logic: str
    guidance: str
    source: str


class TraceMetadata(TypedDict):
    nodes_visited: list[str]
    sop_candidates_count: int
    candidate_sop_ids: list[str]          # TRIGGERED SOPs (activity + conditions met)
    covered_sop_ids: list[str]            # COVERAGE SOPs (activity match only, conditions may not be met)
    resolution_strategy_applied: str
    weather_fields_used: list[str]
    policy_validation_result: Optional[str]   # "satisfied" | "not_satisfied" | None
    error_detail: Optional[str]


class BotState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    user_message: str
    session_id: str
    chat_history: list[dict]        # [{"role": "user"|"assistant", "content": str}]

    # ── Parsed ────────────────────────────────────────────────────────────
    parsed_intent: Optional[ParsedIntent]

    # ── Resolved ──────────────────────────────────────────────────────────
    location: Optional[LocationInfo]
    weather: Optional[WeatherData]

    # ── Policy ────────────────────────────────────────────────────────────
    candidate_sops: list[SOPRecord]       # ALL matching SOPs (for trace/audit)
    selected_sop: Optional[SOPRecord]     # winner after resolution strategy
    policy_validated: bool                 # set by validate_policy node; drives branching

    # ── Output ────────────────────────────────────────────────────────────
    response: str
    error: Optional[str]
    trace: TraceMetadata
