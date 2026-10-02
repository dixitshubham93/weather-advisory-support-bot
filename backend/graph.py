"""
LangGraph definition for the Weather-Advisory Support Bot.

Graph topology
──────────────

  START
    │
    ▼
  parse_intent
    │
    ├─[no location / not outdoor]──► handle_failure
    │
    ▼
  resolve_location
    │
    ├─[resolution failed]───────────► handle_failure
    │
    ▼
  fetch_weather
    │
    ├─[API unavailable]─────────────► handle_failure
    │
    ▼
  match_sop
    │
    ├─[no SOP coverage for activity]───────────────► handle_no_sop
    ├─[coverage exists, conditions not triggered]──► handle_no_triggered_sop
    │
    ▼
  validate_policy
    │
    ├─[conditions not satisfied]────► handle_no_sop
    │
    ▼
  compose_response
    │
    ▼
  END

All failure branches terminate at handle_failure, handle_no_sop, or
handle_no_triggered_sop, all of which route directly to END.

Critical distinction:
  handle_no_sop         → zero SOPs cover this activity
  handle_no_triggered_sop → SOPs cover it, but current weather is below all thresholds
"""
from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from state import BotState
from nodes import (
    parse_intent,
    resolve_location,
    fetch_weather_node,
    match_sop,
    validate_policy,
    compose_response,
    handle_no_sop,
    handle_no_triggered_sop,
    handle_failure,
)

logger = logging.getLogger(__name__)


# ── Conditional edge functions ────────────────────────────────────────────────

def _after_parse_intent(
    state: BotState,
) -> Literal["resolve_location", "handle_failure"]:
    """
    Route after parse_intent:
      - If parsing failed or no location → handle_failure
      - If not an outdoor query → handle_failure
      - Otherwise → resolve_location
    """
    if state.get("error"):
        return "handle_failure"
    intent = state.get("parsed_intent")
    if not intent:
        return "handle_failure"
    if not intent.get("is_outdoor_query", True):
        # Inject a specific message for non-outdoor queries
        return "handle_failure"
    return "resolve_location"


def _after_resolve_location(
    state: BotState,
) -> Literal["fetch_weather", "handle_failure"]:
    if state.get("error") or not state.get("location"):
        return "handle_failure"
    return "fetch_weather"


def _after_fetch_weather(
    state: BotState,
) -> Literal["match_sop", "handle_failure"]:
    if state.get("error") or not state.get("weather"):
        return "handle_failure"
    return "match_sop"


def _after_match_sop(
    state: BotState,
) -> Literal["validate_policy", "handle_no_triggered_sop", "handle_no_sop"]:
    """
    Route after match_sop based on two separate concepts:

      candidate_sops (triggered)  — SOPs whose conditions ARE met by current weather
      covered_sop_ids (coverage)  — SOPs that cover the activity, regardless of weather

    Routing:
      triggered SOPs exist            → validate_policy → compose_response
      coverage exists, none triggered → handle_no_triggered_sop
        ("conditions are below threshold")
      zero coverage                   → handle_no_sop
        ("we have no policies for this activity")
    """
    if state.get("selected_sop"):
        return "validate_policy"
    covered_ids = state.get("trace", {}).get("covered_sop_ids", [])
    if covered_ids:
        return "handle_no_triggered_sop"
    return "handle_no_sop"


def _after_validate_policy(
    state: BotState,
) -> Literal["compose_response", "handle_no_sop"]:
    """
    Route after validate_policy.

    validate_policy re-runs the deterministic condition evaluator and sets
    state["policy_validated"].  If conditions are NOT satisfied (edge case),
    we route to handle_no_sop rather than producing unsupported advice.
    """
    if not state.get("policy_validated", False):
        return "handle_no_sop"
    return "compose_response"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph for the advisory bot."""
    builder = StateGraph(BotState)

    # Register nodes
    builder.add_node("parse_intent",           parse_intent)
    builder.add_node("resolve_location",        resolve_location)
    builder.add_node("fetch_weather",           fetch_weather_node)
    builder.add_node("match_sop",               match_sop)
    builder.add_node("validate_policy",         validate_policy)
    builder.add_node("compose_response",        compose_response)
    builder.add_node("handle_no_sop",           handle_no_sop)
    builder.add_node("handle_no_triggered_sop", handle_no_triggered_sop)
    builder.add_node("handle_failure",          handle_failure)

    # Entry point
    builder.set_entry_point("parse_intent")

    # Conditional edges (branching logic)
    builder.add_conditional_edges(
        "parse_intent",
        _after_parse_intent,
        {"resolve_location": "resolve_location", "handle_failure": "handle_failure"},
    )
    builder.add_conditional_edges(
        "resolve_location",
        _after_resolve_location,
        {"fetch_weather": "fetch_weather", "handle_failure": "handle_failure"},
    )
    builder.add_conditional_edges(
        "fetch_weather",
        _after_fetch_weather,
        {"match_sop": "match_sop", "handle_failure": "handle_failure"},
    )
    builder.add_conditional_edges(
        "match_sop",
        _after_match_sop,
        {
            "validate_policy":         "validate_policy",
            "handle_no_triggered_sop": "handle_no_triggered_sop",
            "handle_no_sop":           "handle_no_sop",
        },
    )
    builder.add_conditional_edges(
        "validate_policy",
        _after_validate_policy,
        {"compose_response": "compose_response", "handle_no_sop": "handle_no_sop"},
    )

    # Deterministic edges
    builder.add_edge("compose_response",        END)
    builder.add_edge("handle_no_sop",            END)
    builder.add_edge("handle_no_triggered_sop",  END)
    builder.add_edge("handle_failure",           END)

    return builder.compile()


# Compile once — imported by app.py
graph = build_graph()
