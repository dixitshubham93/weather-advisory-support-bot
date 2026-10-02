"""
Policy Engine — loads SOPs from sops/policies.yaml and evaluates them
against live weather data.

Design principles
─────────────────
• Purely deterministic for numeric conditions.
• LLM-assisted semantic matching is used ONLY to select among loaded SOPs;
  it cannot invent new advice.
• Fuzzy conditions call registered Python functions — adding a new fuzzy_fn
  requires adding a function here, not changing LangGraph graph code.
• Resolution strategy when multiple SOPs match:
    1. Highest severity  (critical > high > moderate > low)
    2. Most conditions matched
    3. Alphabetical by id  (deterministic tiebreak)
"""
from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ── Severity ordering ─────────────────────────────────────────────────────────
SEVERITY_RANK = {"critical": 4, "high": 3, "moderate": 2, "low": 1}

# ── SOP file location ─────────────────────────────────────────────────────────
_DEFAULT_SOP_PATH = Path(__file__).parent.parent / "sops" / "policies.yaml"


def load_sops(path: Optional[str] = None) -> list[dict]:
    """Load and return all SOPs from the YAML file."""
    sop_path = Path(path) if path else _DEFAULT_SOP_PATH
    with open(sop_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    policies = data.get("policies", [])
    logger.info("Loaded %d SOPs from %s", len(policies), sop_path)
    return policies


# ─────────────────────────────────────────────────────────────────────────────
# Numeric condition evaluator
# ─────────────────────────────────────────────────────────────────────────────

OPERATORS = {
    "gt":  lambda a, b: a > b,
    "lt":  lambda a, b: a < b,
    "gte": lambda a, b: a >= b,
    "lte": lambda a, b: a <= b,
    "eq":  lambda a, b: a == b,
}


def _evaluate_numeric_condition(condition: dict, weather: dict) -> bool:
    """Return True if the weather satisfies a single numeric condition."""
    field = condition["field"]
    op_name = condition["operator"]
    threshold = condition.get("threshold")

    if field not in weather:
        logger.debug("Weather field '%s' not present — condition fails", field)
        return False

    value = weather[field]
    if value is None:
        return False

    op_fn = OPERATORS.get(op_name)
    if op_fn is None:
        logger.warning("Unknown operator '%s' in condition", op_name)
        return False

    return op_fn(float(value), float(threshold))


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy / composite condition evaluators
# ─────────────────────────────────────────────────────────────────────────────

def _fuzzy_is_good_for_outing(weather: dict) -> tuple[bool, str]:
    """
    Composite comfort check for casual outdoor outings (picnics, strolls, etc.).

    Returns (triggered: bool, summary_message: str).
    'triggered' is True when conditions are NOT ideal, prompting the SOP guidance.
    The summary is embedded in the SOP guidance template.
    """
    temp = weather.get("temperature_2m", 20.0)
    wind = weather.get("wind_speed_10m", 0.0)
    rain = weather.get("precipitation", 0.0)
    uv   = weather.get("uv_index", 0.0)

    issues = []
    notes  = []

    if temp < 8:
        issues.append(f"temperature is quite cold ({temp:.1f} °C)")
    elif temp > 33:
        issues.append(f"temperature is very hot ({temp:.1f} °C)")
    else:
        notes.append(f"temperature is comfortable ({temp:.1f} °C)")

    if wind >= 30:
        issues.append(f"wind is strong ({wind:.1f} km/h)")
    else:
        notes.append(f"wind is light ({wind:.1f} km/h)")

    if rain >= 2:
        issues.append(f"there is rain ({rain:.1f} mm/h)")
    else:
        notes.append(f"no significant rain")

    if uv >= 8:
        issues.append(f"UV index is very high ({uv:.1f}) — sun protection essential")
    elif uv >= 6:
        notes.append(f"UV index is elevated ({uv:.1f}) — apply sunscreen")

    if issues:
        summary = (
            "Conditions have some concerns for a casual outing: "
            + "; ".join(issues)
            + ". "
            + ("Positive aspects: " + "; ".join(notes) + "." if notes else "")
        )
        return True, summary
    else:
        summary = (
            "Conditions look favourable for a casual outing: "
            + "; ".join(notes)
            + ". Enjoy your time outdoors!"
        )
        return True, summary  # Always trigger so we always provide a report


# Registry of fuzzy evaluator functions.
# Key = fuzzy_fn name used in policies.yaml.  Value = callable.
FUZZY_REGISTRY: dict[str, callable] = {
    "is_good_for_outing": _fuzzy_is_good_for_outing,
}


def _evaluate_condition(condition: dict, weather: dict, fuzzy_results: dict) -> bool:
    """Evaluate a single condition against weather data."""
    if condition["operator"] == "fuzzy":
        fn_name = condition.get("fuzzy_fn", "")
        fuzzy_fn = FUZZY_REGISTRY.get(fn_name)
        if not fuzzy_fn:
            logger.warning("Fuzzy function '%s' not registered", fn_name)
            return False
        triggered, summary = fuzzy_fn(weather)
        fuzzy_results["summary"] = summary  # passed back to guidance template
        return triggered
    return _evaluate_numeric_condition(condition, weather)


def _count_matched_conditions(sop: dict, weather: dict) -> tuple[bool, int, dict]:
    """
    Evaluate all conditions for one SOP.

    Returns (sop_matches: bool, conditions_matched_count: int, fuzzy_results: dict).
    """
    conditions = sop.get("conditions", [])
    logic = sop.get("logic", "any")
    fuzzy_results: dict = {}

    if not conditions:
        return False, 0, fuzzy_results

    results = []
    for cond in conditions:
        matched = _evaluate_condition(cond, weather, fuzzy_results)
        results.append(matched)

    matched_count = sum(results)
    if logic == "all":
        sop_matches = all(results)
    else:  # "any"
        sop_matches = any(results)

    return sop_matches, matched_count, fuzzy_results


# ─────────────────────────────────────────────────────────────────────────────
# Activity matching
# ─────────────────────────────────────────────────────────────────────────────

def _activity_matches(sop: dict, activity: str) -> bool:
    """
    Check whether the user's activity matches any SOP activity keyword.
    SOPs with activities=['any'] match everything.
    Matching is case-insensitive substring / token match.
    """
    sop_activities = sop.get("activities", [])
    if "any" in sop_activities:
        return True

    activity_lower = activity.lower()
    for sop_act in sop_activities:
        sop_act_lower = sop_act.lower()
        # Substring match in either direction
        if sop_act_lower in activity_lower or activity_lower in sop_act_lower:
            return True
        # Token overlap
        user_tokens = set(re.split(r"\W+", activity_lower))
        sop_tokens  = set(re.split(r"\W+", sop_act_lower))
        if user_tokens & sop_tokens:
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def find_candidate_sops(
    all_sops: list[dict],
    activity: str,
    weather: dict,
) -> list[tuple[dict, int, dict]]:
    """
    Return SOPs that match both the activity AND at least one weather condition.

    Returns a list of (sop, matched_condition_count, fuzzy_results) tuples.
    These are the TRIGGERED SOPs — both coverage and conditions are satisfied.
    """
    candidates = []
    for sop in all_sops:
        if not _activity_matches(sop, activity):
            continue
        matches, count, fuzzy_results = _count_matched_conditions(sop, weather)
        if matches:
            candidates.append((sop, count, fuzzy_results))

    return candidates


def find_activity_coverage(
    all_sops: list[dict],
    activity: str,
) -> list[dict]:
    """
    Return SOPs whose activity list covers the user's activity,
    WITHOUT evaluating weather conditions.

    This is used to distinguish:
      A) Zero coverage  → "we have no policies for this activity"
      B) Coverage exists but not triggered → "conditions are below threshold"

    These must produce different user-facing responses.
    """
    return [sop for sop in all_sops if _activity_matches(sop, activity)]


def describe_coverage_thresholds(covered_sops: list[dict]) -> str:
    """
    Generate a human-readable summary of the thresholds defined in the
    covered (but not currently triggered) SOPs.

    Used by handle_no_triggered_sop to explain WHY no SOP fired without
    hardcoding any SOP ID or threshold value.

    Example output:
      "SOP-001 (high): cycling wind hazard — triggers when wind_speed_10m ≥ 40 km/h"
    """
    lines = []
    OPERATOR_LABELS = {
        "gte": "≥", "gt": ">", "lte": "≤", "lt": "<", "eq": "=",
    }
    FIELD_UNITS = {
        "wind_speed_10m": "km/h",
        "precipitation": "mm/h",
        "temperature_2m": "°C",
        "apparent_temperature": "°C",
        "uv_index": "",
        "visibility": "m",
        "weather_code": "",
        "relative_humidity_2m": "%",
    }
    for sop in covered_sops:
        conditions = sop.get("conditions", [])
        cond_parts = []
        for c in conditions:
            if c.get("operator") == "fuzzy":
                cond_parts.append(f"composite comfort check")
                continue
            field = c.get("field", "?")
            op = OPERATOR_LABELS.get(c.get("operator", "?"), c.get("operator", "?"))
            thresh = c.get("threshold", "?")
            unit = FIELD_UNITS.get(field, "")
            cond_parts.append(f"{field} {op} {thresh}{(' ' + unit) if unit else ''}")
        cond_str = " AND ".join(cond_parts) if sop.get("logic") == "all" else " OR ".join(cond_parts)
        lines.append(
            f"- **{sop['id']}** ({sop['severity'].upper()}): {sop['title']} "
            f"— triggers when {cond_str}"
        )
    return "\n".join(lines)


def resolve_sop(
    candidates: list[tuple[dict, int, dict]],
) -> tuple[Optional[dict], Optional[dict], str]:
    """
    Apply resolution strategy to pick one SOP from candidates.

    Strategy:
      1. Highest severity
      2. Most conditions matched
      3. Alphabetical id (deterministic tiebreak)

    Returns (selected_sop, fuzzy_results, strategy_description).
    """
    if not candidates:
        return None, None, "no candidates"

    if len(candidates) == 1:
        sop, _, fuzzy_results = candidates[0]
        return sop, fuzzy_results, "single match"

    sorted_candidates = sorted(
        candidates,
        key=lambda t: (
            -SEVERITY_RANK.get(t[0]["severity"], 0),
            -t[1],
            t[0]["id"],
        ),
    )
    sop, _, fuzzy_results = sorted_candidates[0]
    strategy = (
        f"resolved {len(candidates)} candidates — "
        f"selected '{sop['id']}' (severity={sop['severity']}, "
        f"conditions_matched={sorted_candidates[0][1]})"
    )
    return sop, fuzzy_results, strategy


def render_guidance(sop: dict, weather: dict, fuzzy_results: Optional[dict] = None) -> str:
    """
    Fill in the guidance template with actual weather values.
    Also injects fuzzy_outing_summary for SOP-010.
    """
    guidance = sop["guidance"]

    # Replace {field_name} placeholders with actual weather values
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key == "fuzzy_outing_summary" and fuzzy_results:
            return fuzzy_results.get("summary", "")
        val = weather.get(key)
        if val is not None:
            return f"{val:.1f}" if isinstance(val, float) else str(val)
        return match.group(0)  # leave unreplaced if missing

    return re.sub(r"\{(\w+)\}", _replace, guidance)
