"""
Unit tests for policy_engine.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from policy_engine import (
    load_sops,
    find_candidate_sops,
    resolve_sop,
    render_guidance,
    _evaluate_numeric_condition,
    _activity_matches,
    FUZZY_REGISTRY,
    SEVERITY_RANK,
)

SOP_PATH = str(Path(__file__).parent.parent / "sops" / "policies.yaml")


@pytest.fixture(scope="module")
def all_sops():
    return load_sops(SOP_PATH)


class TestSOPLoading:
    def test_loads_all_sops(self, all_sops):
        assert len(all_sops) >= 10

    def test_required_fields_present(self, all_sops):
        required = {"id", "title", "category", "severity", "activities", "conditions", "guidance"}
        for sop in all_sops:
            missing = required - sop.keys()
            assert not missing, f"SOP {sop.get('id')} missing fields: {missing}"

    def test_severity_values_valid(self, all_sops):
        valid = set(SEVERITY_RANK.keys())
        for sop in all_sops:
            assert sop["severity"] in valid, f"SOP {sop['id']} has invalid severity: {sop['severity']}"

    def test_at_least_3_categories(self, all_sops):
        cats = {s["category"] for s in all_sops}
        assert len(cats) >= 3, f"Need ≥3 categories, got: {cats}"

    def test_at_least_3_severity_levels(self, all_sops):
        sevs = {s["severity"] for s in all_sops}
        assert len(sevs) >= 3, f"Need ≥3 severity levels, got: {sevs}"


class TestNumericConditionEvaluation:
    def test_gte_passes(self):
        cond = {"field": "wind_speed_10m", "operator": "gte", "threshold": 40}
        assert _evaluate_numeric_condition(cond, {"wind_speed_10m": 50.0})

    def test_gte_fails(self):
        cond = {"field": "wind_speed_10m", "operator": "gte", "threshold": 40}
        assert not _evaluate_numeric_condition(cond, {"wind_speed_10m": 39.9})

    def test_lt_passes(self):
        cond = {"field": "temperature_2m", "operator": "lt", "threshold": 0}
        assert _evaluate_numeric_condition(cond, {"temperature_2m": -3.0})

    def test_missing_field_fails(self):
        cond = {"field": "no_such_field", "operator": "gte", "threshold": 10}
        assert not _evaluate_numeric_condition(cond, {"wind_speed_10m": 20.0})

    def test_boundary_gte_exact(self):
        cond = {"field": "wind_speed_10m", "operator": "gte", "threshold": 40}
        assert _evaluate_numeric_condition(cond, {"wind_speed_10m": 40.0})


class TestActivityMatching:
    def test_exact_match(self):
        sop = {"activities": ["cycling"]}
        assert _activity_matches(sop, "cycling")

    def test_any_matches_everything(self):
        sop = {"activities": ["any"]}
        assert _activity_matches(sop, "surfing")
        assert _activity_matches(sop, "yoga")

    def test_substring_match(self):
        sop = {"activities": ["mountain biking"]}
        assert _activity_matches(sop, "mountain biking in the hills")

    def test_token_overlap(self):
        sop = {"activities": ["trail running"]}
        assert _activity_matches(sop, "running on trails")

    def test_no_match(self):
        sop = {"activities": ["swimming"]}
        assert not _activity_matches(sop, "cycling")


class TestResolutionStrategy:
    def make_sop(self, id_, severity):
        return {
            "id": id_,
            "title": id_,
            "category": "test",
            "severity": severity,
            "activities": [],
            "conditions": [],
            "logic": "any",
            "guidance": "test",
            "source": "test",
        }

    def test_critical_beats_high(self):
        candidates = [
            (self.make_sop("SOP-A", "high"), 1, {}),
            (self.make_sop("SOP-B", "critical"), 1, {}),
        ]
        selected, _, _ = resolve_sop(candidates)
        assert selected["id"] == "SOP-B"

    def test_more_conditions_wins_tiebreak(self):
        candidates = [
            (self.make_sop("SOP-A", "high"), 1, {}),
            (self.make_sop("SOP-B", "high"), 3, {}),
        ]
        selected, _, _ = resolve_sop(candidates)
        assert selected["id"] == "SOP-B"

    def test_alphabetical_tiebreak(self):
        candidates = [
            (self.make_sop("SOP-B", "high"), 2, {}),
            (self.make_sop("SOP-A", "high"), 2, {}),
        ]
        selected, _, _ = resolve_sop(candidates)
        assert selected["id"] == "SOP-A"

    def test_single_candidate_returned(self):
        sop = self.make_sop("SOP-X", "moderate")
        candidates = [(sop, 1, {})]
        selected, _, strategy = resolve_sop(candidates)
        assert selected["id"] == "SOP-X"
        assert strategy == "single match"

    def test_empty_candidates_returns_none(self):
        selected, _, strategy = resolve_sop([])
        assert selected is None


class TestGuidanceRendering:
    def test_placeholder_replaced(self):
        sop = {
            "id": "SOP-T",
            "title": "Test",
            "category": "test",
            "severity": "low",
            "activities": [],
            "conditions": [],
            "logic": "any",
            "guidance": "Wind is {wind_speed_10m} km/h.",
            "source": "test",
        }
        weather = {"wind_speed_10m": 55.0}
        result = render_guidance(sop, weather)
        assert "55.0" in result
        assert "{wind_speed_10m}" not in result

    def test_missing_field_kept_as_is(self):
        sop = {
            "id": "SOP-T",
            "title": "Test",
            "category": "test",
            "severity": "low",
            "activities": [],
            "conditions": [],
            "logic": "any",
            "guidance": "Temp {temperature_2m}.",
            "source": "test",
        }
        result = render_guidance(sop, {})
        assert "{temperature_2m}" in result  # unreplaced

    def test_fuzzy_summary_injected(self):
        sop = {
            "id": "SOP-010",
            "title": "General Outing",
            "category": "general_conditions",
            "severity": "low",
            "activities": ["picnic"],
            "conditions": [{"field": "__composite__", "operator": "fuzzy", "fuzzy_fn": "is_good_for_outing"}],
            "logic": "any",
            "guidance": "{fuzzy_outing_summary}",
            "source": "test",
        }
        weather = {"temperature_2m": 22.0, "wind_speed_10m": 10.0, "precipitation": 0.0, "uv_index": 3.0}
        fuzzy_results = {"summary": "Conditions look great!"}
        result = render_guidance(sop, weather, fuzzy_results)
        assert "Conditions look great!" in result
