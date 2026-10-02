"""
Unit tests for weather.py (Open-Meteo client)
"""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from weather import resolve_location, fetch_weather, WeatherClientError, HOURLY_FIELDS


class TestResolveLocation:
    def _mock_get(self, result: dict):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = result
        return mock_resp

    def test_successful_resolution(self):
        with patch("weather.httpx.get") as mock_get:
            mock_get.return_value = self._mock_get({
                "results": [{
                    "name": "London",
                    "country": "United Kingdom",
                    "latitude": 51.5074,
                    "longitude": -0.1278,
                    "timezone": "Europe/London",
                }]
            })
            result = resolve_location("London")
            assert result["name"] == "London"
            assert result["latitude"] == 51.5074
            assert result["country"] == "United Kingdom"

    def test_empty_results_raises(self):
        with patch("weather.httpx.get") as mock_get:
            mock_get.return_value = self._mock_get({"results": []})
            with pytest.raises(WeatherClientError):
                resolve_location("NonExistentPlace")

    def test_network_error_raises(self):
        with patch("weather.httpx.get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("unreachable")
            with pytest.raises(WeatherClientError):
                resolve_location("London")

    def test_http_error_raises(self):
        with patch("weather.httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404", request=MagicMock(), response=MagicMock()
            )
            mock_get.return_value = mock_resp
            with pytest.raises(WeatherClientError):
                resolve_location("London")


class TestFetchWeather:
    def _mock_response(self, current: dict) -> dict:
        return {"current": current, "hourly": {}}

    def test_parses_all_required_fields(self):
        current = {
            "temperature_2m": 20.0,
            "precipitation": 0.5,
            "wind_speed_10m": 25.0,
            "weather_code": 2,
            "uv_index": 4.0,
            "visibility": 9000.0,
            "relative_humidity_2m": 65.0,
            "apparent_temperature": 19.5,
        }
        with patch("weather.httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = self._mock_response(current)
            mock_get.return_value = mock_resp

            result = fetch_weather(51.5, -0.12, "Europe/London")

        for field in HOURLY_FIELDS:
            assert field in result, f"Field '{field}' missing from result"
        assert result["temperature_2m"] == 20.0
        assert result["wind_speed_10m"] == 25.0
        assert "raw" in result

    def test_falls_back_to_hourly_when_current_missing(self):
        payload = {
            "current": {},
            "hourly": {
                "temperature_2m": [18.0, 19.0],
                "precipitation": [0.0, 0.1],
                "wind_speed_10m": [10.0, 12.0],
                "weather_code": [1, 1],
                "uv_index": [3.0, 3.5],
                "visibility": [10000.0, 10000.0],
                "relative_humidity_2m": [55.0, 56.0],
                "apparent_temperature": [17.5, 18.0],
            }
        }
        with patch("weather.httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = payload
            mock_get.return_value = mock_resp

            result = fetch_weather(51.5, -0.12)

        assert result["temperature_2m"] == 18.0, "Should use first hourly value as fallback"

    def test_network_error_raises(self):
        with patch("weather.httpx.get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("DNS failure")
            with pytest.raises(WeatherClientError):
                fetch_weather(51.5, -0.12)

    def test_http_error_raises(self):
        with patch("weather.httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "503", request=MagicMock(), response=MagicMock()
            )
            mock_get.return_value = mock_resp
            with pytest.raises(WeatherClientError):
                fetch_weather(51.5, -0.12)

    def test_missing_visibility_defaults_to_clear(self):
        current = {
            "temperature_2m": 15.0,
            "precipitation": 0.0,
            "wind_speed_10m": 5.0,
            "weather_code": 0,
            "uv_index": 2.0,
            "relative_humidity_2m": 40.0,
            "apparent_temperature": 14.0,
            # visibility is absent
        }
        with patch("weather.httpx.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {"current": current, "hourly": {}}
            mock_get.return_value = mock_resp

            result = fetch_weather(51.5, -0.12)

        # Must not raise; visibility should be a reasonable default
        assert result.get("visibility") is not None
        assert result["visibility"] >= 0
