"""
Open-Meteo API client.

Two responsibilities:
  1. Geocoding  — resolve city name → (lat, lon, display_name, timezone)
  2. Forecast   — fetch current-hour weather values for a given lat/lon

Both functions raise WeatherClientError on any failure so that the graph
can route to the handle_failure node without guessing.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── Configurable URLs (override in tests) ─────────────────────────────────────
GEOCODE_URL = os.getenv(
    "OPEN_METEO_GEOCODE_URL",
    "https://geocoding-api.open-meteo.com/v1/search",
)
FORECAST_URL = os.getenv(
    "OPEN_METEO_FORECAST_URL",
    "https://api.open-meteo.com/v1/forecast",
)

# Fields explicitly requested from the forecast endpoint.
# Must align with WeatherData TypedDict fields.
HOURLY_FIELDS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "weather_code",
    "uv_index",
    "visibility",
    "relative_humidity_2m",
    "apparent_temperature",
]

TIMEOUT = 10.0  # seconds


class WeatherClientError(Exception):
    """Raised whenever geocoding or forecast fetching fails."""


def resolve_location(city: str) -> dict:
    """
    Resolve a city name to geographic coordinates.

    Returns a dict with keys: name, country, latitude, longitude, timezone.
    Raises WeatherClientError if the location cannot be resolved.
    """
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    try:
        resp = httpx.get(GEOCODE_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        raise WeatherClientError(
            f"Geocoding API unavailable for '{city}': {exc}"
        ) from exc
    except Exception as exc:
        raise WeatherClientError(
            f"Unexpected error resolving '{city}': {exc}"
        ) from exc

    results = data.get("results") or []
    if not results:
        raise WeatherClientError(
            f"Location '{city}' could not be resolved. "
            "Please check the spelling or try a nearby major city."
        )

    top = results[0]
    return {
        "name": top.get("name", city),
        "country": top.get("country", ""),
        "latitude": top["latitude"],
        "longitude": top["longitude"],
        "timezone": top.get("timezone", "UTC"),
    }


def fetch_weather(latitude: float, longitude: float, timezone: str = "auto") -> dict:
    """
    Fetch the current-hour weather values for the given coordinates.

    Returns a WeatherData-compatible dict.
    Raises WeatherClientError on any API or data-integrity failure.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_FIELDS),
        "current": ",".join(HOURLY_FIELDS),
        "timezone": timezone,
        "forecast_days": 1,
    }

    try:
        resp = httpx.get(FORECAST_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        raise WeatherClientError(
            f"Weather API unavailable: {exc}"
        ) from exc
    except Exception as exc:
        raise WeatherClientError(
            f"Unexpected error fetching weather: {exc}"
        ) from exc

    # Prefer the 'current' block; fall back to first hourly index.
    current = data.get("current") or {}

    def _get(field: str) -> Optional[float]:
        val = current.get(field)
        if val is not None:
            return val
        # fallback: first hourly value
        hourly = data.get("hourly", {})
        values = hourly.get(field, [])
        return values[0] if values else None

    missing = []
    result: dict = {}
    for field in HOURLY_FIELDS:
        val = _get(field)
        if val is None:
            missing.append(field)
        result[field] = val if val is not None else 0.0

    if missing:
        logger.warning("Weather fields missing from API response: %s", missing)

    # Visibility may be absent — default 10 000 m (clear) if truly missing
    if result.get("visibility") is None or result.get("visibility") == 0.0:
        raw_vis = _get("visibility")
        result["visibility"] = raw_vis if raw_vis is not None else 10000.0

    result["raw"] = data
    return result
