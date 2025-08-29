from __future__ import annotations

from typing import Any, Dict


class WeatherTool:
    """Toy weather tool (stub)."""

    _WEATHER_HINTS = (
        "weather", "thoi tiet", "thời tiết", "forecast", "rain", "sunny", "temperature",
    )

    def looks_like_weather(self, question: str) -> bool:
        text = (question or "").strip().lower()
        return any(h in text for h in self._WEATHER_HINTS)

    def use_weather(self, question: str) -> Dict[str, Any]:
        return {"tool": "weather", "location": None, "status": "sunny"}


