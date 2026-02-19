"""Open-Meteo weather analysis tool with Ollama explanation."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from loguru import logger

from models.schemas import WeatherAnalysis, WeatherDaySummary

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
REQUEST_TIMEOUT = 15
OLLAMA_MODEL = "llama3.2"
ANOMALY_THRESHOLD = 1.5

EXPLAIN_PROMPT = """\
You are a friendly weather analyst. Given the following weather data for
{location}, write a 3–4 paragraph summary covering:
- Temperature trends over the period
- Any anomalous days and why they stand out
- Precipitation and wind patterns
- Data quality: if missingness_pct is present and any value is above 0, briefly note any gaps or reliability.
- A practical outlook / recommendation

Data:
{data}
"""


class WeatherTool:
    """Fetches, analyses, and explains weather data via Open-Meteo + Ollama."""

    def __init__(self) -> None:
        logger.info("Initialising WeatherTool")
        self._llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.4)
        logger.info("WeatherTool ready")

    def geocode_city(self, city_name: str) -> dict:
        params = {"name": city_name, "count": 1, "language": "en", "format": "json"}
        resp = requests.get(GEOCODING_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("results"):
            raise ValueError(f"City '{city_name}' not found. Try a different name.")
        result = data["results"][0]
        logger.info("Geocoded '{}' → {}, {} ({:.4f}, {:.4f})",
                     city_name, result["name"], result.get("country", ""),
                     result["latitude"], result["longitude"])
        return {
            "name": result["name"], "country": result.get("country", ""),
            "latitude": result["latitude"], "longitude": result["longitude"],
            "timezone": result.get("timezone", ""),
        }

    def fetch_weather(self, latitude: float, longitude: float, days: int) -> dict:
        params = {
            "latitude": latitude, "longitude": longitude,
            "hourly": "temperature_2m,precipitation,windspeed_10m",
            "forecast_days": days,
        }
        try:
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            logger.info("Fetched weather data ({} days)", days)
            return resp.json()
        except Exception:
            logger.exception("Failed to fetch weather data")
            raise

    def analyze(self, data: dict, location_name: str) -> dict:
        hourly = data.get("hourly", {})
        temps = hourly.get("temperature_2m", [])
        precip = hourly.get("precipitation", [])
        winds = hourly.get("windspeed_10m", [])
        times = hourly.get("time", [])

        n_temp_ok = sum(1 for t in temps if t is not None)
        n_precip_ok = sum(1 for p in precip if p is not None)
        n_wind_ok = sum(1 for w in winds if w is not None)
        missingness_pct = {
            "temperature_2m": round(100.0 * (1 - n_temp_ok / max(len(temps), 1)), 2),
            "precipitation": round(100.0 * (1 - n_precip_ok / max(len(precip), 1)), 2),
            "windspeed_10m": round(100.0 * (1 - n_wind_ok / max(len(winds), 1)), 2),
        }
        missingness_alerts = [k for k, v in missingness_pct.items() if v > 0]

        daily_summaries: list[dict] = []
        num_days = len(temps) // 24
        for d in range(num_days):
            s, e = d * 24, (d + 1) * 24
            day_temps = [t for t in temps[s:e] if t is not None]
            day_precip = [p for p in precip[s:e] if p is not None]
            day_winds = [w for w in winds[s:e] if w is not None]
            date_str = times[s][:10] if s < len(times) else str(d)
            daily_summaries.append({
                "date": date_str,
                "avg_temp": _mean(day_temps),
                "max_temp": max(day_temps) if day_temps else 0.0,
                "min_temp": min(day_temps) if day_temps else 0.0,
                "total_precipitation": sum(day_precip),
                "avg_windspeed": _mean(day_winds),
            })

        avg_temps = [s["avg_temp"] for s in daily_summaries]
        mean_temp = _mean(avg_temps)
        stdev_temp = _stdev(avg_temps)

        for s in daily_summaries:
            s["is_anomaly"] = (
                stdev_temp > 0 and abs(s["avg_temp"] - mean_temp) > ANOMALY_THRESHOLD * stdev_temp
            )

        rolling = [
            round(_mean(avg_temps[max(0, i - 2):i + 1]), 2)
            for i in range(len(avg_temps))
        ]

        overall = {
            "global_max_temp": max(avg_temps) if avg_temps else 0.0,
            "global_min_temp": min(avg_temps) if avg_temps else 0.0,
            "global_avg_temp": round(mean_temp, 2),
            "volatility": round(stdev_temp, 2),
            "rolling_3day_avg": rolling,
            "anomaly_days": [s["date"] for s in daily_summaries if s.get("is_anomaly")],
            "missingness_pct": missingness_pct,
            "missingness_alerts": missingness_alerts,
        }
        return {"daily": daily_summaries, "overall": overall, "location": location_name}

    def explain(self, analysis: dict, location_name: str) -> str:
        try:
            prompt = ChatPromptTemplate.from_template(EXPLAIN_PROMPT)
            result = (prompt | self._llm).invoke({"location": location_name, "data": str(analysis)})
            explanation = result.content or ""  # type: ignore[union-attr]
            logger.info("Generated weather explanation ({} chars)", len(explanation))
            return explanation
        except Exception:
            logger.exception("Weather explanation generation failed")
            return "Weather explanation could not be generated."

    def run(self, city_name: str, days: int = 7) -> WeatherAnalysis:
        """Geocode → fetch → analyse → explain."""
        try:
            geo = self.geocode_city(city_name)
            location_label = f"{geo['name']}, {geo['country']}"
            raw = self.fetch_weather(geo["latitude"], geo["longitude"], days)
            analysis = self.analyze(raw, location_label)
            explanation = self.explain(analysis, location_label)
            daily_models = [
                WeatherDaySummary(
                    date=d["date"], avg_temp=round(d["avg_temp"], 2),
                    max_temp=round(d["max_temp"], 2), min_temp=round(d["min_temp"], 2),
                    total_precipitation=round(d["total_precipitation"], 2),
                    avg_windspeed=round(d["avg_windspeed"], 2),
                    is_anomaly=d.get("is_anomaly", False),
                )
                for d in analysis["daily"]
            ]
            return WeatherAnalysis(
                location=location_label, country=geo["country"], timezone=geo["timezone"],
                period_days=days, daily_summary=daily_models,
                overall=analysis["overall"], explanation=explanation,
            )
        except ValueError as exc:
            logger.warning("Geocoding failed: {}", exc)
            return WeatherAnalysis(
                location=city_name, period_days=days,
                daily_summary=[], overall={}, explanation=str(exc),
            )
        except Exception:
            logger.exception("WeatherTool.run failed")
            return WeatherAnalysis(
                location=city_name, period_days=days,
                daily_summary=[], overall={}, explanation="Weather analysis could not be completed.",
            )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))
