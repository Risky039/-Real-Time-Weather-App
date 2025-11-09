"""
Weather App — OpenWeatherMap

Supports:
1) Streamlit UI (real-time weather & forecast)
2) CLI fallback (offline or live)
3) Unit tests for core functionality
"""

from __future__ import annotations
import argparse
import importlib.util
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests

# -----------------------------
# Config & Constants
# -----------------------------
BASE_URL = "https://api.openweathermap.org/data/2.5"
DEFAULT_CITY = "Lagos"
DEFAULT_UNITS = "metric"  # or "imperial"

EMOJI = {
    "Thunderstorm": "⛈️",
    "Drizzle": "🌦️",
    "Rain": "🌧️",
    "Snow": "❄️",
    "Clear": "☀️",
    "Clouds": "☁️",
    "Mist": "🌫️",
    "Smoke": "🌫️",
    "Haze": "🌫️",
    "Dust": "🌫️",
    "Fog": "🌁",
    "Sand": "🌫️",
    "Ash": "🌋",
    "Squall": "💨",
    "Tornado": "🌪️",
}

# Offline sample data
SAMPLE_CURRENT: Dict[str, Any] = {
    "name": "Lagos",
    "sys": {"country": "NG"},
    "dt": 1_700_000_000,
    "main": {"temp": 30.5, "feels_like": 33.1, "humidity": 65, "pressure": 1011},
    "wind": {"speed": 4.2},
    "weather": [{"main": "Clear", "description": "clear sky"}],
}

SAMPLE_FORECAST: Dict[str, Any] = {
    "list": [
        {"dt": 1_700_000_000, "main": {"temp": 30.0, "feels_like": 32.0, "humidity": 60}, "wind": {"speed": 4.5}, "weather": [{"description": "clear sky"}]},
        {"dt": 1_700_010_800, "main": {"temp": 29.0, "feels_like": 31.0, "humidity": 62}, "wind": {"speed": 4.0}, "weather": [{"description": "few clouds"}]},
        {"dt": 1_700_021_600, "main": {"temp": 28.5, "feels_like": 30.0, "humidity": 64}, "wind": {"speed": 3.6}, "weather": [{"description": "scattered clouds"}]},
        {"dt": 1_700_108_400, "main": {"temp": 31.0, "feels_like": 33.5, "humidity": 58}, "wind": {"speed": 5.2}, "weather": [{"description": "light rain"}]},
        {"dt": 1_700_119_200, "main": {"temp": 30.2, "feels_like": 32.1, "humidity": 60}, "wind": {"speed": 4.8}, "weather": [{"description": "broken clouds"}]},
        {"dt": 1_700_130_000, "main": {"temp": 29.7, "feels_like": 31.4, "humidity": 63}, "wind": {"speed": 4.1}, "weather": [{"description": "overcast clouds"}]},
    ]
}

# -----------------------------
# Utility Functions
# -----------------------------
def fmt_temp(val: float, units: str) -> str:
    unit_symbol = "C" if units == "metric" else "F"
    try:
        return f"{round(float(val))}°{unit_symbol}"
    except Exception:
        return f"--°{unit_symbol}"

def has_streamlit() -> bool:
    return importlib.util.find_spec("streamlit") is not None

def fetch_current_weather(city: str, api_key: str, units: str) -> Tuple[Dict[str, Any] | None, str | None]:
    if not city or not api_key:
        return None, "Missing city or API key"
    try:
        r = requests.get(f"{BASE_URL}/weather", params={"q": city, "appid": api_key, "units": units}, timeout=20)
        if r.status_code != 200:
            try:
                err = r.json()
            except Exception:
                err = {"message": r.text}
            return None, err.get("message", f"Error {r.status_code}")
        return r.json(), None
    except Exception as e:
        return None, str(e)

def fetch_forecast(city: str, api_key: str, units: str) -> Tuple[Dict[str, Any] | None, str | None]:
    if not city or not api_key:
        return None, "Missing city or API key"
    try:
        r = requests.get(f"{BASE_URL}/forecast", params={"q": city, "appid": api_key, "units": units}, timeout=20)
        if r.status_code != 200:
            try:
                err = r.json()
            except Exception:
                err = {"message": r.text}
            return None, err.get("message", f"Error {r.status_code}")
        return r.json(), None
    except Exception as e:
        return None, str(e)

def transform_forecast_to_df(forecast: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in forecast.get("list", []):
        rows.append({
            "datetime": pd.to_datetime(item.get("dt", 0), unit="s", utc=True),
            "temp": item.get("main", {}).get("temp"),
            "feels_like": item.get("main", {}).get("feels_like"),
            "humidity": item.get("main", {}).get("humidity"),
            "wind": item.get("wind", {}).get("speed"),
            "description": str(item.get("weather", [{}])[0].get("description", "")).title(),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("datetime").sort_index()
    return df

def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    agg = df.resample("D").agg({"temp": ["min", "max", "mean"], "humidity": "mean", "wind": "mean"})
    agg.columns = ["_".join(col) if isinstance(col, tuple) else col for col in agg.columns]
    return agg

# -----------------------------
# Streamlit UI
# -----------------------------
def run_streamlit_app() -> None:
    import streamlit as st
    st.set_page_config(page_title="Weather App", page_icon="⛅", layout="centered")
    st.title("⛅ Real-Time Weather App")
    st.caption("Powered by OpenWeatherMap")

    api_key = os.getenv("261feb8852bdfa1f1a0006b95db21a40", "")
    if not api_key:
        with st.expander("Set your API key (session only)"):
            api_key = st.text_input("OpenWeatherMap API key:", type="password")
            st.info("Get a free key at https://openweathermap.org/")

    with st.sidebar:
        st.header("Settings")
        city = st.text_input("City name", value=DEFAULT_CITY)
        units_label = st.radio("Units", ["Metric (°C)", "Imperial (°F)"], index=0)
        units = "metric" if units_label.startswith("Metric") else "imperial"
        show_forecast = st.checkbox("Show 5-day forecast", value=True)

    fetch_current_cached = fetch_current_weather
    fetch_forecast_cached = fetch_forecast
    try:
        fetch_current_cached = st.cache_data(ttl=600)(fetch_current_weather)
        fetch_forecast_cached = st.cache_data(ttl=600)(fetch_forecast)
    except Exception:
        pass

    if api_key:
        with st.spinner("Fetching weather..."):
            current, err = fetch_current_cached(city, api_key, units)
    else:
        current, err = None, "Enter your API key to continue"

    if err:
        st.error(err)
    elif current:
        name = current.get("name", city)
        country = current.get("sys", {}).get("country", "")
        main = current.get("main", {})
        wind = current.get("wind", {})
        w = current.get("weather", [{}])[0]
        conditions = w.get("main", "")
        description = w.get("description", "").title()
        emoji = EMOJI.get(conditions, "🌍")
        dt = datetime.fromtimestamp(current.get("dt", 0)).strftime("%A, %d %B %Y %H:%M")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader(f"{emoji} {name}, {country}")
            st.write(dt)
            st.metric("Temperature", fmt_temp(main.get("temp", 0), units), f"Feels like {fmt_temp(main.get('feels_like', 0), units)}")
        with col2:
            st.write(f"**Conditions:** {description}")
            st.write(f"**Humidity:** {main.get('humidity', 0)}%")
            st.write(f"**Pressure:** {main.get('pressure', 0)} hPa")
            st.write(f"**Wind:** {wind.get('speed', 0)} {'m/s' if units=='metric' else 'mph'}")
            st.progress(min(100, int(main.get("humidity", 0))))

    if api_key and show_forecast:
        forecast, ferr = fetch_forecast_cached(city, api_key, units)
        if ferr:
            st.warning(f"Forecast unavailable: {ferr}")
        else:
            df = transform_forecast_to_df(forecast)
            if not df.empty:
                st.subheader("5-Day / 3-Hour Forecast")
                st.line_chart(df[["temp", "feels_like"]])
                daily = daily_summary(df)
                units_suffix = "°C" if units == "metric" else "°F"
                daily = daily.rename(columns={
                    "temp_min": f"Min Temp ({units_suffix})",
                    "temp_max": f"Max Temp ({units_suffix})",
                    "temp_mean": f"Avg Temp ({units_suffix})",
                    "humidity_mean": "Avg Humidity (%)",
                    "wind_mean": f"Avg Wind ({'m/s' if units=='metric' else 'mph'})",
                })
                st.dataframe(daily.style.format(precision=1))

    st.divider()
    st.markdown(
        """
**Tips**
- Set your `OPENWEATHER_API_KEY` environment variable:
  - Windows: `setx OPENWEATHER_API_KEY "YOUR_KEY"`
  - Linux/macOS: `export OPENWEATHER_API_KEY=YOUR_KEY`
- Run: `streamlit run app.py`
"""
    )

# -----------------------------
# CLI Fallback
# -----------------------------
def run_cli(city: str, units: str, offline: bool) -> None:
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    if offline or not api_key:
        current, cerr = SAMPLE_CURRENT, None
        forecast, ferr = SAMPLE_FORECAST, None
        note = "(offline sample data)"
    else:
        current, cerr = fetch_current_weather(city, api_key, units)
        forecast, ferr = fetch_forecast(city, api_key, units)
        note = "(live)" if not (cerr or ferr) else "(partial)"

    print(f"\n⛅ Weather App CLI {note}")
    print(f"City: {city} | Units: {units}")

    if cerr:
        print(f"Current weather error: {cerr}")
    else:
        name = current.get("name", city)
        country = current.get("sys", {}).get("country", "")
        w = current.get("weather", [{}])[0]
        conditions = w.get("main", "")
        description = w.get("description", "").title()
        emoji = EMOJI.get(conditions, "🌍")
        main = current.get("main", {})
        wind = current.get("wind", {})
        dt = datetime.fromtimestamp(current.get("dt", 0), tz=timezone.utc)
        print(f"{emoji} {name}, {country} — {dt:%Y-%m-%d %H:%M UTC}")
        print(f"  Temp: {fmt_temp(main.get('temp', 0), units)} (feels like {fmt_temp(main.get('feels_like', 0), units)})")
        print(f"  Conditions: {description}")
        print(f"  Humidity: {main.get('humidity', 0)}% | Pressure: {main.get('pressure', 0)} hPa | Wind: {wind.get('speed', 0)} {'m/s' if units=='metric' else 'mph'}")

    if ferr:
        print(f"Forecast error: {ferr}")
    else:
        df = transform_forecast_to_df(forecast)
        if df.empty:
            print("No forecast data available.")
        else:
            daily = daily_summary(df)
            print("\nDaily Summary:")
            for idx, row in daily.iterrows():
                date = idx.strftime("%Y-%m-%d")
                print(
                    f"  {date} | Min {row['temp_min']:.1f}, Max {row['temp_max']:.1f}, Avg {row['temp_mean']:.1f} | "
                    f"Humidity {row['humidity_mean']:.0f}% | Wind {row['wind_mean']:.1f}"
                )

# -----------------------------
# Unit Tests
# -----------------------------
def _build_sample_df() -> pd.DataFrame:
    return transform_forecast_to_df(SAMPLE_FORECAST)

def run_tests() -> bool:
    import unittest

    class WeatherTests(unittest.TestCase):
        def test_fmt_temp_metric(self):
            self.assertEqual(fmt_temp(30.4, "metric"), "30°C")
            self.assertEqual(fmt_temp(30.6, "metric"), "31°C")

        def test_fmt_temp_imperial(self):
            self.assertEqual(fmt_temp(86.2, "imperial"), "86°F")

        def test_emoji_mapping(self):
            self.assertEqual(EMOJI.get("Rain"), "🌧️")
            self.assertIn("Clear", EMOJI)

        def test_transform_forecast_to_df(self):
            df = _build_sample_df()
            self.assertFalse(df.empty)
            self.assertIn("temp", df.columns)
            self.assertIn("feels_like", df.columns)
            self.assertIn("humidity", df.columns)
            self.assertIn("wind", df.columns)
            self.assertIn("description", df.columns)

        def test_daily_summary(self):
            df = _build_sample_df()
            daily = daily_summary(df)
            self.assertFalse(daily.empty)
            self.assertIn("temp_min", daily.columns)
            self.assertIn("temp_max", daily.columns)
            self.assertIn("temp_mean", daily.columns)
            self.assertIn("humidity_mean", daily.columns)
            self.assertIn("wind_mean", daily.columns)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(WeatherTests)
    result = unittest.TextTestRunner().run(suite)
    return result.wasSuccessful()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather App")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    parser.add_argument("--offline", action="store_true", help="Use offline sample data")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--city", type=str, default=DEFAULT_CITY, help="City name")
    parser.add_argument("--units", type=str, default=DEFAULT_UNITS, choices=["metric", "imperial"], help="Units")
    args = parser.parse_args()

    if args.test or os.getenv("RUN_TESTS") == "1":
        success = run_tests()
        sys.exit(0 if success else 1)

    if args.cli or not has_streamlit():
        run_cli(args.city, args.units, args.offline)
    else:
        run_streamlit_app()
