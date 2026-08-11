"""
fetch_features.py

Step 1 of the AQI Feature Pipeline: dual-source fetch.

  - OpenWeather Air Pollution API = PRIMARY source. Updates ~hourly with
    real pollutant concentrations, so this is what actually feeds the
    model's hourly variation / change-rate features.
  - AQICN (WAQI) = SECONDARY / fallback + staleness check. Ground stations
    on AQICN often update on a multi-hour cycle, so hitting it every hour
    can return duplicate readings. We detect that and flag it rather than
    silently treating a repeated value as a fresh data point.

This is a LOCAL, MANUAL-RUN script for now. No cloud, no scheduling yet.
Run it a few times (spaced out) to build up rows and sanity-check the schema.
"""

import os
import csv
import json
import requests
from datetime import datetime, timezone

# ---- CONFIG ----
# OpenWeather: https://openweathermap.org/api/air-pollution
#   PowerShell temp:      $env:OPENWEATHER_API_KEY = "your_key_here"
#   PowerShell permanent: setx OPENWEATHER_API_KEY "your_key_here"
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# AQICN: https://aqicn.org/data-platform/token/
#   PowerShell temp:      $env:AQICN_API_TOKEN = "your_token_here"
#   PowerShell permanent: setx AQICN_API_TOKEN "your_token_here"
AQICN_API_TOKEN = os.environ.get("AQICN_API_TOKEN")

# Karachi coordinates / city name
LAT = 24.8607
LON = 67.0011
CITY = "karachi"

CSV_PATH = "aqi_features.csv"
STATE_PATH = "last_aqicn_state.json"  # tracks last AQICN reading for staleness check

OPENWEATHER_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
AQICN_URL = f"https://api.waqi.info/feed/{CITY}/"


# ---------------------------------------------------------------------------
# OpenWeather (primary)
# ---------------------------------------------------------------------------
def fetch_openweather():
    """Fetch pollutant concentrations from OpenWeather. Returns dict or None."""
    if not OPENWEATHER_API_KEY:
        print("  [OpenWeather] Skipped: OPENWEATHER_API_KEY not set.")
        return None

    try:
        params = {"lat": LAT, "lon": LON, "appid": OPENWEATHER_API_KEY}
        response = requests.get(OPENWEATHER_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        entry = data["list"][0]
        components = entry["components"]
        dt = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)

        return {
            "ow_timestamp_utc": dt.isoformat(),
            "ow_aqi_index": entry["main"]["aqi"],  # OpenWeather's 1-5 scale
            "ow_co": components.get("co"),
            "ow_no": components.get("no"),
            "ow_no2": components.get("no2"),
            "ow_o3": components.get("o3"),
            "ow_so2": components.get("so2"),
            "ow_pm2_5": components.get("pm2_5"),
            "ow_pm10": components.get("pm10"),
            "ow_nh3": components.get("nh3"),
        }
    except requests.exceptions.HTTPError as e:
        print(f"  [OpenWeather] HTTP error (key may still be inactive): {e}")
        return None
    except Exception as e:
        print(f"  [OpenWeather] Failed: {e}")
        return None


# ---------------------------------------------------------------------------
# AQICN (secondary / staleness check)
# ---------------------------------------------------------------------------
def load_last_aqicn_state():
    if os.path.isfile(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return None


def save_last_aqicn_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)


def fetch_aqicn():
    """Fetch AQI + pollutant readings from AQICN, with staleness detection.

    Returns dict or None. Includes 'aqicn_is_stale': True if this reading's
    station timestamp matches the last time we fetched (i.e. no new ground
    station update happened between calls).
    """
    if not AQICN_API_TOKEN:
        print("  [AQICN] Skipped: AQICN_API_TOKEN not set.")
        return None

    try:
        params = {"token": AQICN_API_TOKEN}
        response = requests.get(AQICN_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            print(f"  [AQICN] API returned an error: {data}")
            return None

        raw = data["data"]
        iaqi = raw.get("iaqi", {})
        station_dt_str = raw.get("time", {}).get("s")  # station's own timestamp

        def get_val(pollutant):
            entry = iaqi.get(pollutant)
            return entry.get("v") if entry else None

        # --- staleness check: compare this station timestamp to the last one we saved ---
        last_state = load_last_aqicn_state()
        is_stale = bool(last_state and last_state.get("station_timestamp") == station_dt_str)

        save_last_aqicn_state({"station_timestamp": station_dt_str, "aqi": raw.get("aqi")})

        return {
            "aqicn_station_timestamp": station_dt_str,
            "aqicn_aqi": raw.get("aqi"),
            "aqicn_pm25": get_val("pm25"),
            "aqicn_pm10": get_val("pm10"),
            "aqicn_o3": get_val("o3"),
            "aqicn_no2": get_val("no2"),
            "aqicn_so2": get_val("so2"),
            "aqicn_co": get_val("co"),
            "aqicn_dominant_pollutant": raw.get("dominentpol"),
            "aqicn_is_stale": is_stale,
        }
    except Exception as e:
        print(f"  [AQICN] Failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Change-rate feature (needs the previous row for comparison)
# ---------------------------------------------------------------------------
def get_last_row_from_csv(path=CSV_PATH):
    """Return the last row of the CSV as a dict, or None if the file doesn't
    exist yet or has no data rows."""
    if not os.path.isfile(path):
        return None

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return rows[-1] if rows else None


def compute_change_rate_features(current_row, last_row):
    """Compute AQI/pollutant change-rate features by comparing against the
    previous saved row. Rate is normalized per hour so gaps between runs
    (5 min vs 3 hours) don't distort the numbers.

    Returns a dict of change-rate fields. Values are None if there's no
    previous row, or if either reading is missing.
    """
    change_features = {
        "aqi_change_rate": None,
        "pm25_change_rate": None,
        "hours_since_last_reading": None,
    }

    if not last_row:
        return change_features  # first-ever row, nothing to compare against

    try:
        current_time = datetime.fromisoformat(current_row["fetched_at_utc"])
        last_time = datetime.fromisoformat(last_row["fetched_at_utc"])
        hours_elapsed = (current_time - last_time).total_seconds() / 3600.0

        if hours_elapsed <= 0:
            return change_features  # guard against clock issues / duplicate runs

        change_features["hours_since_last_reading"] = round(hours_elapsed, 3)

        # AQI change rate: prefer OpenWeather's 1-5 index since it's the primary source
        curr_aqi = current_row.get("ow_aqi_index")
        prev_aqi = last_row.get("ow_aqi_index")
        if curr_aqi not in (None, "") and prev_aqi not in (None, ""):
            rate = (float(curr_aqi) - float(prev_aqi)) / hours_elapsed
            change_features["aqi_change_rate"] = round(rate, 4)

        # PM2.5 change rate: finer-grained than the 1-5 index, useful as a second signal
        curr_pm25 = current_row.get("ow_pm2_5")
        prev_pm25 = last_row.get("ow_pm2_5")
        if curr_pm25 not in (None, "") and prev_pm25 not in (None, ""):
            rate = (float(curr_pm25) - float(prev_pm25)) / hours_elapsed
            change_features["pm25_change_rate"] = round(rate, 4)

    except (KeyError, ValueError, TypeError) as e:
        print(f"  [change-rate] Could not compute: {e}")

    return change_features



def build_feature_row(ow_data, aqicn_data):
    """Merge both sources into one row with time features + a data_source flag."""
    now = datetime.now(timezone.utc)

    row = {
        "fetched_at_utc": now.isoformat(),
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),
    }

    # Decide primary AQI-driving source for this row
    if ow_data:
        row["data_source"] = "openweather"
    elif aqicn_data and not aqicn_data.get("aqicn_is_stale"):
        row["data_source"] = "aqicn_fresh"
    elif aqicn_data:
        row["data_source"] = "aqicn_stale"
    else:
        row["data_source"] = "none"

    # Flatten both sources' fields into the row (prefixed, so nothing collides)
    if ow_data:
        row.update(ow_data)
    if aqicn_data:
        row.update(aqicn_data)

    return row


def append_to_csv(row, path=CSV_PATH):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    print("Fetching from OpenWeather (primary)...")
    ow_data = fetch_openweather()

    print("Fetching from AQICN (secondary / staleness check)...")
    aqicn_data = fetch_aqicn()

    if aqicn_data and aqicn_data.get("aqicn_is_stale"):
        print("  [AQICN] WARNING: station timestamp unchanged since last fetch "
              "-> this reading is STALE, flagged accordingly, not treated as new signal.")

    row = build_feature_row(ow_data, aqicn_data)

    # Compute change-rate features by comparing against the last saved row,
    # BEFORE we append the new row (otherwise we'd be comparing to ourselves).
    last_row = get_last_row_from_csv()
    change_features = compute_change_rate_features(row, last_row)
    row.update(change_features)

    append_to_csv(row)

    print("\nSaved feature row:")
    for k, v in row.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {CSV_PATH}")


if __name__ == "__main__":
    main()