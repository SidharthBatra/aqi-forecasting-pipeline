"""
fetch_features.py

Step 1 of the AQI Feature Pipeline: cloud-based hourly fetch that writes
directly into the Hopsworks feature group train_model.py trains from
(aqi_karachi_features) -- no local files, so this runs unattended on
GitHub Actions.

REWRITE (2026-08-25) of the original local-CSV prototype. Two real
problems fixed, plus one latent bug caught along the way:

1. STATE PERSISTENCE. The original script tracked the previous reading
   (for change-rate features) and AQICN's last station timestamp (for
   staleness detection) in local files (aqi_features.csv,
   last_aqicn_state.json). GitHub Actions runners are stateless -- a
   fresh machine every hour, nothing carries over between runs. State now
   lives in Hopsworks itself: this script reads back the last
   CONTEXT_HOURS of rows from the feature group before computing
   anything, instead of relying on local files.

2. SCHEMA / TRAIN-SERVE CONSISTENCY. The original script wrote columns
   (ow_pm2_5, ow_aqi_index -- OpenWeather's 1-5 category, aqicn_*) that
   don't match aqi_karachi_features' actual schema, and never computed
   the continuous EPA-formula 'aqi' that IS the model's real target.
   Pushed as-is, live rows would have been unusable for training -- wrong
   columns, wrong units. Fixed by importing compute_true_aqi.py's exact
   EPA breakpoint + rolling-window functions (not reimplementing them) so
   live 'aqi' is derived identically to how the backfill's target was
   built -- including the 24h/8h TIME-BASED rolling windows, which need
   the last ~24h of context, not just this hour's instant reading. That
   context is what the Hopsworks read in fix #1 provides.

3. LATENT UNIT-MISMATCH BUG. AQICN's API returns each pollutant's own AQI
   SUB-INDEX ("iaqi", already on a 0-500 scale), not a raw concentration
   in ug/m3. The original script's structure implied AQICN readings could
   fill in for OpenWeather's pollutant fields on failure -- but that would
   silently mix AQI sub-indices into columns that are supposed to hold raw
   ug/m3 concentrations (what the training data and the EPA formula both
   expect). So AQICN here is a cross-check / staleness signal ONLY --
   logged, never written into the pollutant columns. If OpenWeather fails
   for an hour, this script skips that hour's Feature Store insert rather
   than filling it with wrong-unit AQICN data or a fabricated value. A
   documented missing hour is a gap train_model.py's hourly-grid
   reindexing already handles correctly; a silently wrong hour is not.

NOTE: like the other Hopsworks-touching scripts written this session,
this hasn't been run against your live project from this environment --
only syntax-checked and logic-smoke-tested with synthetic context data.

Requires HOPSWORKS_API_KEY and OPENWEATHER_API_KEY as environment
variables (GitHub Actions secrets in CI, Codespaces secrets locally).
AQICN_API_TOKEN is optional -- if unset, the cross-check/staleness log is
skipped but the pipeline still runs fine on OpenWeather alone.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from compute_true_aqi import (
    POLLUTANT_COLUMNS,
    clean_sentinel_values,
    compute_sub_indices,
    categorize,
)

# ---- CONFIG ----
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
AQICN_API_TOKEN = os.environ.get("AQICN_API_TOKEN")
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")

LAT = 24.8607
LON = 67.0011
CITY = "karachi"

OPENWEATHER_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
AQICN_URL = f"https://api.waqi.info/feed/{CITY}/"

TIMESTAMP_COL = "timestamp_utc"

# Must match ingest_to_hopsworks.py / train_model.py's raw feature group.
RAW_FEATURE_GROUP_NAME = "aqi_karachi_features"
RAW_FEATURE_GROUP_VERSION = 4

# How far back to pull for the EPA rolling-window context. 30h gives
# margin over the 24h PM2.5/PM10 window even if an hour or two is missing.
CONTEXT_HOURS = 30

# AQICN ground stations update on a multi-hour cycle (per the mentor's
# brief) -- flag a station reading older than this as stale rather than
# treat it as a fresh corroborating signal.
AQICN_STALE_THRESHOLD_HOURS = 3


def fetch_openweather():
    """Raw pollutant concentrations (ug/m3) + OpenWeather's own 1-5
    category, straight from the Air Pollution API. Returns None on any
    failure -- caller decides what to do (currently: skip this hour)."""
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
            "dt": dt,
            "aqi_index_openweather_1to5": entry["main"]["aqi"],
            "co": components.get("co"),
            "no": components.get("no"),
            "no2": components.get("no2"),
            "o3": components.get("o3"),
            "so2": components.get("so2"),
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "nh3": components.get("nh3"),
        }
    except requests.exceptions.HTTPError as e:
        print(f"  [OpenWeather] HTTP error (key may still be inactive): {e}")
        return None
    except Exception as e:
        print(f"  [OpenWeather] Failed: {e}")
        return None


def fetch_aqicn():
    """AQICN cross-check + staleness signal ONLY -- see module docstring
    (fix #3) on why its 'iaqi' values never get written into the
    pollutant columns."""
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
        station_dt_str = raw.get("time", {}).get("s")
        is_stale = None
        if station_dt_str:
            try:
                station_dt = datetime.strptime(station_dt_str, "%Y-%m-%d %H:%M:%S")
                station_dt = station_dt.replace(tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - station_dt).total_seconds() / 3600.0
                is_stale = age_hours > AQICN_STALE_THRESHOLD_HOURS
            except ValueError:
                pass

        return {
            "aqicn_aqi": raw.get("aqi"),
            "aqicn_dominant_pollutant": raw.get("dominentpol"),
            "aqicn_station_timestamp": station_dt_str,
            "aqicn_is_stale": is_stale,
        }
    except Exception as e:
        print(f"  [AQICN] Failed: {e}")
        return None


def fetch_recent_context(fs):
    """Pulls the last CONTEXT_HOURS from the Hopsworks feature group --
    this is the Hopsworks-backed replacement for the original script's
    local-file state, and supplies the history the EPA rolling-window
    calc and change-rate features both need. Empty DataFrame if the
    feature group has no rows yet or the read fails (script still runs;
    compute_live_aqi() degrades to using only this hour's reading)."""
    try:
        fg = fs.get_feature_group(name=RAW_FEATURE_GROUP_NAME, version=RAW_FEATURE_GROUP_VERSION)
        df = fg.read()
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=CONTEXT_HOURS)
        recent = df[df[TIMESTAMP_COL] >= cutoff].sort_values(TIMESTAMP_COL)
        print(f"  Pulled {len(recent)} rows of context from the last {CONTEXT_HOURS}h.")
        return recent
    except Exception as e:
        print(
            f"  WARNING: couldn't read recent context from Hopsworks "
            f"({type(e).__name__}: {e}) -- computing 'aqi' from this hour's "
            f"reading alone, with no rolling-window history."
        )
        return pd.DataFrame()


def compute_live_aqi(context_df, ow_data):
    """Appends the new OpenWeather reading to the recent context and runs
    the SAME EPA rolling-window computation compute_true_aqi.py uses on
    the backfill, so live 'aqi' is derived identically to training 'aqi'.
    Returns (aqi, dominant_pollutant, category)."""
    new_row = {col: ow_data.get(col) for col in POLLUTANT_COLUMNS.values()}
    new_row[TIMESTAMP_COL] = ow_data["dt"]
    combined = pd.concat([context_df, pd.DataFrame([new_row])], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP_COL).drop_duplicates(subset=[TIMESTAMP_COL], keep="last")
    combined = clean_sentinel_values(combined)
    combined_indexed = combined.set_index(TIMESTAMP_COL)

    sub_indices = compute_sub_indices(combined_indexed)
    last = sub_indices.iloc[-1]
    aqi = float(last.max())
    dominant = str(last.idxmax())
    category = categorize(aqi)
    return aqi, dominant, category


def compute_change_rates(context_df, current_dt, current_aqi, current_pm25):
    """AQI/PM2.5 change rate vs. the most recent row already in Hopsworks,
    normalized per hour. (None, None) if there's no prior row or elapsed
    time is non-positive (clock issue / duplicate run within the hour)."""
    if context_df.empty or "aqi" not in context_df.columns:
        return None, None
    prev = context_df.sort_values(TIMESTAMP_COL).iloc[-1]
    prev_dt = prev[TIMESTAMP_COL]
    hours_elapsed = (current_dt - prev_dt).total_seconds() / 3600.0
    if hours_elapsed <= 0:
        return None, None

    aqi_rate = None
    if pd.notna(prev.get("aqi")) and current_aqi is not None:
        aqi_rate = round((current_aqi - prev["aqi"]) / hours_elapsed, 4)

    pm25_rate = None
    if pd.notna(prev.get("pm2_5")) and current_pm25 is not None:
        pm25_rate = round((current_pm25 - prev["pm2_5"]) / hours_elapsed, 4)

    return aqi_rate, pm25_rate


def build_row(ow_data, context_df):
    dt = ow_data["dt"]
    aqi, dominant, category = compute_live_aqi(context_df, ow_data)
    aqi_rate, pm25_rate = compute_change_rates(context_df, dt, aqi, ow_data.get("pm2_5"))

    return {
        TIMESTAMP_COL: dt.isoformat(),
        "hour": dt.hour,
        "day": dt.day,
        "month": dt.month,
        "year": dt.year,
        "day_of_week": dt.weekday(),
        "co": ow_data.get("co"),
        "no": ow_data.get("no"),
        "no2": ow_data.get("no2"),
        "o3": ow_data.get("o3"),
        "so2": ow_data.get("so2"),
        "pm2_5": ow_data.get("pm2_5"),
        "pm10": ow_data.get("pm10"),
        "nh3": ow_data.get("nh3"),
        "aqi_index_openweather_1to5": ow_data.get("aqi_index_openweather_1to5"),
        # Open-Meteo is a backfill-only cross-check source (see
        # backfill_historical.py) -- never available live, so these stay
        # None. Already excluded from model features in train_model.py
        # for exactly this train-serve-skew reason.
        "om_pm2_5": None,
        "om_pm10": None,
        "om_co": None,
        "om_no2": None,
        "om_so2": None,
        "om_o3": None,
        "om_us_aqi": None,
        "pm25_source_diff": None,
        "pm25_source_diff_pct": None,
        "aqi": aqi,
        "aqi_dominant_pollutant": dominant,
        "aqi_category": category,
        "aqi_change_rate": aqi_rate,
        "pm25_change_rate": pm25_rate,
        # Derived from the Unix timestamp, not sequential -- avoids any
        # collision with the backfill's sequential 0..16751 row_ids.
        "row_id": int(dt.timestamp()),
    }


def main():
    if not HOPSWORKS_API_KEY:
        print("ERROR: HOPSWORKS_API_KEY not set.")
        sys.exit(1)

    import hopsworks

    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    print("Fetching from OpenWeather (primary)...")
    ow_data = fetch_openweather()

    print("Fetching from AQICN (cross-check / staleness only)...")
    aqicn_data = fetch_aqicn()
    if aqicn_data:
        if aqicn_data.get("aqicn_is_stale"):
            print(
                f"  [AQICN] STALE: station reading is older than "
                f"{AQICN_STALE_THRESHOLD_HOURS}h ({aqicn_data.get('aqicn_station_timestamp')}) "
                f"-- logged for visibility, not used quantitatively."
            )
        else:
            print(
                f"  [AQICN] aqi={aqicn_data.get('aqicn_aqi')}, "
                f"dominant={aqicn_data.get('aqicn_dominant_pollutant')}"
            )

    if not ow_data:
        print(
            "\nNo OpenWeather reading available this hour -- skipping this "
            "hour's Feature Store insert rather than writing a wrong-unit or "
            "incomplete row. This shows up as a real gap in the data; "
            "train_model.py's hourly-grid reindexing already handles gaps "
            "correctly."
        )
        sys.exit(0)  # a documented skipped hour, not a pipeline failure

    print(f"\nPulling last {CONTEXT_HOURS}h of context from Hopsworks...")
    context_df = fetch_recent_context(fs)

    row = build_row(ow_data, context_df)

    print("\nNew feature row:")
    for k, v in row.items():
        print(f"  {k}: {v}")

    row_df = pd.DataFrame([row])
    row_df[TIMESTAMP_COL] = pd.to_datetime(row_df[TIMESTAMP_COL], utc=True)

    print(f"\nInserting into Hopsworks feature group {RAW_FEATURE_GROUP_NAME} v{RAW_FEATURE_GROUP_VERSION}...")
    fg = fs.get_feature_group(name=RAW_FEATURE_GROUP_NAME, version=RAW_FEATURE_GROUP_VERSION)
    fg.insert(row_df, write_options={"wait_for_job": False})
    print("Insert submitted.")


if __name__ == "__main__":
    main()