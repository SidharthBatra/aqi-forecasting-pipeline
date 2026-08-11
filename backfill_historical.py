"""
backfill_historical.py

Historical Data Backfill: pulls past hourly pollutant data from OpenWeather's
Air Pollution History API and builds a training-ready CSV in one shot,
instead of waiting months for live collection.

OpenWeather's free tier includes history back to 27 Nov 2020, hourly
granularity, in a single call per date range (not one call per hour).

Usage:
    python backfill_historical.py

Adjust BACKFILL_MONTHS below to control how far back to pull (6-24 typical).
"""

import os
import csv
import time
import requests
from datetime import datetime, timedelta, timezone

# ---- CONFIG ----
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

LAT = 24.8607
LON = 67.0011

# Mentor's target: 6 months minimum, 2 years ideal. Set to 24 for the full run.
BACKFILL_MONTHS = 24

OUTPUT_CSV = "aqi_historical_backfill.csv"

HISTORY_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

# OpenWeather's earliest available date for this API
EARLIEST_AVAILABLE = datetime(2020, 11, 27, tzinfo=timezone.utc)

# Some OpenWeather plans cap how many days you can request in a single call.
# Chunking by 30-day windows keeps each request small and avoids hitting
# any response-size or range limits, and makes retries cheap if one chunk fails.
CHUNK_DAYS = 30

# Open-Meteo Air Quality API - no key required, CAMS reanalysis model.
# Used here as a secondary source: cross-checks OpenWeather's historical
# values and can fill gaps if an OpenWeather chunk fails.
OPEN_METEO_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Open-Meteo handles wide date ranges comfortably in one call; still chunk
# for consistency with the OpenWeather loop and to keep retries cheap.
OPEN_METEO_CHUNK_DAYS = 90


def month_delta(dt, months):
    """Subtract `months` calendar months from a datetime (approximate, day-based)."""
    return dt - timedelta(days=months * 30)


def fetch_history_chunk(start_dt, end_dt):
    """Fetch one chunk of historical data. Returns list of hourly records."""
    params = {
        "lat": LAT,
        "lon": LON,
        "start": int(start_dt.timestamp()),
        "end": int(end_dt.timestamp()),
        "appid": OPENWEATHER_API_KEY,
    }

    response = requests.get(HISTORY_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("list", [])


def fetch_open_meteo_chunk(start_dt, end_dt):
    """Fetch one chunk of historical air quality data from Open-Meteo.
    Returns a dict keyed by ISO-hour timestamp -> pollutant values.
    """
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "timezone": "UTC",
    }

    response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    result = {}
    for i, t in enumerate(times):
        # Open-Meteo gives "2024-01-01T00:00" (no seconds, no tz suffix) - normalize
        dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        key = dt.isoformat()
        result[key] = {
            "om_pm2_5": hourly.get("pm2_5", [None] * len(times))[i],
            "om_pm10": hourly.get("pm10", [None] * len(times))[i],
            "om_co": hourly.get("carbon_monoxide", [None] * len(times))[i],
            "om_no2": hourly.get("nitrogen_dioxide", [None] * len(times))[i],
            "om_so2": hourly.get("sulphur_dioxide", [None] * len(times))[i],
            "om_o3": hourly.get("ozone", [None] * len(times))[i],
            "om_us_aqi": hourly.get("us_aqi", [None] * len(times))[i],
        }
    return result


def fetch_all_open_meteo(overall_start, overall_end):
    """Fetch Open-Meteo data across the full backfill range, chunked."""
    all_data = {}
    chunk_start = overall_start

    while chunk_start < overall_end:
        chunk_end = min(chunk_start + timedelta(days=OPEN_METEO_CHUNK_DAYS), overall_end)
        print(f"  [Open-Meteo] Fetching {chunk_start.date()} -> {chunk_end.date()}...")
        try:
            chunk_data = fetch_open_meteo_chunk(chunk_start, chunk_end)
            print(f"    Got {len(chunk_data)} hourly records")
            all_data.update(chunk_data)
        except Exception as e:
            print(f"    [Open-Meteo] Failed on this chunk, skipping: {e}")

        chunk_start = chunk_end
        time.sleep(0.5)

    return all_data


def merge_open_meteo(rows, om_data):
    """Attach Open-Meteo fields to each OpenWeather row by matching hour
    timestamp, and compute a cross-check diff on PM2.5 so large disagreements
    between the two sources are visible rather than silently trusted."""
    matched = 0
    for row in rows:
        # Round OpenWeather's timestamp down to the hour to match Open-Meteo's hourly keys
        ow_time = datetime.fromisoformat(row["timestamp_utc"])
        hour_key = ow_time.replace(minute=0, second=0, microsecond=0).isoformat()

        om_row = om_data.get(hour_key)
        if om_row:
            row.update(om_row)
            matched += 1
            if row.get("pm2_5") is not None and om_row.get("om_pm2_5") is not None:
                row["pm25_source_diff"] = round(abs(row["pm2_5"] - om_row["om_pm2_5"]), 3)
            else:
                row["pm25_source_diff"] = None
        else:
            for field in ("om_pm2_5", "om_pm10", "om_co", "om_no2", "om_so2", "om_o3", "om_us_aqi"):
                row[field] = None
            row["pm25_source_diff"] = None

    print(f"  Matched Open-Meteo data for {matched}/{len(rows)} rows")
    return rows


def record_to_row(entry):
    """Convert one OpenWeather history record into a flat feature row."""
    components = entry.get("components", {})
    dt = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)

    return {
        "timestamp_utc": dt.isoformat(),
        "hour": dt.hour,
        "day": dt.day,
        "month": dt.month,
        "year": dt.year,
        "day_of_week": dt.weekday(),
        "aqi_index": entry.get("main", {}).get("aqi"),
        "co": components.get("co"),
        "no": components.get("no"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "so2": components.get("so2"),
        "pm2_5": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "nh3": components.get("nh3"),
    }


def add_change_rate_features(rows):
    """Add hour-over-hour AQI and PM2.5 change rate, computed after sorting
    chronologically (backfilled data doesn't arrive in guaranteed order across
    chunks, so we sort first)."""
    rows.sort(key=lambda r: r["timestamp_utc"])

    prev = None
    for row in rows:
        if prev is None:
            row["aqi_change_rate"] = None
            row["pm25_change_rate"] = None
        else:
            curr_time = datetime.fromisoformat(row["timestamp_utc"])
            prev_time = datetime.fromisoformat(prev["timestamp_utc"])
            hours_elapsed = (curr_time - prev_time).total_seconds() / 3600.0

            if hours_elapsed > 0 and row["aqi_index"] is not None and prev["aqi_index"] is not None:
                row["aqi_change_rate"] = round((row["aqi_index"] - prev["aqi_index"]) / hours_elapsed, 4)
            else:
                row["aqi_change_rate"] = None

            if hours_elapsed > 0 and row["pm2_5"] is not None and prev["pm2_5"] is not None:
                row["pm25_change_rate"] = round((row["pm2_5"] - prev["pm2_5"]) / hours_elapsed, 4)
            else:
                row["pm25_change_rate"] = None

        prev = row

    return rows


def main():
    if not OPENWEATHER_API_KEY:
        raise RuntimeError(
            "OPENWEATHER_API_KEY environment variable not set. "
            "Set it the same way you did for fetch_features.py."
        )

    overall_start = max(month_delta(datetime.now(timezone.utc), BACKFILL_MONTHS), EARLIEST_AVAILABLE)
    overall_end = datetime.now(timezone.utc)

    print(f"Backfilling from {overall_start.date()} to {overall_end.date()} "
          f"({BACKFILL_MONTHS} months requested)...")

    all_rows = []
    chunk_start = overall_start

    while chunk_start < overall_end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), overall_end)

        print(f"  Fetching {chunk_start.date()} -> {chunk_end.date()}...")
        try:
            records = fetch_history_chunk(chunk_start, chunk_end)
            print(f"    Got {len(records)} hourly records")
            all_rows.extend(record_to_row(r) for r in records)
        except requests.exceptions.HTTPError as e:
            print(f"    HTTP error on this chunk, skipping: {e}")
        except Exception as e:
            print(f"    Failed on this chunk, skipping: {e}")

        chunk_start = chunk_end
        time.sleep(1)  # small courtesy delay between requests

    if not all_rows:
        print("\nNo data retrieved. Check your API key and date range.")
        return

    print(f"\nTotal OpenWeather records fetched: {len(all_rows)}")

    print("\nFetching Open-Meteo (secondary source, cross-check)...")
    om_data = fetch_all_open_meteo(overall_start, overall_end)
    all_rows = merge_open_meteo(all_rows, om_data)

    print("\nComputing change-rate features...")
    all_rows = add_change_rate_features(all_rows)

    print(f"Writing to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    # Flag rows where the two sources disagree a lot on PM2.5, for visibility
    # in your report - mirrors the staleness-flag approach from the live script.
    DISAGREEMENT_THRESHOLD = 25  # µg/m³, adjust based on what you see in practice
    flagged = [r for r in all_rows if r.get("pm25_source_diff") is not None
               and r["pm25_source_diff"] > DISAGREEMENT_THRESHOLD]

    print(f"\nDone. {len(all_rows)} rows saved to {OUTPUT_CSV}")
    print(f"Rows with PM2.5 disagreement > {DISAGREEMENT_THRESHOLD} between "
          f"OpenWeather and Open-Meteo: {len(flagged)} ({100*len(flagged)/len(all_rows):.1f}%)")
    print("This is your training dataset for the Training Pipeline step.")


if __name__ == "__main__":
    main()