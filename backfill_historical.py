"""
backfill_historical.py

Historical Data Backfill: pulls past hourly pollutant data from OpenWeather's
Air Pollution History API and builds a training-ready CSV in one shot,
instead of waiting months for live collection.

OpenWeather's free tier includes history back to 27 Nov 2020, hourly
granularity, in a single call per date range (not one call per hour).
Confirmed still current and free-tier-inclusive as of Aug 2026 (source:
OpenWeather's own Air Pollution API docs/blog).

Usage:
    python backfill_historical.py

Adjust BACKFILL_MONTHS below to control how far back to pull (6-24 typical).

REVISION HISTORY:
  Originally extended to BACKFILL_MONTHS=200 (pulling the full ~5.7yr
  history back to Nov 2020) after thorough_eda_v2.py showed a 24-month
  backfill only covers ~1.57 years under an 80/20 chronological split --
  close to a single seasonal cycle. That did give more seasonal cycles,
  but also pulled in the COVID-era (2020-2021) low-traffic period and,
  per trend_source_crosscheck.py, years where OpenWeather's historical
  reconstruction ran 33-53% high vs. an independent source.
  Reverted back to BACKFILL_MONTHS=24 (the mentor's stated "2yr ideal")
  by explicit choice -- scope control over maximizing seasonal coverage.
  Known, accepted tradeoff: 24 months under an 80/20 chronological split
  gives back roughly the ~1.57-year training window (~1 seasonal cycle)
  that was originally flagged as a likely contributor to the 48h/72h
  models' negative R2 -- this hasn't been re-solved, it's been
  deliberately reintroduced. Revisit train_model.py's per-horizon results
  with this in mind rather than assuming the sufficiency issue is gone.

Also fixed: pm25_source_diff (the OpenWeather vs Open-Meteo PM2.5
cross-check) was an ABSOLUTE difference, which the EDA showed correlates
0.97 with raw pm2_5 itself -- it was mostly just echoing the pollution
magnitude rather than signaling genuine source disagreement. Added
pm25_source_diff_pct alongside it (a relative/percentage difference,
independent of magnitude) for use as a model feature; the original
absolute pm25_source_diff is kept unchanged since the disagreement-report
section below is written in µg/m3 terms.
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

# Mentor's target: 6 months minimum, 2 years ideal. Set to the "ideal" 24
# by explicit choice -- see REVISION note above for the tradeoff being
# accepted here (fewer seasonal cycles than the 200-month version had).
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

# Floor used when computing a percentage PM2.5 disagreement, so a pair of
# near-zero readings (e.g. 0.4 vs 0.6 ug/m3) doesn't produce a misleadingly
# huge percentage from dividing by a tiny number.
PM25_PCT_DIFF_FLOOR = 5.0  # ug/m3


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
    timestamp, and compute cross-check diffs on PM2.5 (both absolute and
    percentage) so large disagreements between the two sources are visible
    rather than silently trusted."""
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
                abs_diff = abs(row["pm2_5"] - om_row["om_pm2_5"])
                row["pm25_source_diff"] = round(abs_diff, 3)
                denom = max(row["pm2_5"], om_row["om_pm2_5"], PM25_PCT_DIFF_FLOOR)
                row["pm25_source_diff_pct"] = round(abs_diff / denom * 100, 2)
            else:
                row["pm25_source_diff"] = None
                row["pm25_source_diff_pct"] = None
        else:
            for field in ("om_pm2_5", "om_pm10", "om_co", "om_no2", "om_so2", "om_o3", "om_us_aqi"):
                row[field] = None
            row["pm25_source_diff"] = None
            row["pm25_source_diff_pct"] = None

    print(f"  Matched Open-Meteo data for {matched}/{len(rows)} rows")
    return rows


def clean_concentration(value):
    """Pollutant concentrations can never legitimately be negative. Some
    historical hours from OpenWeather's history endpoint return a -9999
    "no data" sentinel instead of a real reading or a null (confirmed
    directly: an earlier backfill run had pm10/no2/o3 columns with
    actual_range starting at -9999.0). Convert any negative reading to
    None here, at parse time, so it's never written to the CSV as if it
    were real data."""
    if value is not None and value < 0:
        return None
    return value


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
        "co": clean_concentration(components.get("co")),
        "no": clean_concentration(components.get("no")),
        "no2": clean_concentration(components.get("no2")),
        "o3": clean_concentration(components.get("o3")),
        "so2": clean_concentration(components.get("so2")),
        "pm2_5": clean_concentration(components.get("pm2_5")),
        "pm10": clean_concentration(components.get("pm10")),
        "nh3": clean_concentration(components.get("nh3")),
    }


def dedupe_by_timestamp(rows):
    """The OpenWeather History API treats the start/end of each chunk
    request as inclusive on both ends, so consecutive 30-day chunks each
    re-fetch their shared boundary hour (confirmed directly: full chunks
    return 721 hourly records for a 30-day window instead of 720). That
    produces a duplicate row for that hour. Left in, duplicate-timestamp
    rows sit adjacent after sorting with zero elapsed time between them,
    which breaks any time-based diff (e.g. change-rate features) computed
    downstream. Drop them here, keeping the first occurrence."""
    seen = set()
    deduped = []
    n_dupes = 0
    for row in rows:
        ts = row["timestamp_utc"]
        if ts in seen:
            n_dupes += 1
            continue
        seen.add(ts)
        deduped.append(row)
    if n_dupes:
        print(f"  Removed {n_dupes} duplicate-timestamp rows (chunk-boundary overlap).")
    return deduped


def add_change_rate_features(rows):
    """Add hour-over-hour AQI and PM2.5 change rate, computed after sorting
    chronologically (backfilled data doesn't arrive in guaranteed order across
    chunks, so we sort first).

    Note: aqi_change_rate here is necessarily based on the raw OpenWeather
    1-5 aqi_index, since the real continuous AQI doesn't exist until
    compute_true_aqi.py runs downstream of this script. compute_true_aqi.py
    overwrites this column with a properly recomputed version based on the
    real AQI once it's available -- don't rely on this raw version as a
    model feature.
    """
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

    total_days = (overall_end - overall_start).days
    n_ow_chunks = -(-total_days // CHUNK_DAYS)  # ceil division
    n_om_chunks = -(-total_days // OPEN_METEO_CHUNK_DAYS)
    print(
        f"Backfilling from {overall_start.date()} to {overall_end.date()} "
        f"({total_days} days, {total_days / 365.25:.2f} years)..."
    )
    print(
        f"Expecting ~{n_ow_chunks} OpenWeather chunks and ~{n_om_chunks} "
        f"Open-Meteo chunks. This will take a while longer than the "
        f"previous 24-month run -- each chunk prints progress as it goes."
    )

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

    print("\nDeduplicating chunk-boundary overlaps...")
    all_rows = dedupe_by_timestamp(all_rows)
    print(f"  {len(all_rows)} unique hourly rows remain")

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