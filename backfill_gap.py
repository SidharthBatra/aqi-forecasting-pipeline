"""
backfill_gap.py

One-off targeted gap-fill for aqi_karachi_features v4. Does NOT re-run the
full 2-year backfill -- it fetches only the missing window between the
newest row with a real (non-NaN) 'aqi' and now, and inserts those rows
into the EXISTING feature group.

Why this exists: fetch_features.py originally wrote 'timestamp_utc' from
OpenWeather's live 'dt' field without flooring to the hour, so every live
row it inserted before that fix landed off the exact-hourly grid
train_model.load_and_prepare_grid() reindexes onto, and was silently
dropped. That left ~8 days of missing hours between the 2-year backfill's
last row (2026-08-24 20:00) and the present. The flooring fix (see
fetch_features.py) stops the bleeding going forward but does not fill the
hole that already exists -- that's what this script does, once.

Reuses, rather than reimplements:
  - backfill_historical.py's fetch_history_chunk() / record_to_row() /
    dedupe_by_timestamp() for pulling the gap window from OpenWeather's
    History API, identically to how the original 2-year backfill was
    built.
  - compute_true_aqi.py's compute_sub_indices() / categorize() /
    clean_sentinel_values() for computing the EPA rolling-window 'aqi',
    identically to both the backfill and fetch_features.py's live path.

Requires HOPSWORKS_API_KEY and OPENWEATHER_API_KEY in the environment,
same as every other script in this repo.

Usage:
    python backfill_gap.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from backfill_historical import (
    CHUNK_DAYS,
    dedupe_by_timestamp,
    fetch_history_chunk,
    record_to_row,
)
from compute_true_aqi import (
    POLLUTANT_COLUMNS,
    categorize,
    clean_sentinel_values,
    compute_sub_indices,
)
from fetch_features import CONTEXT_HOURS, RAW_FEATURE_GROUP_NAME, RAW_FEATURE_GROUP_VERSION

TIMESTAMP_COL = "timestamp_utc"

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")


def fetch_gap_window(gap_start, gap_end):
    """Pulls [gap_start, gap_end] from OpenWeather's History API, chunked
    the same way backfill_historical.py chunks the full backfill (a ~8-day
    gap fits in one chunk today, but this keeps the script correct if the
    gap is ever larger). Returns a list of flat row dicts, deduped at
    chunk boundaries exactly like the original backfill."""
    all_rows = []
    chunk_start = gap_start
    while chunk_start <= gap_end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), gap_end)
        print(f"  Fetching {chunk_start.isoformat()} -> {chunk_end.isoformat()}...")
        records = fetch_history_chunk(chunk_start, chunk_end)
        print(f"    Got {len(records)} hourly records")
        all_rows.extend(record_to_row(r) for r in records)
        chunk_start = chunk_end + timedelta(hours=1)

    all_rows = dedupe_by_timestamp(all_rows)
    return all_rows


def main():
    if not HOPSWORKS_API_KEY:
        print("ERROR: HOPSWORKS_API_KEY not set.")
        sys.exit(1)
    if not OPENWEATHER_API_KEY:
        print("ERROR: OPENWEATHER_API_KEY not set.")
        sys.exit(1)

    import hopsworks

    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=RAW_FEATURE_GROUP_NAME, version=RAW_FEATURE_GROUP_VERSION)

    print(f"Reading {RAW_FEATURE_GROUP_NAME} v{RAW_FEATURE_GROUP_VERSION}...")
    df = fg.read()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    print(f"  {len(df)} rows read.")

    # --- 1. Determine the gap empirically ---
    # A hour-misaligned row (a pre-flooring-fix live row, see
    # fetch_features.py) can carry a non-null 'aqi' yet still not occupy any
    # slot on the hourly grid load_and_prepare_grid() reindexes onto -- it
    # must NOT count as "coverage" here, or the gap start gets computed past
    # a timestamp that was never actually on the grid, and a real multi-day
    # hole reads as "already current".
    hour_aligned = (df[TIMESTAMP_COL].dt.minute == 0) & (df[TIMESTAMP_COL].dt.second == 0)
    valid = df[df["aqi"].notna() & hour_aligned]
    if valid.empty:
        print("ERROR: no hour-aligned row in the feature group has a non-null 'aqi' -- "
              "can't determine a gap start. Run backfill_historical.py first.")
        sys.exit(1)
    newest_valid_ts = valid[TIMESTAMP_COL].max()
    gap_start = newest_valid_ts + timedelta(hours=1)
    now_floored = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    print(
        f"\nNewest row with a real 'aqi': {newest_valid_ts.isoformat()}\n"
        f"Resolved gap window to fetch: {gap_start.isoformat()} -> {now_floored.isoformat()}"
    )
    if gap_start > now_floored:
        print("Nothing to backfill -- already current.")
        return

    # --- 2. Fetch the gap window from OpenWeather (reusing backfill_historical.py) ---
    print("\nFetching gap window from OpenWeather History API...")
    gap_rows = fetch_gap_window(gap_start, now_floored)
    if not gap_rows:
        print("No records returned for the gap window -- nothing to insert.")
        return

    gap_df = pd.DataFrame(gap_rows)
    gap_df[TIMESTAMP_COL] = pd.to_datetime(gap_df[TIMESTAMP_COL], utc=True)
    # record_to_row()'s dt comes straight from OpenWeather's History API,
    # which the original 2-year backfill confirmed lands exactly on :00 --
    # floor anyway (matches the fetch_features.py flooring fix) so a
    # single misbehaving record can't silently reintroduce the alignment
    # bug this whole gap exists because of.
    gap_df[TIMESTAMP_COL] = gap_df[TIMESTAMP_COL].dt.floor("h")
    gap_df = gap_df[(gap_df[TIMESTAMP_COL] >= gap_start) & (gap_df[TIMESTAMP_COL] <= now_floored)]
    gap_df = gap_df.drop_duplicates(subset=[TIMESTAMP_COL], keep="last").sort_values(TIMESTAMP_COL)
    if gap_df.empty:
        print("No in-window records after filtering -- nothing to insert.")
        return
    print(f"  {len(gap_df)} gap-window rows to insert, "
          f"{gap_df[TIMESTAMP_COL].min().isoformat()} -> {gap_df[TIMESTAMP_COL].max().isoformat()}")

    # --- 5. Compute rolling AQI WITH trailing context from Hopsworks ---
    # The EPA aqi for the first ~24h of gap rows depends on the 24h
    # PRECEDING them, which live in the existing backfill, not in what was
    # just fetched. Pull that trailing context straight from the feature
    # group (already-cleaned, already-correct data) rather than re-hitting
    # OpenWeather for it.
    context_cutoff = newest_valid_ts - timedelta(hours=CONTEXT_HOURS - 1)
    context_cols = [TIMESTAMP_COL] + list(POLLUTANT_COLUMNS.values())
    context_df = df[(df[TIMESTAMP_COL] >= context_cutoff) & (df[TIMESTAMP_COL] <= newest_valid_ts)][context_cols].copy()
    print(f"\nUsing {len(context_df)} rows of trailing context "
          f"({context_cutoff.isoformat()} -> {newest_valid_ts.isoformat()}) "
          f"for the rolling-window aqi calc.")

    combined = pd.concat([context_df, gap_df[context_cols]], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP_COL).drop_duplicates(subset=[TIMESTAMP_COL], keep="last")
    combined = clean_sentinel_values(combined)
    combined_indexed = combined.set_index(TIMESTAMP_COL)

    print("Computing EPA sub-indices over the concatenated (context + gap) series...")
    sub_indices = compute_sub_indices(combined_indexed)
    aqi = sub_indices.max(axis=1)
    dominant = sub_indices.idxmax(axis=1)
    category = aqi.apply(categorize)

    # Change-rate features, computed the same way compute_true_aqi.py
    # recomputes them (against the real 'aqi', normalized per elapsed
    # hour) -- the combined series' ordering means the first gap row's
    # rate is measured against the last context row, not against nothing.
    hours_elapsed = combined_indexed.index.to_series().diff().dt.total_seconds() / 3600.0
    aqi_rate = (aqi.diff() / hours_elapsed).round(4)
    pm25_rate = (combined_indexed["pm2_5"].diff() / hours_elapsed).round(4)

    aqi_lookup = pd.DataFrame({
        "aqi": aqi,
        "aqi_dominant_pollutant": dominant,
        "aqi_category": category,
        "aqi_change_rate": aqi_rate,
        "pm25_change_rate": pm25_rate,
    })

    gap_df = gap_df.set_index(TIMESTAMP_COL).join(aqi_lookup, how="left").reset_index()

    # --- Build the final rows matching aqi_karachi_features' schema
    # (same columns/None-fills fetch_features.py's build_row() writes) ---
    gap_df = gap_df.rename(columns={"aqi_index": "aqi_index_openweather_1to5"})
    for col in ("om_pm2_5", "om_pm10", "om_co", "om_no2", "om_so2", "om_o3", "om_us_aqi",
                "pm25_source_diff", "pm25_source_diff_pct"):
        gap_df[col] = None
    # row_id derived from the (floored) epoch timestamp, identically to
    # fetch_features.py -- consistent with live rows and idempotent on
    # re-run (re-running this script for an already-filled hour upserts
    # the same row_id rather than duplicating it).
    gap_df["row_id"] = gap_df[TIMESTAMP_COL].apply(lambda ts: int(ts.timestamp()))

    # --- 3. Assert hour-alignment before insert ---
    misaligned = gap_df[TIMESTAMP_COL].dt.minute.ne(0) | gap_df[TIMESTAMP_COL].dt.second.ne(0)
    assert not misaligned.any(), (
        f"{int(misaligned.sum())} gap row(s) are not hour-aligned -- refusing to insert. "
        f"First offender: {gap_df.loc[misaligned, TIMESTAMP_COL].iloc[0]}"
    )

    float_cols = [
        "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
        "aqi_index_openweather_1to5", "aqi",
        "aqi_change_rate", "pm25_change_rate",
        "om_pm2_5", "om_pm10", "om_co", "om_no2", "om_so2", "om_o3", "om_us_aqi",
        "pm25_source_diff", "pm25_source_diff_pct",
    ]
    for col in float_cols:
        gap_df[col] = gap_df[col].astype(float)

    n_aqi_nan = int(gap_df["aqi"].isna().sum())
    if n_aqi_nan:
        print(f"  NOTE: {n_aqi_nan} gap row(s) still have NaN 'aqi' after the rolling "
              f"calc (missing pollutant inputs from OpenWeather for that hour) -- "
              f"these insert as documented gaps, same as fetch_features.py would.")

    print(f"\nInserting {len(gap_df)} rows into {RAW_FEATURE_GROUP_NAME} v{RAW_FEATURE_GROUP_VERSION}...")
    fg.insert(gap_df, write_options={"wait_for_job": False})

    # --- 7. Summary ---
    print("\n" + "=" * 60)
    print("GAP-FILL SUMMARY")
    print("=" * 60)
    print(f"  Window fetched : {gap_start.isoformat()} -> {now_floored.isoformat()}")
    print(f"  Rows inserted  : {len(gap_df)}")
    print(f"  First timestamp: {gap_df[TIMESTAMP_COL].min().isoformat()}")
    print(f"  Last timestamp : {gap_df[TIMESTAMP_COL].max().isoformat()}")
    print(f"  Rows with NaN aqi (missing pollutant inputs): {n_aqi_nan}")


if __name__ == "__main__":
    main()
