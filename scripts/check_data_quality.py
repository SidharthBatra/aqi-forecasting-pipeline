"""
check_data_quality.py

Quick sanity check on the backfilled dataset before moving to training:
  - How much does AQI/PM2.5 actually vary hour to hour? (catches an
    over-smooth model output that would give the model no signal to learn)
  - Are there long runs of identical consecutive values? (would indicate
    stale/interpolated data slipping through)
  - Basic missing-value summary.

Run this after backfill_historical.py.
"""

import csv
import statistics

CSV_PATH = "aqi_historical_backfill.csv"


def main():
    with open(CSV_PATH, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"Total rows: {len(rows)}\n")

    # --- Missing value summary ---
    print("Missing value counts (key columns):")
    for col in ["aqi_index", "pm2_5", "om_pm2_5", "aqi_change_rate"]:
        missing = sum(1 for r in rows if not r.get(col))
        print(f"  {col}: {missing} missing ({100*missing/len(rows):.1f}%)")

    # --- Variance check ---
    pm25_values = [float(r["pm2_5"]) for r in rows if r.get("pm2_5")]
    aqi_values = [float(r["aqi_index"]) for r in rows if r.get("aqi_index")]

    print(f"\nPM2.5 stats: min={min(pm25_values):.2f}, max={max(pm25_values):.2f}, "
          f"mean={statistics.mean(pm25_values):.2f}, stdev={statistics.stdev(pm25_values):.2f}")
    print(f"AQI index stats: min={min(aqi_values):.0f}, max={max(aqi_values):.0f}, "
          f"mean={statistics.mean(aqi_values):.2f}, stdev={statistics.stdev(aqi_values):.2f}")

    # --- Consecutive identical value runs (flat-line detection) ---
    max_run = 1
    current_run = 1
    for i in range(1, len(pm25_values)):
        if pm25_values[i] == pm25_values[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    print(f"\nLongest run of identical consecutive PM2.5 values: {max_run} hours")
    if max_run > 12:
        print("  -> Worth investigating: 12+ identical hours in a row could mean "
              "stale/interpolated data in that stretch.")
    else:
        print("  -> Looks fine, no suspiciously long flat stretches.")


if __name__ == "__main__":
    main()