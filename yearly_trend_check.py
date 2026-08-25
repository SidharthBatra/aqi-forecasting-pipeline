"""
yearly_trend_check.py

Quick, no-retraining check of whether the extended backfill (now starting
Nov 2020) includes a genuine regime shift -- specifically, whether the
COVID lockdown period (2020-2021, when Karachi traffic/industrial activity
was unusually low) looks meaningfully different from later years. If it
does, training on it could be actively hurting the model's calibration of
"normal" AQI levels, which would show up exactly the way it did in
train_model.py: 48h/72h models underperforming even a naive persistence
baseline, despite having far more total training data than before.

Run this before deciding whether to exclude/downweight the earliest
years -- it reads the already-computed aqi_historical_backfill_v2.csv, no
API calls or retraining needed.
"""

import pandas as pd

INPUT_CSV = "aqi_historical_backfill_v2.csv"
TIMESTAMP_COL = "timestamp_utc"
TARGET_COL = "aqi"


def main():
    df = pd.read_csv(INPUT_CSV, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df["year"] = df[TIMESTAMP_COL].dt.year

    print("=" * 70)
    print("AQI BY YEAR")
    print("=" * 70)
    yearly = df.groupby("year")[TARGET_COL].agg(["mean", "median", "std", "count"])
    print(yearly)

    print("\n" + "=" * 70)
    print("AQI CATEGORY MIX BY YEAR (% of rows)")
    print("=" * 70)
    if "aqi_category" in df.columns:
        mix = pd.crosstab(df["year"], df["aqi_category"], normalize="index") * 100
        print(mix.round(1))

    print("\n" + "=" * 70)
    print("COVID-ERA (2020-2021) vs LATER (2022+) COMPARISON")
    print("=" * 70)
    covid_era = df[df["year"].isin([2020, 2021])][TARGET_COL]
    later = df[df["year"] >= 2022][TARGET_COL]
    print(f"2020-2021: n={len(covid_era)}  mean={covid_era.mean():.1f}  median={covid_era.median():.1f}  std={covid_era.std():.1f}")
    print(f"2022+:     n={len(later)}  mean={later.mean():.1f}  median={later.median():.1f}  std={later.std():.1f}")
    pct_diff = (later.mean() - covid_era.mean()) / covid_era.mean() * 100
    print(f"\nMean AQI is {pct_diff:+.1f}% {'higher' if pct_diff > 0 else 'lower'} in 2022+ vs 2020-2021.")
    print(
        "A large gap here (rule of thumb: >15-20%) supports excluding or "
        "downweighting the COVID-era rows from training -- they'd represent "
        "a different pollution regime than what the model needs to forecast "
        "going forward. A small gap means this probably isn't the main "
        "driver of the 48h/72h underperformance, and the RidgeCV alpha grid "
        "fix / feature pruning are more likely to matter more."
    )

    print("\n" + "=" * 70)
    print("TEST-WINDOW YEARS (most recent ~20% of the timeline, per")
    print("train_model.py's chronological split)")
    print("=" * 70)
    n = len(df)
    test_start_idx = int(n * 0.8)
    test_df = df.iloc[test_start_idx:]
    print(
        f"Test window: {test_df[TIMESTAMP_COL].min()} -> {test_df[TIMESTAMP_COL].max()}"
    )
    print(f"Test window mean AQI: {test_df[TARGET_COL].mean():.1f}  median: {test_df[TARGET_COL].median():.1f}")
    print(f"Full dataset mean AQI: {df[TARGET_COL].mean():.1f}  median: {df[TARGET_COL].median():.1f}")


if __name__ == "__main__":
    main()