"""
eda_recheck.py

Quick re-check of the AQI signal after compute_true_aqi.py replaces the
coarse OpenWeather 1-5 category with a real continuous AQI. Run this
before we rebuild train_model.py -- same discipline as the original
eda.py run, so we're not building the multi-horizon training pipeline on
a target we haven't validated.

Usage: python3 eda_recheck.py
Reads aqi_historical_backfill_v2.csv (output of compute_true_aqi.py).
"""

import pandas as pd

INPUT_CSV = "aqi_historical_backfill_v2.csv"
TIMESTAMP_COL = "timestamp_utc"
TARGET_COL = "aqi"


def main():
    df = pd.read_csv(INPUT_CSV, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    print(f"Rows: {len(df)}  |  Span: {df[TIMESTAMP_COL].min()} -> {df[TIMESTAMP_COL].max()}")

    print("\n--- AQI variance check (new continuous target) ---")
    print(f"AQI range: {df[TARGET_COL].min():.1f} - {df[TARGET_COL].max():.1f}")
    print(f"AQI mean: {df[TARGET_COL].mean():.1f}  std dev: {df[TARGET_COL].std():.2f}")
    diffs = df[TARGET_COL].diff().abs()
    print(f"Mean absolute hour-to-hour change: {diffs.mean():.2f}")
    print(f"% of hours with < 0.5 AQI change: {(diffs < 0.5).mean() * 100:.1f}%")

    print("\n--- Autocorrelation at forecast horizons ---")
    for h in [1, 24, 48, 72]:
        corr = df[TARGET_COL].corr(df[TARGET_COL].shift(-h))
        print(f"  Correlation(AQI now, AQI +{h}h): {corr:.4f}")

    if "aqi_category" in df.columns:
        print("\n--- AQI category distribution ---")
        print(df["aqi_category"].value_counts())

    if "aqi_dominant_pollutant" in df.columns:
        print("\n--- Dominant pollutant distribution ---")
        print(df["aqi_dominant_pollutant"].value_counts())


if __name__ == "__main__":
    main()