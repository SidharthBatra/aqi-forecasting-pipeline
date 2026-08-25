"""
eda.py

Exploratory Data Analysis: fetches features from Hopsworks and checks
whether the data actually supports 24h/48h/72h AQI forecasting before
we rebuild the training pipeline.

Answers:
  - How much does AQI vary hour to hour? (is there signal to forecast at all)
  - Autocorrelation at 1h, 24h, 48h, 72h lags (do past values predict future ones)
  - Missing data / gaps that would break lagged feature construction
  - Seasonal/daily patterns worth encoding as features later

Run inside your Codespace:
    pip install hopsworks pandas matplotlib
    python eda.py

Requires HOPSWORKS_API_KEY set as an environment variable / Codespaces secret.
Saves plots to eda_output/ as PNG files.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt

FEATURE_GROUP_NAME = "aqi_karachi_features"
FEATURE_GROUP_VERSION = 1
TARGET_COLUMN = "aqi_index"

OUTPUT_DIR = "eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_data():
    import hopsworks

    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        print("ERROR: HOPSWORKS_API_KEY environment variable not set.")
        sys.exit(1)

    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    df = fg.read()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    print(f"Fetched {len(df)} rows spanning {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")
    return df


def check_gaps(df):
    """Check for missing hours - gaps break lagged feature construction."""
    print("\n--- Checking for time gaps ---")
    time_diffs = df["timestamp_utc"].diff().dt.total_seconds() / 3600.0
    expected_hourly = (time_diffs == 1.0).sum()
    gaps = time_diffs[time_diffs > 1.0]

    print(f"Rows with exactly 1-hour spacing: {expected_hourly} / {len(df) - 1}")
    print(f"Number of gaps (>1 hour jump): {len(gaps)}")
    if len(gaps) > 0:
        print(f"Largest gap: {gaps.max():.1f} hours")
        print(f"Total missing hours (approx): {(gaps - 1).sum():.0f}")


def check_variance(df):
    """Is there enough hour-to-hour movement to forecast, or is it mostly flat?"""
    print("\n--- AQI variance check ---")
    print(f"AQI index range: {df[TARGET_COLUMN].min()} - {df[TARGET_COLUMN].max()}")
    print(f"AQI index std dev: {df[TARGET_COLUMN].std():.3f}")

    hour_to_hour_change = df[TARGET_COLUMN].diff().abs()
    print(f"Mean absolute hour-to-hour change: {hour_to_hour_change.mean():.3f}")
    print(f"% of hours with zero change: {(hour_to_hour_change == 0).mean() * 100:.1f}%")


def compute_autocorrelation(df):
    """Correlation between AQI now and AQI N hours in the future.
    High correlation at a given lag = past values are informative for
    forecasting that far ahead. Low/near-zero = that horizon may be hard
    to predict from history alone."""
    print("\n--- Autocorrelation at forecast horizons ---")
    lags_to_check = {"1h": 1, "24h": 24, "48h": 48, "72h": 72}
    results = {}

    for label, lag in lags_to_check.items():
        shifted = df[TARGET_COLUMN].shift(-lag)
        valid = df[TARGET_COLUMN].notna() & shifted.notna()
        corr = df.loc[valid, TARGET_COLUMN].corr(shifted[valid])
        results[label] = corr
        print(f"  Correlation(AQI now, AQI +{label}): {corr:.4f}")

    return results


def plot_time_series(df):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["timestamp_utc"], df[TARGET_COLUMN], linewidth=0.5)
    ax.set_title("AQI Index Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI Index")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "aqi_time_series.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved: {path}")


def plot_autocorrelation_function(df):
    from pandas.plotting import autocorrelation_plot
    fig, ax = plt.subplots(figsize=(10, 4))
    autocorrelation_plot(df[TARGET_COLUMN].dropna(), ax=ax)
    ax.set_xlim(0, 168)  # one week of hourly lags
    ax.set_title("Autocorrelation Function (up to 1 week of lags)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "autocorrelation.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved: {path}")


def plot_daily_seasonality(df):
    df = df.copy()
    df["hour"] = df["timestamp_utc"].dt.hour
    hourly_avg = df.groupby("hour")[TARGET_COLUMN].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourly_avg.index, hourly_avg.values, marker="o")
    ax.set_title("Average AQI by Hour of Day")
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Average AQI Index")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "daily_seasonality.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved: {path}")


def verdict(df, autocorr_results):
    """Plain-language readout on whether forecasting looks viable."""
    print("\n" + "=" * 50)
    print("VERDICT")
    print("=" * 50)

    n_rows = len(df)
    span_days = (df["timestamp_utc"].max() - df["timestamp_utc"].min()).days

    print(f"Data span: {span_days} days ({n_rows} rows)")

    for label, corr in autocorr_results.items():
        if corr > 0.5:
            quality = "strong signal - good"
        elif corr > 0.2:
            quality = "moderate signal - workable"
        else:
            quality = "weak signal - this horizon will be hard to forecast accurately"
        print(f"  {label} horizon: correlation={corr:.3f} -> {quality}")

    print("\nRecommendation: proceed with lagged-feature rebuild. Even weak-signal")
    print("horizons are worth modeling - the metrics (RMSE/MAE/R2) will honestly")
    print("reflect difficulty rather than the false near-perfect scores we saw")
    print("from the leaky same-timestamp features.")


def main():
    df = fetch_data()
    check_gaps(df)
    check_variance(df)
    autocorr_results = compute_autocorrelation(df)

    print("\n--- Generating plots ---")
    plot_time_series(df)
    plot_autocorrelation_function(df)
    plot_daily_seasonality(df)

    verdict(df, autocorr_results)

    print(f"\nAll plots saved to ./{OUTPUT_DIR}/ - view them in the file explorer.")


if __name__ == "__main__":
    main()