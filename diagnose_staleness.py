"""
Diagnostic: why does the dashboard show an old timestamp when the raw feature
group is being updated hourly?

Run this from the repo root in your Codespace:

    python diagnose_staleness.py

It needs HOPSWORKS_API_KEY in the environment (already set as a Codespaces
secret). It only READS from Hopsworks -- it writes nothing and changes nothing.

The hypothesis being tested: there is a gap in the raw feature group between
where the 2-year backfill ended and where hourly ingestion actually began.
Because features are built on a complete hourly grid, gap hours become NaN
placeholders, which makes the long rolling windows (aqi_rollmean_168h etc.)
NaN for every recent row whose 7-day lookback touches the gap. Rows with an
incomplete feature vector can't be fed to the models, so the dashboard falls
back to the newest fully-populated row -- the last row before the gap.
"""

import os
import sys

import pandas as pd

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 200)

try:
    from train_model import (
        TIMESTAMP_COL,
        load_and_prepare_grid,
        add_time_features,
        add_lag_and_rolling_features,
        get_feature_columns,
    )
except ImportError as e:
    sys.exit(
        f"Couldn't import from train_model.py: {e}\n"
        "Run this from the repo root (the directory containing train_model.py)."
    )


def main():
    if not os.environ.get("HOPSWORKS_API_KEY"):
        sys.exit("HOPSWORKS_API_KEY is not set in this environment.")

    print("=" * 78)
    print("STEP 1: read the raw feature group (Hopsworks first, per read_source_data)")
    print("=" * 78)

    df = load_and_prepare_grid()

    # load_and_prepare_grid may return the timestamp as an index or a column;
    # normalize to a plain column so the checks below work either way.
    if TIMESTAMP_COL not in df.columns:
        df = df.reset_index()
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    print(f"\nrows: {len(df)}")
    print(f"earliest: {df[TIMESTAMP_COL].min()}")
    print(f"latest:   {df[TIMESTAMP_COL].max()}")
    print(f"now(UTC): {pd.Timestamp.utcnow()}")

    print("\n--- last 40 timestamps, with whether 'aqi' is present ---")
    tail = df[[TIMESTAMP_COL, "aqi"]].tail(40).copy()
    tail["aqi_is_nan"] = tail["aqi"].isna()
    print(tail.to_string(index=False))

    print("\n" + "=" * 78)
    print("STEP 2: find gaps in the hourly sequence")
    print("=" * 78)

    gaps = df[TIMESTAMP_COL].diff()
    big = gaps[gaps > pd.Timedelta("1h")]
    if big.empty:
        print("\nNo gaps > 1h. (This would weaken the gap hypothesis.)")
    else:
        print(f"\n{len(big)} gap(s) larger than 1 hour:")
        for idx, delta in big.items():
            print(
                f"  gap of {delta} ending at {df.loc[idx, TIMESTAMP_COL]} "
                f"(previous row: {df.loc[idx - 1, TIMESTAMP_COL]})"
            )

    # Even on a complete grid, rows can exist with a NaN aqi (placeholders).
    nan_aqi = df["aqi"].isna().sum()
    print(f"\nrows with NaN 'aqi' anywhere in the frame: {nan_aqi}")
    if nan_aqi:
        nan_span = df.loc[df["aqi"].isna(), TIMESTAMP_COL]
        print(f"  spanning {nan_span.min()} .. {nan_span.max()}")

    print("\n" + "=" * 78)
    print("STEP 3: engineer features and see which are NaN for the newest rows")
    print("=" * 78)

    eng = df.set_index(TIMESTAMP_COL)
    eng = add_time_features(eng)
    eng = add_lag_and_rolling_features(eng)

    try:
        feat_cols = get_feature_columns(eng)
    except TypeError:
        feat_cols = get_feature_columns()

    feat_cols = [c for c in feat_cols if c in eng.columns]
    print(f"\n{len(feat_cols)} feature columns checked.")

    recent = eng[feat_cols].tail(20)
    nan_counts = recent.isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

    if nan_counts.empty:
        print(
            "\nNo NaNs in the feature columns of the last 20 rows.\n"
            "=> The gap hypothesis is WRONG. The newest rows are fully usable,\n"
            "   so the staleness is coming from somewhere else in dashboard.py\n"
            "   (row selection, sorting, or the prediction path)."
        )
    else:
        print("\nFeatures with NaNs in the last 20 rows (count out of 20):")
        print(nan_counts.to_string())
        print(
            "\n=> The gap hypothesis is SUPPORTED. Rows this recent have an\n"
            "   incomplete feature vector, so they can't be scored, and the\n"
            "   dashboard falls back to an older fully-populated row."
        )

    print("\n--- newest row that has NO NaN in any feature column ---")
    complete = eng[feat_cols].notna().all(axis=1)
    if complete.any():
        newest_complete = complete[complete].index.max()
        print(f"  {newest_complete}")
        print(
            "  ^ If this matches the stale date shown on the dashboard, that\n"
            "    confirms the fallback behaviour end to end."
        )
    else:
        print("  none -- no row has a complete feature vector at all.")


if __name__ == "__main__":
    main()