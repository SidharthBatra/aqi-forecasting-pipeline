"""
thorough_eda_v2.py

Full data-quality and sufficiency pass on aqi_historical_backfill_v2.csv,
run BEFORE any further train_model.py iteration. Two things prompted
this: (1) aqi_change_rate looking mostly zero, which smells like a
leftover computed against the old 1-5 aqi_index rather than the new
continuous aqi, and (2) an unverified claim (mine) that the new 'aqi'
column is genuinely continuous rather than secretly still a handful of
discrete buckets.

Checks, in order:
  1. Shape / date range / dtypes
  2. Missing-value report for every column
  3. Zero-value / staleness check on aqi_change_rate and
     pm25_change_rate specifically, cross-referenced against how often
     the OLD 1-5 category actually changed -- this either confirms or
     kills the "stale feature" hypothesis with a number, not a guess
  4. Target ('aqi') distribution: unique value count, rounded value
     counts, skew, histogram -- directly answers "is this actually
     continuous or effectively N buckets"
  5. Cross-check new 'aqi' against the old 1-5 category to confirm the
     relationship makes sense (higher category should track higher
     continuous aqi) and that they aren't secretly redundant
  6. Outlier scan (IQR-based) per pollutant column
  7. Correlation matrix highlights: feature pairs correlated >0.95,
     which flags redundancy that can destabilize Ridge/RF (e.g.
     Open-Meteo vs OpenWeather pollutant pairs, overlapping rolling
     windows)
  8. Data sufficiency summary: span in days/years, how many full
     seasonal cycles are actually available to a chronological
     train/test split (not just total row count) -- a dataset can have
     plenty of ROWS and still be short on independent SEASONS, which is
     what the model actually needs to learn seasonal AQI patterns
"""

import numpy as np
import pandas as pd

INPUT_CSV = "aqi_historical_backfill_v2.csv"
TIMESTAMP_COL = "timestamp_utc"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    df = pd.read_csv(INPUT_CSV, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # ---------------------------------------------------------------
    section("1. SHAPE / DATE RANGE / DTYPES")
    span_days = (df[TIMESTAMP_COL].max() - df[TIMESTAMP_COL].min()).days
    print(f"Rows: {len(df)}   Columns: {len(df.columns)}")
    print(f"Span: {df[TIMESTAMP_COL].min()} -> {df[TIMESTAMP_COL].max()}  ({span_days} days)")
    print("\nColumn dtypes:")
    print(df.dtypes)

    # ---------------------------------------------------------------
    section("2. MISSING VALUES PER COLUMN")
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    report = report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)
    if report.empty:
        print("No missing values in any column.")
    else:
        print(report)

    # ---------------------------------------------------------------
    section("3. aqi_change_rate / pm25_change_rate STALENESS CHECK")
    for col in ["aqi_change_rate", "pm25_change_rate"]:
        if col not in df.columns:
            print(f"'{col}' not present in this CSV.")
            continue
        zero_pct = (df[col] == 0).mean() * 100
        print(f"\n{col}:")
        print(f"  % of rows exactly 0: {zero_pct:.1f}%")
        print(f"  describe(): \n{df[col].describe()}")

    if "aqi_index_openweather_1to5" in df.columns:
        old_unchanged_pct = (df["aqi_index_openweather_1to5"].diff() == 0).mean() * 100
        print(
            f"\nFor comparison: % of hours where the OLD 1-5 category did NOT "
            f"change: {old_unchanged_pct:.1f}%"
        )
        if "aqi_change_rate" in df.columns:
            zero_pct = (df["aqi_change_rate"] == 0).mean() * 100
            print(
                f"If aqi_change_rate's zero-rate ({zero_pct:.1f}%) is close to this "
                f"number, it was almost certainly computed against the OLD 1-5 "
                f"column and needs to be recomputed against the new continuous "
                f"'aqi', or dropped from the feature set."
            )

    # ---------------------------------------------------------------
    section("4. IS THE NEW 'aqi' TARGET ACTUALLY CONTINUOUS?")
    if "aqi" in df.columns:
        aqi = df["aqi"].dropna()
        print(f"Non-null aqi rows: {len(aqi)}")
        print(f"Unique values: {aqi.nunique()}  (out of {len(aqi)} rows)")
        print(f"Min={aqi.min():.2f}  Max={aqi.max():.2f}  Mean={aqi.mean():.2f}  Std={aqi.std():.2f}")
        print(f"Skew: {aqi.skew():.3f}")
        print("\nTop 10 most repeated aqi values (rounded to nearest int):")
        print(aqi.round(0).value_counts().head(10))
        print(
            "\nInterpretation: a genuinely continuous variable should have a "
            "unique-value count in the thousands (close to row count) and no "
            "single rounded value dominating more than a few percent of rows. "
            "If unique count is small (tens/hundreds) or one value dominates, "
            "this is still effectively discretized and regression metrics are "
            "the wrong framing."
        )
    else:
        print("'aqi' column not found.")

    # ---------------------------------------------------------------
    section("5. NEW 'aqi' vs OLD 1-5 CATEGORY CROSS-CHECK")
    if "aqi" in df.columns and "aqi_index_openweather_1to5" in df.columns:
        cross = df.groupby("aqi_index_openweather_1to5")["aqi"].agg(["mean", "std", "count"])
        print(cross)
        corr = df["aqi"].corr(df["aqi_index_openweather_1to5"])
        print(f"\nCorrelation(new aqi, old 1-5 category): {corr:.3f}")
        print(
            "Expect a clear monotonic step-up in mean aqi as the old category "
            "rises 1->5, with meaningful within-category spread (std > 0) -- "
            "that confirms the new target adds real resolution the old "
            "category didn't have, rather than just being a rescaled copy."
        )

    # ---------------------------------------------------------------
    section("6. OUTLIER SCAN (IQR METHOD) PER POLLUTANT COLUMN")
    pollutant_cols = [c for c in ["pm2_5", "pm10", "co", "no", "no2", "so2", "nh3", "o3"] if c in df.columns]
    for col in pollutant_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        print(
            f"{col:10s}  IQR=[{q1:.1f}, {q3:.1f}]  outlier_bounds=[{lo:.1f}, {hi:.1f}]  "
            f"outliers={n_out} ({n_out / len(df) * 100:.1f}%)  "
            f"actual_range=[{df[col].min():.1f}, {df[col].max():.1f}]"
        )

    # ---------------------------------------------------------------
    section("7. HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.95)")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )
    high_corr = pairs[pairs > 0.95]
    if high_corr.empty:
        print("No feature pairs with |r| > 0.95.")
    else:
        print(high_corr.to_string())
        print(
            "\nThese pairs are near-duplicate information for a linear model "
            "(Ridge) and add noise/variance for a tree model (RF) without "
            "adding signal. Worth pruning one from each pair before the next "
            "training run."
        )

    # ---------------------------------------------------------------
    section("8. DATA SUFFICIENCY SUMMARY")
    n_years = span_days / 365.25
    train_days = span_days * 0.8  # matches train_model.py's 80/20 chronological split
    train_years = train_days / 365.25
    print(f"Total span: {span_days} days ({n_years:.2f} years)")
    print(
        f"With an 80/20 chronological split, TRAINING data covers roughly the "
        f"first {train_days:.0f} days (~{train_years:.2f} years) and TEST covers "
        f"the remaining ~{span_days - train_days:.0f} days."
    )
    print(
        f"\n{train_years:.2f} years of training data means the model sees at "
        f"most {int(train_years) + 1} partial/full pass(es) through the annual "
        f"seasonal cycle before being evaluated on a season-position it has "
        f"seen very few (often just one, sometimes zero) prior examples of. "
        f"That alone -- independent of any feature bug -- limits how well any "
        f"model can learn 'what August/December AQI looks like' from a single "
        f"prior August/December. This is a real, structural sufficiency "
        f"limit, not something fixable by better hyperparameters."
    )
    n_gaps = (df[TIMESTAMP_COL].diff().dt.total_seconds() / 3600 > 1).sum()
    print(f"\nTime gaps (>1h jump): {n_gaps}")


if __name__ == "__main__":
    main()