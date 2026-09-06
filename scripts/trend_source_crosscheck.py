"""
trend_source_crosscheck.py

yearly_trend_check.py found a large, roughly monotonic decline in mean AQI
across 2020-2026 (252.6 -> 82.9), not just a COVID-era step change. Before
deciding how to handle this in training, this script checks whether the
decline is REAL (both independent data sources agree) or a DATA ARTIFACT
specific to OpenWeather's historical model (e.g. sparser input data for
older years in a region with historically thin ground-station coverage).

Method: Open-Meteo's om_us_aqi is available (CAMS reanalysis, independent
of OpenWeather) for the ~40% of rows where the cross-check matched. Compare
its year-over-year trend against our OpenWeather-derived 'aqi' for the SAME
matched rows -- if they move together, the decline is likely real; if
Open-Meteo is much flatter, OpenWeather's historical data likely drifted
for reasons unrelated to actual air quality.

No API calls -- reads the existing aqi_historical_backfill_v2.csv.
"""

import pandas as pd

INPUT_CSV = "aqi_historical_backfill_v2.csv"
TIMESTAMP_COL = "timestamp_utc"


def main():
    df = pd.read_csv(INPUT_CSV, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    df["year"] = df[TIMESTAMP_COL].dt.year

    matched = df[df["om_us_aqi"].notna()].copy()
    print(f"Rows with both sources present: {len(matched)} / {len(df)} ({len(matched)/len(df)*100:.1f}%)")

    print("\n" + "=" * 78)
    print("YEAR-OVER-YEAR MEAN AQI: OPENWEATHER-DERIVED vs OPEN-METEO (same rows)")
    print("=" * 78)
    yearly = matched.groupby("year").agg(
        n=("aqi", "size"),
        openweather_aqi_mean=("aqi", "mean"),
        open_meteo_us_aqi_mean=("om_us_aqi", "mean"),
    )
    yearly["diff"] = yearly["openweather_aqi_mean"] - yearly["open_meteo_us_aqi_mean"]
    yearly["pct_diff"] = yearly["diff"] / yearly["open_meteo_us_aqi_mean"] * 100
    print(yearly.round(1))

    print("\n" + "=" * 78)
    print("DOES OPEN-METEO SHOW THE SAME DECLINE?")
    print("=" * 78)
    years_present = sorted(yearly.index)
    if len(years_present) >= 2:
        first_year, last_year = years_present[0], years_present[-1]
        ow_change = (
            (yearly.loc[last_year, "openweather_aqi_mean"] - yearly.loc[first_year, "openweather_aqi_mean"])
            / yearly.loc[first_year, "openweather_aqi_mean"] * 100
        )
        om_change = (
            (yearly.loc[last_year, "open_meteo_us_aqi_mean"] - yearly.loc[first_year, "open_meteo_us_aqi_mean"])
            / yearly.loc[first_year, "open_meteo_us_aqi_mean"] * 100
        )
        print(f"OpenWeather-derived 'aqi':  {first_year} -> {last_year}: {ow_change:+.1f}%")
        print(f"Open-Meteo 'om_us_aqi':     {first_year} -> {last_year}: {om_change:+.1f}%")
        print(
            "\nIf these two percentages are similar in size and direction, the "
            "decline is corroborated by an independent source -- likely a real "
            "trend (or at least a trend both sources agree on), and training "
            "should account for it (e.g. recency-weighting samples) rather "
            "than being treated as noise to fix with hyperparameters.\n"
            "If Open-Meteo's change is much smaller (or the opposite sign), "
            "OpenWeather's historical data for this location/period likely "
            "drifted for reasons unrelated to actual air quality -- worth "
            "flagging as a known data-source limitation in your report rather "
            "than treating the trend as ground truth to fit."
        )

    print("\n" + "=" * 78)
    print("PER-ROW CORRELATION (matched rows only)")
    print("=" * 78)
    corr = matched["aqi"].corr(matched["om_us_aqi"])
    print(f"Correlation(OpenWeather-derived aqi, Open-Meteo om_us_aqi): {corr:.3f}")
    print(
        "High per-row correlation with a diverging multi-year trend would "
        "mean the two sources agree on short-term ups and downs but disagree "
        "on the long-run baseline -- consistent with a slow calibration "
        "drift in one source rather than a fundamental measurement problem."
    )


if __name__ == "__main__":
    main()