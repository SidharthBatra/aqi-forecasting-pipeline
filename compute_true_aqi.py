"""
compute_true_aqi.py

Recomputes a standard, continuous AQI (US EPA breakpoint methodology,
0-500+ scale) from the raw pollutant concentrations already present in
aqi_historical_backfill.csv.

WHY THIS SCRIPT EXISTS
-----------------------
EDA showed the existing `aqi_index` column (range 1.0-5.56, 93.6% of hours
with zero hour-to-hour change) is OpenWeather's own coarse 1-5 category
(1=Good ... 5=Very Poor) -- confirmed directly, not the continuous 0-500
AQI used in official reporting (US EPA-style, and what AQICN reports).

The mentor's instruction "predict AQI directly, not PM2.5 converted to
AQI" is about the model's OUTPUT -- it shouldn't predict a pollutant
concentration and then post-hoc convert it. It still needs a properly
computed AQI as the TRAINING TARGET, and that requires running pollutant
concentrations through the EPA breakpoint formula. Doing that here, once,
to build the target column is standard practice, not the thing the
mentor was warning against.

METHODOLOGY
-----------
For each pollutant:
  1. Convert OpenWeather's ug/m3 concentration to the units the EPA
     breakpoint tables use (ppm for CO, ppb for NO2/SO2/O3; PM2.5/PM10
     stay in ug/m3).
  2. Apply the EPA-specified averaging window using a TIME-BASED rolling
     mean (not a row-count window, so the 17 gaps in this dataset --
     including one 121-hour gap -- don't silently pull in data across a
     hole):
        PM2.5, PM10  -> 24-hour rolling mean
        O3, CO       -> 8-hour rolling mean
        NO2, SO2     -> 1-hour (raw) value
     (Simplification: official EPA methodology switches SO2 to a 24h
     average above AQI 200. Not implemented here since PM2.5 is almost
     always the dominant pollutant in Karachi -- see aqi_dominant_pollutant
     in the output if you want to confirm that assumption.)
  3. Run each averaged concentration through the official piecewise-linear
     EPA breakpoint formula to get a per-pollutant sub-index.
  4. Take the max sub-index across pollutants as the overall AQI for that
     hour (the standard "dominant pollutant" approach -- the same method
     AQICN uses for the number this pipeline already cross-checks
     against).

The EPA table officially tops out at AQI 500 (PM2.5 = 500.4 ug/m3).
Karachi's backfill has PM2.5 up to ~968 ug/m3 (confirmed plausible by
check_data_quality.py), so concentrations above the top breakpoint are
extrapolated linearly using the slope of the top segment, uncapped --
matching how real-time trackers handle extreme pollution days rather than
flatlining everything severe at 500.

OUTPUT
------
Writes aqi_historical_backfill_v2.csv (original file untouched):
  - aqi                         : new continuous AQI target -- what
                                   train_model.py should predict
  - aqi_dominant_pollutant      : which pollutant produced the max
                                   sub-index (useful later for SHAP /
                                   alert messaging)
  - aqi_category                : Good / Moderate / Unhealthy for
                                   Sensitive Groups / Unhealthy / Very
                                   Unhealthy / Hazardous
  - aqi_index_openweather_1to5  : the old OpenWeather category, kept for
                                   reference/comparison only -- NOT a
                                   model target going forward
"""

import numpy as np
import pandas as pd

INPUT_CSV = "aqi_historical_backfill.csv"
OUTPUT_CSV = "aqi_historical_backfill_v2.csv"
TIMESTAMP_COL = "timestamp_utc"

# Column names for raw pollutant concentrations (ug/m3), as fetched from
# OpenWeather's Air Pollution API. Edit these if your CSV uses different
# names.
POLLUTANT_COLUMNS = {
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "co": "co",
    "no2": "no2",
    "so2": "so2",
    "o3": "o3",
}

# Molar masses (g/mol) for ug/m3 -> ppb/ppm conversion at 25C, 1 atm
# (molar volume = 24.45 L/mol)
MOLAR_MASS = {"co": 28.01, "no2": 46.0055, "so2": 64.066, "o3": 48.00}
MOLAR_VOLUME = 24.45

# EPA breakpoint tables: (conc_lo, conc_hi, aqi_lo, aqi_hi)
BREAKPOINTS = {
    "pm2_5": [  # ug/m3, 24h avg
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ],
    "pm10": [  # ug/m3, 24h avg
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ],
    "co": [  # ppm, 8h avg
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500),
    ],
    "so2": [  # ppb, 1h avg
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 804, 301, 400),
        (805, 1004, 401, 500),
    ],
    "no2": [  # ppb, 1h avg
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500),
    ],
    "o3": [  # ppb, 8h avg (table valid to AQI 300; Karachi's AQI is
        # essentially always PM-driven so the 1h high-O3 table isn't
        # implemented)
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300),
    ],
}


def ugm3_to_ppb(conc_ugm3, molar_mass):
    return (conc_ugm3 * MOLAR_VOLUME) / molar_mass


def ugm3_to_ppm(conc_ugm3, molar_mass):
    return ugm3_to_ppb(conc_ugm3, molar_mass) / 1000.0


def breakpoint_interp(conc, table):
    """Standard EPA piecewise-linear breakpoint formula, with linear
    extrapolation past the top of the table for extreme pollution events."""
    if pd.isna(conc) or conc < 0:
        return np.nan

    for lo_c, hi_c, lo_i, hi_i in table:
        if lo_c <= conc <= hi_c:
            return (hi_i - lo_i) / (hi_c - lo_c) * (conc - lo_c) + lo_i

    if conc > table[-1][1]:
        lo_c, hi_c, lo_i, hi_i = table[-1]
        slope = (hi_i - lo_i) / (hi_c - lo_c)
        return hi_i + slope * (conc - hi_c)

    return np.nan  # below zero handled above; shouldn't hit this branch


def compute_sub_indices(df_indexed):
    """df_indexed must have a sorted, tz-aware DatetimeIndex."""
    sub = {}

    # PM2.5, PM10: 24h time-based rolling mean, ug/m3, no unit conversion
    for pol in ["pm2_5", "pm10"]:
        col = POLLUTANT_COLUMNS[pol]
        rolled = df_indexed[col].rolling("24h", min_periods=1).mean()
        sub[pol] = rolled.apply(lambda c, p=pol: breakpoint_interp(c, BREAKPOINTS[p]))

    # O3: 8h time-based rolling mean, ug/m3 -> ppb
    col = POLLUTANT_COLUMNS["o3"]
    rolled_ppb = df_indexed[col].rolling("8h", min_periods=1).mean().apply(
        lambda c: ugm3_to_ppb(c, MOLAR_MASS["o3"])
    )
    sub["o3"] = rolled_ppb.apply(lambda c: breakpoint_interp(c, BREAKPOINTS["o3"]))

    # CO: 8h time-based rolling mean, ug/m3 -> ppm
    col = POLLUTANT_COLUMNS["co"]
    rolled_ppm = df_indexed[col].rolling("8h", min_periods=1).mean().apply(
        lambda c: ugm3_to_ppm(c, MOLAR_MASS["co"])
    )
    sub["co"] = rolled_ppm.apply(lambda c: breakpoint_interp(c, BREAKPOINTS["co"]))

    # NO2, SO2: raw 1h value, ug/m3 -> ppb
    for pol in ["no2", "so2"]:
        col = POLLUTANT_COLUMNS[pol]
        ppb = df_indexed[col].apply(lambda c, p=pol: ugm3_to_ppb(c, MOLAR_MASS[p]))
        sub[pol] = ppb.apply(lambda c: breakpoint_interp(c, BREAKPOINTS[pol]))

    return pd.DataFrame(sub, index=df_indexed.index)


def categorize(aqi):
    if pd.isna(aqi):
        return np.nan
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def clean_sentinel_values(df):
    """Pollutant concentrations can never legitimately be negative. Some
    historical hours from the backfill (older OpenWeather responses, and
    potentially Open-Meteo) carry a -9999 "no data" sentinel instead of a
    real reading or a null -- seen directly in the real backfill's outlier
    scan (pm10/no2/o3 all had actual_range starting at -9999.0). Left as
    -9999, that's not an outlier a model can shrug off, it's a value that
    actively lies about the pollution level. Replace any negative reading
    in any concentration column with NaN so it's treated as genuinely
    missing rather than as data."""
    conc_cols = list(POLLUTANT_COLUMNS.values()) + [
        c for c in ["om_pm2_5", "om_pm10", "om_co", "om_no2", "om_so2", "om_o3", "nh3", "no"]
        if c in df.columns
    ]
    conc_cols = sorted(set(c for c in conc_cols if c in df.columns))
    total_replaced = 0
    for col in conc_cols:
        bad = df[col] < 0
        n_bad = int(bad.sum())
        if n_bad:
            print(f"  {col}: replacing {n_bad} negative/sentinel values with NaN")
            df.loc[bad, col] = np.nan
            total_replaced += n_bad
    if total_replaced == 0:
        print("  No negative/sentinel values found.")
    return df


def main():
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV, parse_dates=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # backfill_historical.py's chunked OpenWeather History API calls
    # overlap by one hour at each chunk boundary (confirmed: full 30-day
    # chunks return 721 hourly records instead of 720), producing duplicate
    # timestamp rows. After sorting, those sit adjacent with zero elapsed
    # time between them, which breaks any time-based diff (this is what
    # caused aqi_change_rate to show -inf/inf/NaN in thorough_eda_v2.py).
    # Drop the duplicates here so it's fixed without needing to re-run the
    # multi-minute API backfill.
    n_before = len(df)
    df = df.drop_duplicates(subset=[TIMESTAMP_COL], keep="first").reset_index(drop=True)
    n_dupes = n_before - len(df)
    if n_dupes:
        print(f"Dropped {n_dupes} duplicate-timestamp rows (chunk-boundary overlap).")

    missing = [c for c in POLLUTANT_COLUMNS.values() if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing expected pollutant columns: {missing}. "
            f"Edit POLLUTANT_COLUMNS at the top of this script to match "
            f"your CSV's actual column names, then re-run."
        )

    print("Scanning for sentinel/negative concentration values...")
    df = clean_sentinel_values(df)

    df_indexed = df.set_index(TIMESTAMP_COL)  # rolling() needs a DatetimeIndex

    print("Computing per-pollutant EPA sub-indices (time-based rolling windows)...")
    sub_indices = compute_sub_indices(df_indexed).reset_index(drop=True)

    df["aqi"] = sub_indices.max(axis=1)
    df["aqi_dominant_pollutant"] = sub_indices.idxmax(axis=1)
    df["aqi_category"] = df["aqi"].apply(categorize)

    # thorough_eda_v2.py confirmed aqi_change_rate's zero-rate (93.6%)
    # exactly matched how often the OLD 1-5 aqi_index failed to change --
    # it was computed upstream in backfill_historical.py against that
    # stale field, before this script's continuous 'aqi' existed. Recompute
    # it properly now, against the real target, using the same
    # elapsed-hours normalization as the original.
    if "aqi_change_rate" in df.columns:
        hours_elapsed = df[TIMESTAMP_COL].diff().dt.total_seconds() / 3600.0
        # Belt-and-suspenders: the duplicate-timestamp dedup above should
        # already prevent hours_elapsed==0, but guard explicitly anyway --
        # a stray zero here produces +/-inf, which silently corrupts
        # every downstream describe()/corr()/StandardScaler call it
        # touches (that's exactly what produced the -inf/inf/NaN seen in
        # thorough_eda_v2.py's staleness check).
        rate = df["aqi"].diff() / hours_elapsed
        rate = rate.replace([np.inf, -np.inf], np.nan)
        df["aqi_change_rate"] = rate.round(4)

    if "aqi_index" in df.columns:
        df = df.rename(columns={"aqi_index": "aqi_index_openweather_1to5"})

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {OUTPUT_CSV} ({len(df)} rows)")
    print(
        f"New 'aqi' column: min={df['aqi'].min():.1f}, max={df['aqi'].max():.1f}, "
        f"mean={df['aqi'].mean():.1f}, std={df['aqi'].std():.2f}"
    )
    print(f"NaN rows in 'aqi' (missing pollutant inputs): {df['aqi'].isna().sum()}")
    print("\nDominant pollutant breakdown:")
    print(df["aqi_dominant_pollutant"].value_counts())
    print("\nAQI category breakdown:")
    print(df["aqi_category"].value_counts())
    print(
        "\nNext: run eda_recheck.py against the new 'aqi' column before we "
        "rebuild train_model.py -- a continuous target will have a "
        "different signal profile than the coarse 1-5 field did."
    )


if __name__ == "__main__":
    main()