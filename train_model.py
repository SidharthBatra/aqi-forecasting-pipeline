"""
train_model.py

Trains AQI forecasting models for three separate horizons (24h, 48h, 72h)
using ONLY lagged/historical features -- no same-timestamp pollutant
concentrations as inputs to a same-timestamp target. That was the bug in
the first version (RF got RMSE=0.0168 / R2=0.9995 by reverse-engineering
the deterministic AQI formula from concentrations at the exact same
hour). This version predicts aqi_target_Hh = aqi shifted H hours into the
future, using only information available at "now" (t).

A note on one legitimate, minor mechanical overlap: the AQI target itself
is built (in compute_true_aqi.py) from a 24h trailing rolling average of
PM2.5/PM10 ending at the target's own timestamp. For the 24h horizon
specifically, the rolling window backing aqi_target_24h (spanning
[t, t+24]) technically includes the pollutant reading at t, one of 24
hours in that window. This is a real but tiny (~1/24) overlap, common
when forecasting any rolling-average-derived metric (e.g. forecasting a
7-day moving average using today's value), and is not comparable to the
original bug where the model predicted t from t directly. The 48h and 72h
targets have zero such overlap, since their windows fall entirely after
"now".

METHODOLOGY
-----------
1. Load aqi_historical_backfill_v2.csv (output of compute_true_aqi.py).
2. Reindex to a uniform hourly grid (gaps become NaN placeholder rows) so
   every shift()/rolling() call corresponds to a real elapsed-time offset
   instead of a row-position offset -- this dataset has 17 time gaps,
   including one 121-hour gap, so row-position shifting would silently
   misalign lag/target pairs around them.
3. Engineer lag features (aqi at t-1h/3h/6h/12h/24h/48h/72h, pm2_5/pm10 at
   t-24h), rolling stats (24h/72h/168h mean+std of aqi), and cyclical
   time features (hour/day-of-week/month, sin+cos encoded).
4. Build three shifted targets: aqi_target_24h/48h/72h via aqi.shift(-H).
5. Drop rows with any NaN in the required features/target for that
   horizon (this naturally excludes rows too close to the start/end of
   the series and rows whose lag or target window crosses a data gap).
6. Chronological train/test split per horizon (last 20% of the timeline
   by time, never shuffled) -- a random split would leak future
   information into training via overlapping rolling windows and lag
   features, and would overstate accuracy the same way the original
   same-timestamp bug did.
7. Train Random Forest, Ridge Regression, and a small TensorFlow MLP per
   horizon; evaluate RMSE/MAE/R2 on the held-out test set; save the best
   (lowest test RMSE) of the three; register it in the Hopsworks Model
   Registry as aqi_forecast_{H}h.

Requires HOPSWORKS_API_KEY in the environment to register models. If it's
not set, training and local artifact saving still run (useful for
iterating on the pipeline without touching the registry).

REVISION NOTES (after first real run on the full backfill)
------------------------------------------------------------
The first run of this script against the real 2-year dataset (45
auto-detected feature columns) got negative R2 for Random Forest and
Ridge at the 48h/72h horizons -- worse than just predicting the mean.
Two fixes went into this version:
  - 'year' (and the raw, non-cyclical 'hour'/'day'/'month'/'day_of_week'
    columns already present in the CSV) are now excluded from features.
    With a chronological split, the test set is a contiguous future
    block, so a raw calendar-year value takes on test-set values RF
    never trained on -- and RF cannot extrapolate past training ranges,
    unlike a linear model. This was a likely direct cause of RF's
    collapse.
  - A naive persistence baseline ("AQI in H hours = AQI right now") is
    now computed and printed alongside every horizon's results. Any
    trained model that can't beat it is flagged explicitly rather than
    silently registered as if it were a finished result.
  - Random Forest is now regularized (max_depth=12, min_samples_leaf=5,
    max_features='sqrt') instead of left effectively unbounded, and
    Ridge's alpha is chosen via RidgeCV with TimeSeriesSplit
    (forward-chaining CV folds) instead of a fixed guess, so the
    regularization strength is actually validated against out-of-time
    data rather than assumed.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

INPUT_CSV = "aqi_historical_backfill_v2.csv"
TIMESTAMP_COL = "timestamp_utc"
TARGET_COL = "aqi"
HORIZONS = [24, 48, 72]
LAG_HOURS = [1, 3, 6, 12, 24, 48, 72]
ROLLING_WINDOWS = [24, 72, 168]
MODEL_OUTPUT_DIR = "trained_models"

# Set up by setup_feature_view.py -- register_in_hopsworks() looks these up
# to link registered models to a real Feature View + Training Dataset for
# Hopsworks lineage tracking. TRAINING_DATASET_VERSION defaults to 1 but
# setup_feature_view.py prints the actual version it created; update this
# constant to match if it printed something other than 1.
FEATURE_VIEW_NAME = "aqi_forecast_fv"
FEATURE_VIEW_VERSION = 1
TRAINING_DATASET_VERSION = 1

# trend_source_crosscheck.py found OpenWeather's historical 'aqi' running
# 33-53% above the independent Open-Meteo figure in 2023-2024, converging
# to within 2-3% by 2025-2026 -- evidence that older OpenWeather historical
# reconstructions for this location were miscalibrated (running high), not
# that Karachi's air actually improved that much. Recency-weighting the
# training fit (not the test evaluation, which stays an honest unweighted
# read) lets the model use the full backfill window for seasonal shape
# while prioritizing the verified-accurate recent regime for absolute
# level. HALF_LIFE_DAYS=365 means a training row exactly one year older
# than the most recent data point gets half the fit weight of today's
# data, two years old gets a quarter, etc.
#
# NOTE: backfill_historical.py was reverted from a ~5.7yr pull back to the
# mentor's "2yr ideal" (BACKFILL_MONTHS=24), by explicit choice, after
# this recency-weighting logic was written for the 5.7yr case. Over only
# ~2 years, a 365-day half-life is a much gentler correction (oldest rows
# land around 0.5x weight instead of ~0.02x) -- worth re-checking whether
# it's still doing meaningful work, or whether the source-calibration
# issue is now small enough over this shorter window not to need it.
RECENCY_HALF_LIFE_DAYS = 365

# Columns that must never be used as model inputs: identifiers, and
# columns derived from / duplicating the target in ways that would leak
# or just add noise (the old OpenWeather 1-5 category, and the
# category/dominant-pollutant labels that are just relabelings of the
# CURRENT aqi value, already included as a feature in its own right).
#
# 'year' is deliberately excluded even though it's a legitimate
# same-timestamp value: with a chronological (non-shuffled) train/test
# split, the test set is a contiguous future block, so 'year' takes on
# values in test that are rare or absent in train. Random Forest cannot
# extrapolate past the range of values it split on during training, so a
# raw calendar-year feature actively breaks its ability to generalize to
# the held-out period -- this was likely a real contributor to RF's
# negative R2 on the first run. 'hour'/'day'/'month'/'day_of_week' are
# also excluded as raw integers since they're redundant with the
# hour_sin/cos, dow_sin/cos, month_sin/cos features already engineered
# below, and cutting the redundancy reduces overfitting surface for RF.
EXCLUDE_COLUMNS = {
    "aqi_dominant_pollutant",
    "aqi_category",
    "aqi_index_openweather_1to5",
    "row_id",
    "year",
    "hour",
    "day",
    "month",
    "day_of_week",
    # NOTE: aqi_change_rate is intentionally NOT excluded here anymore.
    # thorough_eda_v2.py found its zero-rate (93.6%) exactly matched the
    # OLD 1-5 category's unchanged-rate, proving it was computed against
    # the stale target -- compute_true_aqi.py now recomputes it against
    # the real continuous 'aqi' before writing aqi_historical_backfill_v2.csv,
    # so it's a legitimate feature again as of that fix.
    #
    # Confirmed via thorough_eda_v2.py section 7: correlates 0.97 with
    # raw pm2_5 itself, so as currently computed it mostly just echoes
    # the pollution magnitude rather than adding an independent
    # source-disagreement signal. Revisit once backfill_historical.py's
    # formula for this column is confirmed/fixed to be relative rather
    # than absolute.
    "pm25_source_diff",
    # om_* (Open-Meteo cross-check columns) and pm25_source_diff_pct
    # (which depends on om_pm2_5) are excluded for two independent
    # reasons, either of which would be sufficient on its own:
    #   1. Train-serve skew: Open-Meteo was only ever wired up as a
    #      backfill-time cross-check source. fetch_features.py (the live
    #      hourly collector) doesn't produce these fields, so a model
    #      trained on them would have inputs missing at real-time
    #      inference.
    #   2. On the real extended (~5.7yr) backfill, om_* is ~60% missing
    #      (Open-Meteo timed out on several chunks pulling this much
    #      history) and missingness is fully correlated across all 7
    #      om_* columns (same failed API calls). train_horizon()'s
    #      dropna() requires every feature non-null, so leaving these in
    #      would silently discard ~60% of the newly-tripled dataset --
    #      largely undoing the point of extending the backfill.
    "om_pm2_5",
    "om_pm10",
    "om_co",
    "om_no2",
    "om_so2",
    "om_o3",
    "om_us_aqi",
    "pm25_source_diff_pct",
}


# Must match ingest_to_hopsworks.py's FEATURE_GROUP_NAME/VERSION -- kept as
# separate constants here (not imported from that script) since CI/CD's
# training_pipeline.yml runs train_model.py standalone. Update both if one
# changes.
RAW_FEATURE_GROUP_NAME = "aqi_karachi_features"
RAW_FEATURE_GROUP_VERSION = 4


def read_source_data():
    """Local CSV if present -- that's the Codespace/dev-loop path, and
    stays the default so nothing about the existing interactive workflow
    changes. Falls back to pulling the raw feature group straight from
    Hopsworks when the CSV isn't there, which is the ONLY option in
    GitHub Actions: runners are stateless/ephemeral, so
    training_pipeline.yml's checkout never includes the local backfill
    CSV (it isn't and shouldn't be committed to git -- it's tens of
    thousands of rows that change hourly). This is what makes the daily
    CI/CD training pipeline actually able to run unattended."""
    if os.path.isfile(INPUT_CSV):
        print(f"Loading local {INPUT_CSV} ...")
        return pd.read_csv(INPUT_CSV, parse_dates=[TIMESTAMP_COL])

    print(f"{INPUT_CSV} not found locally -- pulling from Hopsworks feature group instead...")
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"Neither {INPUT_CSV} nor HOPSWORKS_API_KEY is available -- "
            f"there's no source to load training data from."
        )
    import hopsworks

    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=RAW_FEATURE_GROUP_NAME, version=RAW_FEATURE_GROUP_VERSION)
    df = fg.read()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    print(
        f"Pulled {len(df)} rows from Hopsworks feature group "
        f"{RAW_FEATURE_GROUP_NAME} v{RAW_FEATURE_GROUP_VERSION}."
    )
    return df


def load_and_prepare_grid(path=None):
    # `path` is accepted (and still passed as INPUT_CSV by existing call
    # sites, e.g. setup_feature_view.py) for backward compatibility, but
    # ignored -- read_source_data() is what actually decides local-CSV vs
    # Hopsworks, using the module-level INPUT_CSV constant either way.
    df = read_source_data()
    df = df.sort_values(TIMESTAMP_COL).drop_duplicates(subset=[TIMESTAMP_COL])
    df = df.set_index(TIMESTAMP_COL)

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    n_missing = len(full_index) - len(df)
    print(
        f"Reindexing to a uniform hourly grid: {len(df)} actual rows -> "
        f"{len(full_index)} grid rows ({n_missing} gap-hours become NaN "
        f"placeholders so lag/shift math stays time-accurate across gaps)."
    )
    df = df.reindex(full_index)
    df.index.name = TIMESTAMP_COL
    return df


def add_time_features(df):
    hours = df.index.hour.values
    dow = df.index.dayofweek.values
    month = df.index.month.values

    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


def add_lag_and_rolling_features(df):
    for lag in LAG_HOURS:
        df[f"aqi_lag_{lag}h"] = df[TARGET_COL].shift(lag)

    for window in ROLLING_WINDOWS:
        min_p = max(3, window // 4)
        df[f"aqi_rollmean_{window}h"] = df[TARGET_COL].rolling(window, min_periods=min_p).mean()
        df[f"aqi_rollstd_{window}h"] = df[TARGET_COL].rolling(window, min_periods=min_p).std()

    for pol in ["pm2_5", "pm10"]:
        if pol in df.columns:
            df[f"{pol}_lag_24h"] = df[pol].shift(24)

    return df


def add_targets(df):
    for h in HORIZONS:
        df[f"aqi_target_{h}h"] = df[TARGET_COL].shift(-h)
    return df


def get_feature_columns(df):
    exclude = set(EXCLUDE_COLUMNS) | {f"aqi_target_{h}h" for h in HORIZONS}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features = [c for c in numeric_cols if c not in exclude]
    return features


def chronological_split(df, test_frac=0.2):
    n = len(df)
    split_idx = int(n * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def compute_recency_weights(index, reference_ts, half_life_days=RECENCY_HALF_LIFE_DAYS):
    """Exponential-decay sample weights: a row exactly `half_life_days` older
    than `reference_ts` gets half the weight of a row AT `reference_ts`, one
    half-life further back gets a quarter, etc. `reference_ts` is fixed to
    the single most recent timestamp in the WHOLE dataset (passed in from
    main(), not recomputed per-horizon/per-split) so that weights are
    directly comparable across the 24h/48h/72h horizons and across the
    train/test boundary -- if it were recomputed from each horizon's own
    train_df, the same calendar date would get a different weight at
    different horizons purely because their train sets end at different
    points, which would make the RECENCY_HALF_LIFE_DAYS constant mean a
    different thing in each run.

    Applied ONLY to the training split -- see train_horizon(), which never
    passes these weights into the test-set evaluate() calls or the
    persistence baseline. Weighting the test set would make the reported
    RMSE/MAE/R2 an optimistic, recency-biased read rather than an honest
    measure of forecast accuracy across the whole held-out period.
    """
    age_days = (reference_ts - index).total_seconds() / 86400.0
    # Guard against any pathological negative age (e.g. reference_ts passed
    # in wrong) collapsing weights to >1 or NaN via a negative exponent.
    age_days = np.clip(age_days, 0, None)
    weights = 0.5 ** (age_days / half_life_days)
    return weights.values if hasattr(weights, "values") else np.asarray(weights)


def evaluate(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def build_tf_model(n_features):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(n_features,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_horizon(df, horizon, feature_cols, most_recent_ts):
    target_col = f"aqi_target_{horizon}h"
    cols_needed = feature_cols + [target_col]
    clean = df[cols_needed].dropna()
    print(
        f"\n=== Horizon {horizon}h: {len(clean)} usable rows "
        f"(dropped {len(df) - len(clean)} rows with NaN in required "
        f"features/target) ==="
    )

    train_df, test_df = chronological_split(clean, test_frac=0.2)
    X_train, y_train = train_df[feature_cols].values, train_df[target_col].values
    X_test, y_test = test_df[feature_cols].values, test_df[target_col].values
    print(
        f"Chronological split: {len(train_df)} train rows, {len(test_df)} test rows "
        f"(test = most recent 20% of the timeline, never shuffled into train)"
    )

    # Recency weights -- TRAIN split only. See compute_recency_weights()
    # docstring and the RECENCY_HALF_LIFE_DAYS comment near the top of this
    # file for why: trend_source_crosscheck.py found OpenWeather's
    # historical 'aqi' running 33-53% high in 2023-2024 vs. an independent
    # source, converging by 2025-2026. w_train down-weights (without
    # dropping) the older, less-trustworthy rows during fitting; the test
    # split, persistence baseline, and evaluate() calls below stay
    # completely unweighted so the reported RMSE/MAE/R2 remain an honest,
    # un-recency-biased read of held-out accuracy.
    w_train = compute_recency_weights(train_df.index, most_recent_ts)
    oldest_w, newest_w = w_train.min(), w_train.max()
    print(
        f"Recency weights (train only, half-life={RECENCY_HALF_LIFE_DAYS}d): "
        f"range [{oldest_w:.4f}, {newest_w:.4f}]"
    )

    results = {}

    # --- Naive persistence baseline: "AQI in H hours = AQI right now" ---
    # Not a model, not saved/registered -- purely a sanity floor. Any
    # trained model that can't beat this on a strict future holdout has a
    # real problem, not just "forecasting is hard".
    if "aqi" in df.columns:
        baseline_pred = test_df["aqi"].values
        b_rmse, b_mae, b_r2 = evaluate(y_test, baseline_pred)
        print(f"[baseline] Persistence RMSE={b_rmse:.3f}  MAE={b_mae:.3f}  R2={b_r2:.4f}")
    else:
        b_rmse = b_mae = b_r2 = None

    # --- Random Forest (no scaling needed) ---
    # max_depth/min_samples_leaf/max_features are deliberately conservative
    # (not left at sklearn defaults or an unbounded depth) -- with 40+
    # partly-redundant features (multiple aqi lags, rolling windows, two
    # independent pollutant sources) an unconstrained forest overfits the
    # training block and that overfitting shows up as negative R2 the
    # moment it's evaluated on a genuinely out-of-time test set.
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train, sample_weight=w_train)
    rmse, mae, r2 = evaluate(y_test, rf.predict(X_test))
    results["random_forest"] = {"model": rf, "rmse": rmse, "mae": mae, "r2": r2, "scaler": None}
    print(f"Random Forest     RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}")

    # --- Ridge Regression (needs scaling; scaler fit on train only) ---
    # Alpha is chosen via RidgeCV over a log-spaced grid using
    # TimeSeriesSplit (forward-chaining folds) rather than a fixed guess
    # or standard k-fold CV -- standard k-fold would shuffle temporally
    # adjacent rows into different folds, which leaks information through
    # the overlapping lag/rolling-window features and would pick an alpha
    # that looks good under leakage but under-regularizes for real
    # out-of-time use.
    #
    # Grid widened from [1e-2, 1e3] to [1e-2, 1e6] after the real run on
    # the extended (~5.7yr) backfill: RidgeCV picked alpha=1e3 -- the exact
    # top edge of the old grid -- at both the 48h and 72h horizons. Landing
    # on a grid boundary means the search wants MORE regularization than it
    # was allowed to try, not that 1e3 is actually optimal. Left as-is, the
    # model is silently under-regularized at exactly the horizons that were
    # already failing to beat the persistence baseline.
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    ridge = RidgeCV(alphas=np.logspace(-2, 6, 18), cv=TimeSeriesSplit(n_splits=5))
    ridge.fit(X_train_s, y_train, sample_weight=w_train)
    rmse, mae, r2 = evaluate(y_test, ridge.predict(X_test_s))
    results["ridge"] = {"model": ridge, "rmse": rmse, "mae": mae, "r2": r2, "scaler": scaler}
    print(f"Ridge Regression  RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}  (alpha={ridge.alpha_:.3g})")
    max_alpha = 10 ** 6
    if ridge.alpha_ >= max_alpha * 0.99:
        print(
            f"  WARNING: alpha landed on the top edge of the search grid "
            f"again ({ridge.alpha_:.3g}) -- the true optimum may be even "
            f"higher. Consider widening the grid further if this recurs."
        )

    # --- TensorFlow Neural Net (reuses the Ridge scaler) ---
    tf_model = build_tf_model(X_train_s.shape[1])
    early_stop = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    # Keras applies validation_split BEFORE shuffling/weighting by slicing
    # off the LAST 15% of the (already chronologically-ordered) array --
    # so this remains a held-out-in-time internal validation slice, and
    # sample_weight is only ever applied to the training portion of that
    # split, consistent with w_train/w_test throughout this function.
    tf_model.fit(
        X_train_s,
        y_train,
        sample_weight=w_train,
        validation_split=0.15,
        epochs=100,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0,
    )
    preds = tf_model.predict(X_test_s, verbose=0).flatten()
    rmse, mae, r2 = evaluate(y_test, preds)
    results["tensorflow_nn"] = {
        "model": tf_model,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "scaler": scaler,
    }
    print(f"TensorFlow NN     RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}")

    best_name = min(results, key=lambda k: results[k]["rmse"])
    print(f"Best for {horizon}h horizon: {best_name} (RMSE={results[best_name]['rmse']:.3f})")
    if b_rmse is not None and b_rmse < results[best_name]["rmse"]:
        print(
            f"  WARNING: the persistence baseline (RMSE={b_rmse:.3f}) still beats "
            f"every trained model at this horizon -- treat these model results as "
            f"not yet production-worthy, not as a finished result."
        )

    baseline = {"rmse": b_rmse, "mae": b_mae, "r2": b_r2}
    return results, best_name, baseline


def save_model_artifacts(horizon, model_name, model_info, feature_cols):
    out_dir = os.path.join(MODEL_OUTPUT_DIR, f"aqi_forecast_{horizon}h")
    os.makedirs(out_dir, exist_ok=True)

    if model_name == "tensorflow_nn":
        model_info["model"].save(os.path.join(out_dir, "model.keras"))
    else:
        joblib.dump(model_info["model"], os.path.join(out_dir, "model.joblib"))

    if model_info["scaler"] is not None:
        joblib.dump(model_info["scaler"], os.path.join(out_dir, "scaler.joblib"))

    metadata = {
        "horizon_hours": horizon,
        "model_type": model_name,
        "feature_columns": feature_cols,
        "metrics": {"rmse": model_info["rmse"], "mae": model_info["mae"], "r2": model_info["r2"]},
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return out_dir


def register_in_hopsworks(project, horizon, model_name, model_info, out_dir):
    mr = project.get_model_registry()

    # Link the registered model to the engineered Feature View + Training
    # Dataset built by setup_feature_view.py, so Hopsworks' lineage
    # tracking (the "Feature Views" / "Used to train Models" / "Models
    # Using Features" counts on the project overview) reflects real usage
    # instead of staying at 0. This does NOT change what data the model
    # was actually trained on -- that's still train_model.py's local
    # chronological split, computed before this function is ever called.
    # If setup_feature_view.py hasn't been run yet, or the installed hsml
    # version doesn't accept these kwargs on create_model(), this falls
    # back to plain registration (the original behavior) rather than
    # failing the run over a lineage nice-to-have.
    feature_view = None
    try:
        fs = project.get_feature_store()
        feature_view = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
    except Exception as e:
        print(
            f"  NOTE: couldn't look up feature view {FEATURE_VIEW_NAME} "
            f"v{FEATURE_VIEW_VERSION} ({type(e).__name__}: {e}) -- "
            f"registering without Feature Store lineage. Run "
            f"setup_feature_view.py first to enable it."
        )

    create_model_kwargs = dict(
        name=f"aqi_forecast_{horizon}h",
        metrics={"rmse": model_info["rmse"], "mae": model_info["mae"], "r2": model_info["r2"]},
        description=(
            f"Best of RF/Ridge/TensorFlow for {horizon}h-ahead AQI forecast "
            f"(model_type={model_name}), trained on lagged/historical features only."
        ),
    )

    hw_model = None
    linked = False
    if feature_view is not None:
        try:
            hw_model = mr.python.create_model(
                feature_view=feature_view,
                training_dataset_version=TRAINING_DATASET_VERSION,
                **create_model_kwargs,
            )
            linked = True
        except TypeError as e:
            print(
                f"  NOTE: this hsml version's create_model() doesn't accept "
                f"feature_view/training_dataset_version ({e}) -- "
                f"registering without lineage instead."
            )

    if hw_model is None:
        hw_model = mr.python.create_model(**create_model_kwargs)

    hw_model.save(out_dir)
    print(
        f"Registered aqi_forecast_{horizon}h (v{hw_model.version}) in "
        f"Hopsworks Model Registry"
        + (f", linked to feature view {FEATURE_VIEW_NAME} v{FEATURE_VIEW_VERSION}" if linked else "")
    )


def main():
    df = load_and_prepare_grid(INPUT_CSV)
    df = add_time_features(df)
    df = add_lag_and_rolling_features(df)
    df = add_targets(df)

    feature_cols = get_feature_columns(df)
    print(f"\nUsing {len(feature_cols)} feature columns:")
    for c in feature_cols:
        print(f"  - {c}")

    project = None
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if api_key:
        import hopsworks

        project = hopsworks.login(api_key_value=api_key)
    else:
        print(
            "\nWARNING: HOPSWORKS_API_KEY not set -- training will run and "
            "artifacts will be saved locally, but nothing will be registered "
            "in the Model Registry."
        )

    # Fixed once, from the full dataset -- see compute_recency_weights()
    # docstring for why this must NOT be recomputed per-horizon.
    most_recent_ts = df.index.max()

    summary = []
    for horizon in HORIZONS:
        results, best_name, baseline = train_horizon(df, horizon, feature_cols, most_recent_ts)
        best_info = results[best_name]
        out_dir = save_model_artifacts(horizon, best_name, best_info, feature_cols)
        summary.append(
            {
                "horizon": horizon,
                "best_model": best_name,
                "baseline": baseline,
                **{k: best_info[k] for k in ("rmse", "mae", "r2")},
            }
        )

        if project is not None:
            # Wrapped in try/except after a real run (2026-08-24) crashed
            # the ENTIRE script -- including horizons not yet trained --
            # when a transient network error (RemoteDisconnected) hit
            # mid-upload during the 48h model's hw_model.save(). 72h never
            # got a chance to train as a result. A registration failure at
            # one horizon is now a warning, not a fatal error, so the loop
            # continues to the remaining horizons; the model is still
            # saved locally in MODEL_OUTPUT_DIR either way via
            # save_model_artifacts() above, so nothing is lost -- it just
            # needs a manual/re-run push to Hopsworks later if this fires.
            try:
                register_in_hopsworks(project, horizon, best_name, best_info, out_dir)
            except Exception as e:
                print(
                    f"  WARNING: registering aqi_forecast_{horizon}h in Hopsworks "
                    f"failed ({type(e).__name__}: {e}). Model artifacts are still "
                    f"saved locally in {out_dir} -- continuing to the next horizon "
                    f"rather than aborting the whole run."
                )

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for row in summary:
        print(
            f"  {row['horizon']}h -> {row['best_model']:15s} "
            f"RMSE={row['rmse']:.3f}  MAE={row['mae']:.3f}  R2={row['r2']:.4f}"
            f"   | persistence baseline RMSE={row['baseline']['rmse']:.3f}"
        )


if __name__ == "__main__":
    main()