"""
dashboard.py

Streamlit dashboard for the Karachi AQI forecasting pipeline. Everything
here is cloud-first, matching the rest of this project's
"Hopsworks-first, nothing self-hosted" principle (see train_model.py's
read_source_data()):

  - Feature data comes from the Hopsworks Feature Store, feature group
    aqi_karachi_features (see ingest_to_hopsworks.py / fetch_features.py).
  - Models come from the Hopsworks Model Registry (aqi_forecast_24h/48h/72h),
    downloaded fresh via the API -- nothing shipped as a local model file.
  - The "current AQI" shown here is never recomputed from a different
    formula: fetch_features.py already writes 'aqi' / 'aqi_category' /
    'aqi_dominant_pollutant' into the feature group using
    compute_true_aqi.py's exact EPA breakpoint methodology every hour, so
    this dashboard just reads that column back.

HONESTY NOTE ON MODEL QUALITY (see metadata.json per horizon / project
README): the 24h model is a genuinely useful forecaster (R2=0.71). The
48h model is weaker but still informative (R2=0.35). The 72h model beats
the naive persistence baseline only marginally (R2=0.08) -- it is real
signal, not noise, but should be read as a low-confidence directional
hint, not a reliable forecast. The UI below visually de-emphasizes 72h
accordingly (muted styling + an explicit caption) rather than presenting
all three horizons as equally trustworthy.

Requires HOPSWORKS_API_KEY in the environment (same pattern as every
other script in this repo -- never hardcoded).
"""

import json
import os
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from aqi_alerts import CATEGORY_COLORS, CATEGORY_SEVERITY, check_alerts
from compute_true_aqi import categorize
from train_model import (
    HORIZONS,
    RAW_FEATURE_GROUP_NAME,
    RAW_FEATURE_GROUP_VERSION,
    TIMESTAMP_COL,
    add_lag_and_rolling_features,
    add_time_features,
)

# How many days of history to plot on the time-series chart. The full raw
# feature group is always pulled (needed anyway to compute the 168h rolling
# window feature -- see train_model.ROLLING_WINDOWS), so this is purely a
# display choice, not a feature-engineering requirement.
CHART_HISTORY_DAYS = 10

# Visual "confidence" treatment per horizon, driven by each model's real
# held-out R2 (see metadata.json) -- not a cosmetic choice. Kept as a plain
# dict here (rather than re-deriving it from live metrics) so the labels
# stay stable even if a retrain nudges R2 slightly; the ordering (24h more
# trustworthy than 72h) is the fixed, structural fact worth encoding.
CONFIDENCE_LABELS = {
    24: ("High(er) confidence", "#1b8a5a"),
    48: ("Moderate confidence", "#c98a12"),
    72: ("Low confidence -- directional only", "#8a2020"),
}

st.set_page_config(page_title="Karachi AQI Forecast", layout="wide")


# ---------------------------------------------------------------------------
# Hopsworks connection + data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to Hopsworks...")
def get_project():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "HOPSWORKS_API_KEY environment variable is not set. Set it to a "
            "valid Hopsworks API key (Project Settings -> API Keys) before "
            "running this dashboard."
        )
    import hopsworks

    return hopsworks.login(api_key_value=api_key)


@st.cache_data(ttl=3600, show_spinner="Pulling latest features from Hopsworks...")
def load_raw_features(_project):
    fs = _project.get_feature_store()
    fg = fs.get_feature_group(name=RAW_FEATURE_GROUP_NAME, version=RAW_FEATURE_GROUP_VERSION)
    df = fg.read()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    df = df.sort_values(TIMESTAMP_COL).drop_duplicates(subset=[TIMESTAMP_COL], keep="last")
    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Engineering lag/rolling/time features...")
def build_engineered_features(raw_df):
    """Reproduces train_model.py's load_and_prepare_grid + add_time_features +
    add_lag_and_rolling_features exactly, so the feature vector fed to each
    model at inference time matches training's feature schema precisely.
    Imported functions, not reimplemented, to guarantee that match."""
    d = raw_df.set_index(TIMESTAMP_COL)
    full_index = pd.date_range(d.index.min(), d.index.max(), freq="h")
    d = d.reindex(full_index)
    d.index.name = TIMESTAMP_COL
    d = add_time_features(d)
    d = add_lag_and_rolling_features(d)
    return d


def get_latest_valid_row(engineered_df):
    """The most recent row with a real (non-NaN) 'aqi' reading -- NOT
    necessarily engineered_df.iloc[-1], since a paused feature pipeline (see
    project notes: feature_pipeline.yml is currently paused to save
    free-tier quota) can leave the tail of the hourly grid as gap
    placeholders rather than real data."""
    valid = engineered_df[engineered_df["aqi"].notna()]
    if valid.empty:
        raise RuntimeError(
            "No row with a valid 'aqi' value found in the feature data pulled "
            "from Hopsworks -- the feature group may be empty or missing the "
            "'aqi' column."
        )
    return valid.iloc[-1]


# ---------------------------------------------------------------------------
# Model Registry (cached)
# ---------------------------------------------------------------------------
def _resolve_model_handle(mr, name):
    """Prefer the best-performing registered version (lowest RMSE); some
    hsml client versions don't expose get_best_model, so fall back to the
    highest version number (the most recently registered/trained run)."""
    try:
        return mr.get_best_model(name=name, metric="rmse", direction="min")
    except Exception as e:
        print(f"  NOTE: get_best_model unavailable for {name} ({e}) -- falling back to latest version.")
        candidates = mr.get_models(name)
        if not candidates:
            raise RuntimeError(f"No versions of model '{name}' found in the Hopsworks Model Registry.")
        return max(candidates, key=lambda m: m.version)


@st.cache_resource(show_spinner="Loading models from the Hopsworks Model Registry...")
def load_models(_project):
    mr = _project.get_model_registry()
    models = {}
    for h in HORIZONS:
        name = f"aqi_forecast_{h}h"
        hw_model = _resolve_model_handle(mr, name)
        model_dir = hw_model.download()

        with open(os.path.join(model_dir, "metadata.json")) as f:
            metadata = json.load(f)

        model_type = metadata["model_type"]
        if model_type == "tensorflow_nn":
            import tensorflow as tf

            model = tf.keras.models.load_model(os.path.join(model_dir, "model.keras"))
        else:
            model = joblib.load(os.path.join(model_dir, "model.joblib"))

        # Random Forest never uses a scaler (see train_model.py's
        # train_horizon(), which sets scaler=None for RF) -- but a model
        # directory can still contain a STALE scaler.joblib left over from
        # an earlier training run where the best model for this horizon was
        # Ridge/TensorFlow instead. save_model_artifacts() only overwrites
        # model.joblib/metadata.json on a rerun, it never deletes leftover
        # files from a previous run's different model type, so that stale
        # scaler (fit on a since-changed feature set) can still be sitting
        # there. Only load it for model types that actually need one.
        scaler = None
        if model_type != "random_forest":
            scaler_path = os.path.join(model_dir, "scaler.joblib")
            if os.path.isfile(scaler_path):
                scaler = joblib.load(scaler_path)

        models[h] = {
            "model": model,
            "model_type": model_type,
            "scaler": scaler,
            "feature_columns": metadata["feature_columns"],
            "metrics": metadata["metrics"],
            "version": hw_model.version,
        }
    return models


def predict_horizon(model_info, feature_row):
    cols = model_info["feature_columns"]
    missing = [c for c in cols if c not in feature_row.index or pd.isna(feature_row[c])]
    if missing:
        raise ValueError(f"Missing/NaN required features for prediction: {missing}")

    X = feature_row[cols].to_numpy(dtype=float).reshape(1, -1)
    if model_info["scaler"] is not None:
        X = model_info["scaler"].transform(X)

    if model_info["model_type"] == "tensorflow_nn":
        pred = float(model_info["model"].predict(X, verbose=0).flatten()[0])
    else:
        pred = float(model_info["model"].predict(X)[0])
    return pred


# ---------------------------------------------------------------------------
# SHAP (cached -- TreeExplainer setup + value computation is the slow part)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Computing SHAP feature importance...")
def compute_shap(_models, _engineered_df, _current_row, current_row_timestamp):
    """TreeExplainer against Random Forest models only -- exact and fast for
    tree ensembles, unlike a model-agnostic explainer. All 3 horizons
    currently register a Random Forest as the best model (see each
    trained_models/aqi_forecast_*h/metadata.json), so this covers every
    horizon today; a horizon that later registers a Ridge/TensorFlow model
    instead is simply skipped here (SHAP's TreeExplainer doesn't apply, and
    a slower KernelExplainer isn't worth the wait for a dashboard).

    `current_row_timestamp` (the latest valid row's index, a plain
    timestamp) is passed as an explicitly hashable arg purely so
    st.cache_resource invalidates this cache whenever a new hourly reading
    arrives -- `_current_row`/`_models`/`_engineered_df` are prefixed with
    an underscore precisely so Streamlit skips hashing the unhashable
    objects themselves, but that means the cache would otherwise never
    know the input actually changed.

    The aggregate (background-sample) SHAP values and the single-row
    "current forecast" SHAP values are computed as two SEPARATE calls to
    explainer.shap_values() -- deliberately, so the per-row waterfall chart
    can never end up reusing (or being derived from) the aggregate
    mean(|SHAP value|) array used for the summary bar chart above it."""
    results = {}
    for h, info in _models.items():
        if info["model_type"] != "random_forest":
            continue
        cols = info["feature_columns"]
        explainer = shap.TreeExplainer(info["model"])

        # --- Aggregate background sample: summary bar chart only ---
        background_sample = _engineered_df[cols].dropna().tail(500)
        summary_shap_values = explainer.shap_values(background_sample)

        # --- Single-row explanation for the CURRENT forecast: waterfall only ---
        # `cols` (== feature_columns from metadata.json, produced by
        # train_model.get_feature_columns()) fixes the exact column order
        # the model was trained on; selecting _current_row[cols] guarantees
        # this row is built in that same order before it ever reaches SHAP.
        current_row_df = _current_row[cols].to_frame().T.astype(float)
        current_shap_values = explainer.shap_values(current_row_df)[0]

        # Sanity check: expected_value + sum(this row's shap values) should
        # reconstruct the model's own prediction for this exact row. If it
        # doesn't, the feature order fed to SHAP has drifted from the order
        # the model was actually trained on -- surfaced as a warning in the
        # UI rather than silently trusting a possibly-misaligned chart.
        model_pred = float(info["model"].predict(current_row_df.to_numpy())[0])
        reconstructed = float(explainer.expected_value) + float(np.sum(current_shap_values))
        additivity_ok = abs(model_pred - reconstructed) <= max(1.0, 0.02 * abs(model_pred))

        results[h] = {
            "feature_columns": cols,
            "background_sample": background_sample,
            "summary_shap_values": summary_shap_values,
            "current_shap_values": current_shap_values,
            "expected_value": float(explainer.expected_value),
            "model_prediction": model_pred,
            "reconstructed_prediction": reconstructed,
            "additivity_ok": additivity_ok,
        }
    return results


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def category_badge_html(category, aqi_value=None):
    color = CATEGORY_COLORS.get(category, "#888888")
    text_color = "#000000" if CATEGORY_SEVERITY.get(category, 0) <= 2 else "#ffffff"
    value_str = f"{aqi_value:.0f} &middot; " if aqi_value is not None else ""
    return (
        f'<span style="background-color:{color};color:{text_color};'
        f'padding:0.15rem 0.6rem;border-radius:0.4rem;font-weight:600;">'
        f"{value_str}{category}</span>"
    )


def render_alert_banner(alert_result):
    if alert_result["triggered"]:
        color = CATEGORY_COLORS.get(alert_result["worst_category"], "#ff0000")
        text_color = "#000000" if CATEGORY_SEVERITY.get(alert_result["worst_category"], 0) <= 2 else "#ffffff"
        alerting_labels = [e["label"] for e in alert_result["entries"] if e["is_alert"]]
        st.markdown(
            f"""
            <div style="background-color:{color};color:{text_color};
                        padding:1rem 1.2rem;border-radius:0.5rem;margin-bottom:1rem;">
              <strong>&#9888; Hazardous Air Quality Alert</strong><br/>
              Worst condition: <strong>{alert_result['worst_category']}</strong>
              (triggered by: {", ".join(alerting_labels)})
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # st.success() already renders its own leading icon (set via `icon`
        # below) -- do NOT also prefix the message text with a checkmark
        # character, or it renders twice.
        st.success(
            "No hazardous conditions: current AQI and all 24h/48h/72h "
            "forecasts are below the Unhealthy threshold.",
            icon="✅",
        )


def render_current_conditions(latest_row):
    st.subheader("Current Conditions")
    aqi_val = latest_row["aqi"]
    category = latest_row.get("aqi_category") or categorize(aqi_val)
    dominant = latest_row.get("aqi_dominant_pollutant", "unknown")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Current AQI", f"{aqi_val:.0f}")
    with c2:
        st.markdown("**Category**")
        st.markdown(category_badge_html(category), unsafe_allow_html=True)
    with c3:
        st.metric("Dominant Pollutant", str(dominant).upper())

    st.caption(f"As of {latest_row.name} (UTC)")

    st.markdown("**Pollutant breakdown (current reading, ug/m3)**")
    st.caption(
        "These are instantaneous readings for this hour. The AQI/category/"
        "dominant pollutant above instead use EPA rolling-window averages "
        "(24h for PM2.5/PM10, 8h for O3/CO -- see compute_true_aqi.py), so "
        "the labeled dominant pollutant won't always match whichever value "
        "looks highest below -- that's expected, not a bug."
    )
    pollutants = ["pm2_5", "pm10", "o3", "no2", "so2", "co"]
    cols = st.columns(len(pollutants))
    for col, pol in zip(cols, pollutants):
        val = latest_row.get(pol)
        with col:
            st.metric(pol.replace("_", ".").upper(), f"{val:.1f}" if pd.notna(val) else "n/a")


def render_forecast_cards(predictions, model_r2_lookup):
    st.subheader("Forecast: Next 3 Days")
    cols = st.columns(3)
    for col, h in zip(cols, HORIZONS):
        pred = predictions.get(h)
        label, color = CONFIDENCE_LABELS[h]
        with col:
            if pred is None:
                st.error(f"{h}h forecast unavailable (see error above).")
                continue
            category = categorize(pred)
            st.markdown(
                f"""
                <div style="border:2px solid {color};border-radius:0.6rem;
                            padding:0.9rem;text-align:center;">
                  <div style="font-size:0.85rem;color:{color};font-weight:600;">
                    {h}h AHEAD &middot; {label}
                  </div>
                  <div style="font-size:2.2rem;font-weight:700;">{pred:.0f}</div>
                  <div>{category_badge_html(category)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            r2 = model_r2_lookup.get(h)
            st.caption(f"Model test R2 = {r2:.2f}" if r2 is not None else "Model test R2 unavailable")


def render_timeseries_chart(engineered_df, latest_row, predictions):
    st.subheader("AQI Trend: History + Forecast")
    history = engineered_df[engineered_df["aqi"].notna()].copy()
    cutoff = latest_row.name - timedelta(days=CHART_HISTORY_DAYS)
    history = history[history.index >= cutoff]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["aqi"],
            mode="lines",
            name="Historical AQI",
            line=dict(color="#3366cc"),
        )
    )

    forecast_x = [latest_row.name] + [latest_row.name + timedelta(hours=h) for h in HORIZONS]
    forecast_y = [latest_row["aqi"]] + [predictions.get(h) for h in HORIZONS]
    if all(y is not None for y in forecast_y):
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#dc3912", dash="dash"),
                marker=dict(size=9),
            )
        )

    fig.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="AQI",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_shap_tab(shap_results):
    st.subheader("Model Insights: SHAP Feature Importance")
    st.caption(
        "SHAP values computed with TreeExplainer against each horizon's "
        "Random Forest model (exact for tree ensembles, no approximation)."
    )

    if not shap_results:
        st.warning("No Random Forest models available for SHAP explanation.")
        return

    tabs = st.tabs([f"{h}h horizon" for h in shap_results.keys()])
    for tab, h in zip(tabs, shap_results.keys()):
        with tab:
            result = shap_results[h]
            cols = result["feature_columns"]

            mean_abs = np.abs(result["summary_shap_values"]).mean(axis=0)
            importance = (
                pd.Series(mean_abs, index=cols).sort_values(ascending=False).head(15)
            )

            st.markdown(f"**Top features driving the {h}h forecast (mean |SHAP value|)**")
            bar_fig = go.Figure(
                go.Bar(
                    x=importance.values[::-1],
                    y=importance.index[::-1],
                    orientation="h",
                    marker=dict(color="#3366cc"),
                )
            )
            bar_fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(bar_fig, use_container_width=True, key=f"shap_bar_{h}")

            st.markdown(f"**Why today's {h}h forecast is what it is**")
            if not result["additivity_ok"]:
                st.warning(
                    f"SHAP additivity check failed for the {h}h model: "
                    f"expected_value + sum(shap values) = "
                    f"{result['reconstructed_prediction']:.1f}, but the model's "
                    f"actual prediction for this row is {result['model_prediction']:.1f}. "
                    f"This usually means the feature order fed to SHAP has drifted "
                    f"from train_model.py's get_feature_columns() order -- treat "
                    f"this chart with caution until that's resolved."
                )

            # Per-instance, signed SHAP values for the CURRENT row only --
            # a genuine mix of positive (pushes this forecast up) and
            # negative (pushes it down) contributions, never derived from
            # the aggregate mean(|SHAP value|) array used above.
            row_contrib = (
                pd.Series(result["current_shap_values"], index=cols)
                .sort_values(key=np.abs, ascending=True)
                .tail(12)
            )
            waterfall_fig = go.Figure(
                go.Bar(
                    x=row_contrib.values,
                    y=row_contrib.index,
                    orientation="h",
                    marker=dict(
                        color=["#dc3912" if v > 0 else "#3366cc" for v in row_contrib.values]
                    ),
                )
            )
            waterfall_fig.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="SHAP value (red = pushes AQI up, blue = pushes AQI down)",
            )
            st.plotly_chart(waterfall_fig, use_container_width=True, key=f"shap_row_{h}")
            st.caption(
                f"Base value (average model output) = {result['expected_value']:.1f} "
                f"-> this row's predicted AQI = {result['model_prediction']:.1f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("Karachi AQI Forecasting Dashboard")
    st.caption(
        "24h / 48h / 72h AQI forecasts for Karachi, Pakistan -- Random Forest "
        "models trained on lagged/rolling AQI + pollutant features, served "
        "from the Hopsworks Model Registry."
    )

    top_l, top_r = st.columns([5, 1])
    with top_r:
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    try:
        project = get_project()
        raw_df = load_raw_features(project)
        engineered_df = build_engineered_features(raw_df)
        latest_row = get_latest_valid_row(engineered_df)
        models = load_models(project)
    except Exception as e:
        st.error(
            f"Could not load data/models from Hopsworks: {type(e).__name__}: {e}\n\n"
            "Check that HOPSWORKS_API_KEY is set and valid, that the project "
            "is reachable, and that the feature group / registered models "
            "exist under the expected names and versions."
        )
        st.stop()
        return

    model_r2_lookup = {h: models[h]["metrics"]["r2"] for h in HORIZONS if h in models}

    predictions = {}
    pred_errors = {}
    for h in HORIZONS:
        try:
            predictions[h] = predict_horizon(models[h], latest_row)
        except Exception as e:
            predictions[h] = None
            pred_errors[h] = str(e)

    for h, err in pred_errors.items():
        st.warning(f"{h}h prediction failed: {err}")

    alert_readings = {"current": latest_row["aqi"]}
    alert_readings.update({f"{h}h": predictions.get(h) for h in HORIZONS})
    alert_result = check_alerts(alert_readings)
    render_alert_banner(alert_result)

    tab_forecast, tab_insights = st.tabs(["Forecast Dashboard", "Model Insights"])

    with tab_forecast:
        render_current_conditions(latest_row)
        st.divider()
        render_forecast_cards(predictions, model_r2_lookup)
        st.divider()
        render_timeseries_chart(engineered_df, latest_row, predictions)
        st.caption(
            "Model performance (held-out test set, chronological split): "
            "24h RMSE=11.79 (R2=0.71) -- 48h RMSE=17.40 (R2=0.35) -- "
            "72h RMSE=20.49 (R2=0.08, a real but weak improvement over naive "
            "persistence). Treat the 72h forecast as directional only."
        )

    with tab_insights:
        shap_results = compute_shap(models, engineered_df, latest_row, latest_row.name)
        render_shap_tab(shap_results)


if __name__ == "__main__":
    main()
