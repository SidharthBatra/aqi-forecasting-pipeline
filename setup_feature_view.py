"""
setup_feature_view.py

Adds proper Hopsworks Feature Store lineage on top of the existing
pipeline. Today, train_model.py pulls the raw feature group into pandas,
does all feature engineering locally, and registers models with the
plain mr.python.create_model() call -- which has no link back to any
Feature Group. That's why the Hopsworks project overview shows 0 for
"Feature Views", "Used to train Models", "Models Using Features": those
counters specifically track models registered THROUGH a Feature View +
Training Dataset, which this project has never created. It's not a bug,
just an unused (and more idiomatic) part of Hopsworks' Feature Store.

This script builds the missing pieces:
  1. Runs the SAME feature engineering train_model.py already uses
     (imported directly, not reimplemented) to build the full 32-feature
     + 3-target engineered dataset.
  2. Pushes that engineered dataset to Hopsworks as its own feature group
     (aqi_karachi_engineered_features) -- separate from the raw
     aqi_karachi_features group, since it holds derived (lag/rolling/
     cyclical) columns, not raw pollutant readings.
  3. Builds a Feature View on top of it, with the 3 horizon targets
     (aqi_target_24h/48h/72h) as labels.
  4. Creates ONE Training Dataset from that Feature View.

Deliberately NOT done: recomputing lag/rolling features as Hopsworks
on-demand transformations. Hopsworks' Feature View query layer is built
for joining/selecting across feature groups, not for gap-aware
time-based rolling windows -- forcing that in would mean re-deriving
logic that's already correct and tested in train_model.py. Also
deliberately NOT done: training directly on the split Hopsworks'
create_train_test_split() produces. It doesn't know about this project's
chronological-split-per-horizon requirement (each horizon drops a
different tail of rows depending on which target has enough future data),
so train_model.py keeps doing that split locally -- this Training
Dataset exists so registered models have a real Feature Store object to
link to, not to replace the actual training data path.

NOTE: hsml (Hopsworks' Python client) API details vary by version. This
follows the standard documented Feature View / Training Dataset pattern,
but hasn't been run against a live Hopsworks project from this
environment -- if a method name below doesn't match your installed
`hopsworks`/`hsml` version, that's a real possibility, not something to
assume is a typo in your data.

Run once (re-run only if the feature engineering itself changes):
    python setup_feature_view.py
"""

import os
import sys

import pandas as pd

from train_model import (
    INPUT_CSV,
    TIMESTAMP_COL,
    TARGET_COL,
    HORIZONS,
    load_and_prepare_grid,
    add_time_features,
    add_lag_and_rolling_features,
    add_targets,
    get_feature_columns,
)

ENGINEERED_FG_NAME = "aqi_karachi_engineered_features"
ENGINEERED_FG_VERSION = 1
FEATURE_VIEW_NAME = "aqi_forecast_fv"
FEATURE_VIEW_VERSION = 1


def build_engineered_dataframe():
    """Identical pipeline to train_model.py's main() up to (not including)
    the per-horizon train/test split -- this IS the data every horizon's
    model is actually trained from."""
    df = load_and_prepare_grid(INPUT_CSV)
    df = add_time_features(df)
    df = add_lag_and_rolling_features(df)
    df = add_targets(df)

    # Drop pure gap-placeholder rows (load_and_prepare_grid's hourly
    # reindex fills real gaps with all-NaN rows) -- these were never real
    # hours and have nothing to contribute to the Feature Store. Per-
    # horizon NaN filtering on features/that horizon's specific target
    # still happens downstream in train_model.py exactly as before; this
    # only removes rows with NO reading at all.
    before = len(df)
    df = df[df[TARGET_COL].notna()].copy()
    print(f"Dropped {before - len(df)} pure gap-placeholder rows (no real reading at all).")

    df = df.reset_index()  # TIMESTAMP_COL becomes a column again
    df["row_id"] = df.index
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.where(pd.notnull(df), None)
    return df


def main():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        print("ERROR: HOPSWORKS_API_KEY environment variable not set.")
        sys.exit(1)

    import hopsworks

    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    print(f"Connected to project: {project.name}")

    df = build_engineered_dataframe()
    feature_cols = get_feature_columns(df)
    label_cols = [f"aqi_target_{h}h" for h in HORIZONS]
    print(
        f"\nEngineered dataset: {len(df)} rows, {len(feature_cols)} feature "
        f"columns, {len(label_cols)} label columns"
    )

    print(f"\nCreating/retrieving engineered feature group: {ENGINEERED_FG_NAME} (v{ENGINEERED_FG_VERSION})...")
    engineered_fg = fs.get_or_create_feature_group(
        name=ENGINEERED_FG_NAME,
        version=ENGINEERED_FG_VERSION,
        description=(
            "Fully engineered AQI forecasting features (lags, rolling "
            "stats, cyclical time encodings) + 3 shifted targets, computed "
        ),
        primary_key=["row_id"],
        event_time=TIMESTAMP_COL,
        online_enabled=False,
        time_travel_format="HUDI",
    )
    print(f"Inserting {len(df)} rows (this may take a few minutes to materialize)...")
    engineered_fg.insert(df, write_options={"wait_for_job": False})
    print("Insert submitted.")

    print(f"\nCreating/retrieving feature view: {FEATURE_VIEW_NAME} (v{FEATURE_VIEW_VERSION})...")
    query = engineered_fg.select(feature_cols + label_cols)
    feature_view = fs.get_or_create_feature_view(
        name=FEATURE_VIEW_NAME,
        version=FEATURE_VIEW_VERSION,
        query=query,
        labels=label_cols,
        description=(
            "Feature view over the engineered AQI feature set; labels are "
            "the 3 horizon targets (aqi_target_24h/48h/72h)."
        ),
    )
    print(f"Feature view ready: {FEATURE_VIEW_NAME} v{FEATURE_VIEW_VERSION}")

    print("\nCreating one Training Dataset from the feature view...")
    td_version, td_job = feature_view.create_train_test_split(
        test_size=0.2,
        description=(
            "Reference training dataset for lineage purposes. NOT the "
            "actual chronological split used for training -- see "
            "train_model.py's chronological_split() for that."
        ),
    )
    print(f"Training dataset created: version {td_version}")
    print(
        f"\nIMPORTANT: open train_model.py and set "
        f"TRAINING_DATASET_VERSION = {td_version} (it currently defaults "
        f"to 1 -- update it if this printed a different number) so "
        f"register_in_hopsworks() links new models to this dataset."
    )

    print(
        "\nDone. Re-run train_model.py -- register_in_hopsworks() will now "
        "attempt to attach this feature view + training dataset to each "
        "registered model. If your hsml version's create_model() doesn't "
        "accept those kwargs, it'll print a note and fall back to plain "
        "registration rather than failing the run."
    )


if __name__ == "__main__":
    main()