"""
ingest_to_hopsworks.py

Pushes the CORRECTED, feature-complete AQI dataset
(aqi_historical_backfill_v2.csv, output of compute_true_aqi.py) into a
Hopsworks Feature Group.

REVISION HISTORY:
  v1 -> v2: previously pointed at aqi_historical_backfill.csv (the RAW
    backfill output), which does not have the continuous 'aqi' target,
    'aqi_category', or 'aqi_dominant_pollutant' columns -- those only exist
    after compute_true_aqi.py runs. Also: DEBUG_ROW_LIMIT was left at 500
    (would only push a fraction of the real data), and wait_for_job had
    reverted to True, which is the setting that caused the known Hopsworks
    free-tier flaky-polling false-failure (JobExecutionException even
    though the write succeeds) documented from the original v1 ingest.
  v2 -> v3: the v2 push carried two data bugs later found by
    thorough_eda_v2.py on the extended backfill -- duplicate-timestamp
    rows from chunk-boundary overlap in backfill_historical.py, and -9999
    sentinel values leaking into pm10/no2/o3 as if they were real
    readings. Both are now fixed upstream in compute_true_aqi.py, so v3
    is a clean re-push, not a schema change. Also: the v2 push's
    FEATURE_GROUP_DESCRIPTION was 442 characters and hit Hopsworks'
    256-character limit on the first attempt (error code 270092) --
    oddly, an identical retry succeeded, which is unexplained and not
    something to rely on. Description is shortened well under the limit
    here, with an assertion that fails before any API call if it's ever
    too long again.
  v3 -> v4: backfill_historical.py's BACKFILL_MONTHS was deliberately
    reverted from 200 (~5.7yr, back to Nov 2020) to 24 (the mentor's
    "2yr ideal"), by explicit choice -- see that script's revision notes
    for the accepted tradeoff. v3's 49,222 rows spanning 2020-2026 and
    v4's 16,752 rows spanning Sept 2024-Aug 2026 are materially different
    datasets (different date range, ~1/3 the rows), not just a bugfix
    re-push, so this gets its own version rather than reusing v3 -- v3's
    Feature Group and any models trained from it stay queryable in
    Hopsworks for comparison rather than being silently overwritten.

Feature group version bumps whenever the schema or the underlying data
changes materially -- Hopsworks feature groups are schema/lineage-locked
per version, so reusing an old version number for meaningfully different
data risks a schema-mismatch error or silently mixing clean and
contaminated rows under the same version.

Run this inside your Codespace (or any Linux environment) where `hopsworks`
installs cleanly:
    pip install hopsworks pandas requests
    python ingest_to_hopsworks.py

Requires HOPSWORKS_API_KEY set as an environment variable / Codespaces secret.
"""

import os
import sys
import pandas as pd
import hopsworks

CSV_PATH = "aqi_historical_backfill_v2.csv"

FEATURE_GROUP_NAME = "aqi_karachi_features"
FEATURE_GROUP_VERSION = 4
FEATURE_GROUP_DESCRIPTION = (
    "Hourly Karachi AQI features (OpenWeather + Open-Meteo cross-check). "
    "v4: continuous EPA-formula AQI target, deduped/sentinel-cleaned, "
    "~2yr history from Sept 2024 (reverted from v3's ~5.7yr pull)."
)
assert len(FEATURE_GROUP_DESCRIPTION) <= 256, (
    f"FEATURE_GROUP_DESCRIPTION is {len(FEATURE_GROUP_DESCRIPTION)} chars, "
    f"over Hopsworks' 256-char limit -- shorten it before running."
)
def load_and_prepare_data(csv_path):
    """Load the CSV and do the light cleanup Hopsworks needs:
    - a unique primary key column
    - the event-time column in proper datetime format
    """
    if not os.path.isfile(csv_path):
        print(f"ERROR: {csv_path} not found in the current directory.")
        print("Make sure compute_true_aqi.py has been run (it produces this")
        print("file from aqi_historical_backfill.csv), and that you're running")
        print("this from the folder containing it, or update CSV_PATH above.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {csv_path}")

    if "aqi" not in df.columns:
        print(
            "ERROR: no 'aqi' column found in this CSV. This script expects "
            "the output of compute_true_aqi.py (aqi_historical_backfill_v2.csv), "
            "not the raw backfill file. Run compute_true_aqi.py first."
        )
        sys.exit(1)

    # Hopsworks feature groups need a primary key. timestamp_utc is unique
    # per row (hourly data), so we derive a clean integer id from it.
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df["row_id"] = df.index

    # Hopsworks is picky about column names: lowercase, no special chars.
    # Our columns are already clean (underscores only), but this guards
    # against anything sneaking in from the CSV header.
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Replace NaN with None so Hopsworks handles missing values correctly
    # rather than writing the literal string "NaN".
    df = df.where(pd.notnull(df), None)

    return df


def main():
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        print("ERROR: HOPSWORKS_API_KEY environment variable not set.")
        print("Set it as a Codespaces secret, or run: export HOPSWORKS_API_KEY='your_key'")
        sys.exit(1)

    df = load_and_prepare_data(CSV_PATH)

    # Set to an integer (e.g. 500) to test with a small subset first and
    # isolate whether failures are data-related or scale-related. None
    # inserts the full dataset -- that's what you want for the real run.
    DEBUG_ROW_LIMIT = None
    if DEBUG_ROW_LIMIT:
        print(f"\n[DEBUG MODE] Testing with only the first {DEBUG_ROW_LIMIT} rows.")
        df = df.head(DEBUG_ROW_LIMIT)

    print("\nConnecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    print(f"Connected to project: {project.name}")

    print(f"\nCreating/retrieving feature group: {FEATURE_GROUP_NAME} (v{FEATURE_GROUP_VERSION})...")
    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description=FEATURE_GROUP_DESCRIPTION,
        primary_key=["row_id"],
        event_time="timestamp_utc",
        online_enabled=False,  # offline-only is fine for training data; flip on later if you need low-latency serving
        time_travel_format="HUDI",  # avoids requiring the optional deltalake library
    )

    print(f"\nInserting {len(df)} rows into the feature group...")
    print("(This may take a while longer than the v1 insert -- the extended")
    print(" backfill has ~3-4x more rows. Hopsworks free-tier job-status")
    print(" polling is known to be flaky and can throw JobExecutionException")
    print(" even when the underlying write succeeds -- that's why")
    print(" wait_for_job is set to False below. If this prints a warning or")
    print(" looks like it errored, check the Feature Group's row count in")
    print(" the Hopsworks UI before assuming the insert failed.")
    feature_group.insert(df, write_options={"wait_for_job": False})

    print("\nInsert submitted. Verify the row count in the Hopsworks UI once")
    print("the background job finishes (usually a few minutes for this size).")
    print(f"View it at: https://app.hopsworks.ai (Project: {project.name} -> Feature Store -> {FEATURE_GROUP_NAME})")


if __name__ == "__main__":
    main()