"""
register_existing_model.py

train_model.py already trains and locally saves every horizon's model
before attempting Hopsworks registration -- register_in_hopsworks() just
uploads whatever's sitting in trained_models/aqi_forecast_{H}h/. So when
registration fails for one horizon (as it just did for 48h, on a
Hopsworks-side ClusterJDatastoreException/"Tuple did not exist" during
upload -- a transient backend error, not a bug in our code, consistent
with the free-tier flakiness documented earlier in this project), there's
no need to retrain that horizon from scratch. This script just re-runs
the registration step against the existing local artifacts.

Usage:
    python register_existing_model.py 48
    python register_existing_model.py 24 48 72   # re-register several
"""

import json
import os
import sys

MODEL_OUTPUT_DIR = "trained_models"


def register_one(project, horizon):
    out_dir = os.path.join(MODEL_OUTPUT_DIR, f"aqi_forecast_{horizon}h")
    meta_path = os.path.join(out_dir, "metadata.json")

    if not os.path.isfile(meta_path):
        print(f"  SKIP {horizon}h: no {meta_path} found -- run train_model.py first.")
        return

    with open(meta_path) as f:
        metadata = json.load(f)

    model_name = metadata["model_type"]
    metrics = metadata["metrics"]

    print(f"\n=== Registering aqi_forecast_{horizon}h ({model_name}) from {out_dir} ===")
    print(f"  Metrics on file: RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  R2={metrics['r2']:.4f}")

    mr = project.get_model_registry()
    hw_model = mr.python.create_model(
        name=f"aqi_forecast_{horizon}h",
        metrics=metrics,
        description=(
            f"Best of RF/Ridge/TensorFlow for {horizon}h-ahead AQI forecast "
            f"(model_type={model_name}), trained on lagged/historical features only. "
            f"Registered via register_existing_model.py after the original "
            f"train_model.py run's registration attempt failed on a Hopsworks-side error."
        ),
    )
    hw_model.save(out_dir)
    print(f"  Registered aqi_forecast_{horizon}h (v{hw_model.version}) in Hopsworks Model Registry")


def main():
    horizons = [int(h) for h in sys.argv[1:]]
    if not horizons:
        print("Usage: python register_existing_model.py <horizon> [<horizon> ...]")
        print("Example: python register_existing_model.py 48")
        sys.exit(1)

    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        print("ERROR: HOPSWORKS_API_KEY environment variable not set.")
        sys.exit(1)

    import hopsworks

    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key)
    print(f"Connected to project: {project.name}")

    for horizon in horizons:
        try:
            register_one(project, horizon)
        except Exception as e:
            print(f"  FAILED to register {horizon}h: {type(e).__name__}: {e}")
            print("  If this is the same backend error as before, it's worth waiting a few")
            print("  minutes and retrying -- it's Hopsworks-side flakiness, not a data/code issue.")


if __name__ == "__main__":
    main()