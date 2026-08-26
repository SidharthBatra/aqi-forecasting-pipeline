"""
aqi_alerts.py

Reusable EPA-category-based hazard alert logic for AQI values (current
reading and/or forecasts). Kept as a standalone module -- not embedded
directly in dashboard.py's UI code -- so the same threshold logic can be
reused later for other notification channels (email/Slack) without
duplicating the EPA category definitions.

Threshold policy: an alert triggers when any reading is "Unhealthy" (EPA
AQI > 150) or worse, i.e. the point at which EPA guidance starts advising
the general public (not just sensitive groups) to reduce exposure.
"""

from compute_true_aqi import categorize

# EPA category -> severity rank (higher = worse). Mirrors compute_true_aqi's
# categorize() thresholds exactly -- categorize() is the single source of
# truth for what AQI value maps to what category; this just orders them.
CATEGORY_SEVERITY = {
    "Good": 0,
    "Moderate": 1,
    "Unhealthy for Sensitive Groups": 2,
    "Unhealthy": 3,
    "Very Unhealthy": 4,
    "Hazardous": 5,
}

# Standard AQI category colors (matches AirNow/AQICN conventions), used for
# alert banners and category badges.
CATEGORY_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
}

# Readings at or above this severity rank ("Unhealthy" or worse) trigger a
# hazard alert.
ALERT_SEVERITY_THRESHOLD = CATEGORY_SEVERITY["Unhealthy"]


def get_category(aqi):
    """AQI value -> EPA category label. Delegates to compute_true_aqi's
    categorize() so alert thresholds never drift out of sync with how the
    'aqi' target/feature itself is labeled elsewhere in this pipeline."""
    if aqi is None:
        return None
    return categorize(aqi)


def get_severity(category):
    """EPA category label -> severity rank (0=Good ... 5=Hazardous).
    Unrecognized/None categories default to 0 (treated as non-alerting)
    rather than raising, since this may be called on a value that failed
    to compute (missing pollutant data) rather than a real low reading."""
    return CATEGORY_SEVERITY.get(category, 0)


def check_alerts(readings):
    """Evaluate a set of AQI readings (current + any forecast horizons)
    against the hazard threshold.

    Args:
        readings: dict mapping a label (e.g. "current", "24h", "48h",
            "72h") to an AQI value (float) or None (None entries are
            skipped -- e.g. a horizon whose model failed to load).

    Returns:
        dict with:
          triggered      -- bool, True if ANY reading is Unhealthy or worse
          max_severity    -- int, the worst severity rank seen (-1 if no
                              valid readings at all)
          worst_label     -- label of the worst reading, or None
          worst_category  -- category of the worst reading, or None
          entries         -- list of {label, aqi, category, severity,
                              is_alert} for every valid input reading, in
                              the order given
    """
    entries = []
    max_severity = -1
    worst_label = None
    worst_category = None

    for label, aqi in readings.items():
        if aqi is None:
            continue
        category = get_category(aqi)
        severity = get_severity(category)
        entries.append(
            {
                "label": label,
                "aqi": aqi,
                "category": category,
                "severity": severity,
                "is_alert": severity >= ALERT_SEVERITY_THRESHOLD,
            }
        )
        if severity > max_severity:
            max_severity = severity
            worst_label = label
            worst_category = category

    return {
        "triggered": max_severity >= ALERT_SEVERITY_THRESHOLD,
        "max_severity": max_severity,
        "worst_label": worst_label,
        "worst_category": worst_category,
        "entries": entries,
    }
