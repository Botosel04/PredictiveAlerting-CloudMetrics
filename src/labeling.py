"""
labeling.py
-----------
Loads a CloudWatch CSV and assigns labels for predictive alerting.

Label logic (per anomaly window):
  - Positive (is_incident=1): window_start → exact crash timestamp
    This is the pre-incident warning zone the model learns to detect.
  - Excluded (exclude=1):     crash timestamp → window_end
    Chaotic crash/recovery data that would confuse the model.
  - Negative (is_incident=0): everything else — normal operation.

Why this matters:
  Labeling the full window as positive conflates "about to crash"
  with "currently crashing", which are different signal patterns.
  The model needs to learn the former, not the latter.
"""

import os
import json
import pandas as pd


def load_and_label_file(
    csv_path: str,
    windows_json_path: str,
    labels_json_path: str,
) -> pd.DataFrame:
    """
    Load a CloudWatch CSV and label each timestep for predictive alerting.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with 'timestamp' and 'value' columns.
    windows_json_path : str
        Path to combined_windows.json — provides [window_start, window_end] per file.
    labels_json_path : str
        Path to combined_labels.json — provides exact crash timestamps per file.

    Returns
    -------
    pd.DataFrame with columns: timestamp, value, is_incident, exclude
    """
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["is_incident"] = 0
    df["exclude"] = 0

    file_key_suffix = os.path.basename(csv_path)

    with open(windows_json_path) as f:
        windows_dict = json.load(f)
    with open(labels_json_path) as f:
        labels_dict = json.load(f)

    anomaly_windows = []
    crash_times = []

    for key in windows_dict:
        if file_key_suffix in key:
            anomaly_windows = windows_dict[key]
            crash_times = [pd.to_datetime(t) for t in labels_dict.get(key, [])]
            break

    if not anomaly_windows:
        # No anomalies for this file — all negatives, nothing to exclude
        return df

    for i, window in enumerate(anomaly_windows):
        w_start = pd.to_datetime(window[0])
        w_end = pd.to_datetime(window[1])

        # Use exact crash time if available, otherwise fall back to window midpoint
        if i < len(crash_times):
            crash = crash_times[i]
        else:
            crash = w_start + (w_end - w_start) / 2

        # --- Positive zone: window_start → crash (pre-incident warning) ---
        mask_pos = (df["timestamp"] >= w_start) & (df["timestamp"] < crash)
        df.loc[mask_pos, "is_incident"] = 1

        # --- Exclusion zone: crash → window_end
        exclude_end = w_end
        mask_excl = (df["timestamp"] >= crash) & (df["timestamp"] <= exclude_end)
        df.loc[mask_excl, "exclude"] = 1

    return df


def get_anomaly_windows(
    csv_path: str,
    windows_json_path: str,
    labels_json_path: str,
) -> list[dict]:
    """
    Return structured anomaly metadata for a file, used during evaluation.

    Returns
    -------
    List of dicts with keys: w_start, crash, w_end
    """
    file_key_suffix = os.path.basename(csv_path)

    with open(windows_json_path) as f:
        windows_dict = json.load(f)
    with open(labels_json_path) as f:
        labels_dict = json.load(f)

    for key in windows_dict:
        if file_key_suffix not in key:
            continue
        windows = windows_dict[key]
        crashes = [pd.to_datetime(t) for t in labels_dict.get(key, [])]
        result = []
        for i, w in enumerate(windows):
            w_start = pd.to_datetime(w[0])
            w_end = pd.to_datetime(w[1])
            crash = crashes[i] if i < len(crashes) else w_start + (w_end - w_start) / 2
            result.append({"w_start": w_start, "crash": crash, "w_end": w_end})
        return result

    return []
