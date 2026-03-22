"""
evaluate.py
-----------
Evaluation functions that match the JetBrains task rubric exactly.

The rubric asks for:
  1. Per-incident recall — for what fraction of incidents did the model
     raise at least one alert before the incident started?
  2. False-positive rate at a reasonable level (reported as FP events/day)
  3. Detection lead time — how early was the first alert before the crash?
  4. Precision-recall trade-off discussion

Key distinction:
  Per-STEP recall (sklearn's classification_report) is misleading here.
  A model that predicts everything as positive gets 100% step-recall but
  is completely useless as an alerting system.

  Per-INCIDENT recall counts each anomaly window once. It answers:
  "Would the on-call engineer have received a warning?"
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score


def per_incident_recall(
    y_probs: np.ndarray,
    timestamps: pd.Series,
    anomaly_meta: list[dict],
    threshold: float,
) -> dict:
    """
    Compute per-incident recall, lead times, and false-positive event count.

    Parameters
    ----------
    y_probs : np.ndarray
        Model probability scores for each timestep in the evaluation set.
    timestamps : pd.Series
        Timestamp for each row in y_probs (same index).
    anomaly_meta : list of dict
        Each dict has keys: 'w_start', 'crash', 'w_end'
    threshold : float
        Probability threshold above which an alert is fired.

    Returns
    -------
    dict with keys:
        recall          — float in [0, 1]
        caught          — int, number of incidents with at least one pre-alert
        total           — int, total number of incidents
        lead_times_min  — list of floats (minutes of lead time per caught incident)
        fp_events       — int, number of false-positive alert clusters
        fp_per_day      — float
    """
    timestamps = pd.to_datetime(timestamps).reset_index(drop=True)
    is_alert = (y_probs >= threshold)

    caught = 0
    total = len(anomaly_meta)
    lead_times = []

    valid_zones = []   # (start, end)
    exclude_zones = [] # crash + recovery zones

    for meta in anomaly_meta:
        w_start = meta["w_start"]
        crash   = meta["crash"]
        w_end   = meta["w_end"]

        valid_zones.append((w_start, crash))
        exclude_zones.append((crash, w_end + pd.Timedelta(minutes=60)))

        mask = (timestamps >= w_start) & (timestamps < crash)
        pre_probs = y_probs[mask.values]
        pre_ts    = timestamps[mask.values]

        if len(pre_probs) > 0 and (pre_probs >= threshold).any():
            caught += 1
            first_idx = int(np.where(pre_probs >= threshold)[0][0])
            lead_min = (crash - pre_ts.iloc[first_idx]).total_seconds() / 60
            lead_times.append(lead_min)

    # Count false-positive alert events (consecutive alert ticks = 1 event)
    fp_mask = is_alert.copy()
    for ts_val, fired in zip(timestamps, fp_mask):
        if not fired:
            continue
        for zs, ze in valid_zones:
            if zs <= ts_val < ze:
                fp_mask[timestamps == ts_val] = False
                break
        for zs, ze in exclude_zones:
            if zs <= ts_val <= ze:
                fp_mask[timestamps == ts_val] = False
                break

    # Group consecutive True values into events
    fp_transitions = np.diff(np.insert(fp_mask.astype(int), 0, 0))
    fp_events = int(np.sum(fp_transitions == 1))

    # Duration of the evaluation period in days
    if len(timestamps) > 1:
        duration_days = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 86400
    else:
        duration_days = 1.0

    return {
        "recall":         caught / total if total > 0 else 0.0,
        "caught":         caught,
        "total":          total,
        "lead_times_min": lead_times,
        "fp_events":      fp_events,
        "fp_per_day":     fp_events / duration_days if duration_days > 0 else float("inf"),
    }


def threshold_sweep(
    y_probs: np.ndarray,
    timestamps: pd.Series,
    anomaly_meta: list[dict],
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Sweep thresholds and return a DataFrame of recall/FP metrics at each.

    Useful for picking the operating point and plotting the trade-off curve.

    Parameters
    ----------
    y_probs, timestamps, anomaly_meta : as in per_incident_recall
    thresholds : array of threshold values to try (default: 50 steps from 0.01 to 0.95)

    Returns
    -------
    pd.DataFrame with columns: threshold, recall, caught, total,
                                fp_events, fp_per_day, median_lead_min
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.95, 50)

    rows = []
    for t in thresholds:
        result = per_incident_recall(y_probs, timestamps, anomaly_meta, threshold=t)
        rows.append({
            "threshold":       round(float(t), 4),
            "recall":          round(result["recall"], 3),
            "caught":          result["caught"],
            "total":           result["total"],
            "fp_events":       result["fp_events"],
            "fp_per_day":      round(result["fp_per_day"], 2),
            "median_lead_min": round(float(np.median(result["lead_times_min"])), 1)
                               if result["lead_times_min"] else None,
        })

    return pd.DataFrame(rows)


def find_operating_point(
    sweep_df: pd.DataFrame,
    min_recall: float = 0.80,
) -> pd.Series | None:
    candidates = sweep_df[sweep_df["recall"] >= min_recall]
    if candidates.empty:
        return None
    # Pick the lowest FP/day among all rows that meet the recall target
    return candidates.sort_values("fp_per_day", ascending=True).iloc[0]