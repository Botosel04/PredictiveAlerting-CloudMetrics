"""
features.py
-----------
Feature engineering for predictive alerting on CloudWatch CPU metrics.

All features are computed purely from historical data (no look-ahead),
making them safe to use in a real-time inference pipeline.

Feature groups:
  - Statistical deviation: z_score, median_deviation, clean_deviation
  - Momentum:              value_diff1 (velocity), value_diff2 (acceleration)
  - Trend:                 ema_fast, ema_slow, ema_cross
  - Instability:           rolling_std_24, rolling_std_48
  - Long-run drift:        long_trend (is the 4-hour mean drifting?)
  - Frequency:             wavelet_jitter (Haar detail coefficient)
  - Temporal:              hour_sin, hour_cos (cyclical time encoding)
  - Raw level:             value_norm (absolute CPU matters — near-zero is a regime)
"""

import numpy as np
import pandas as pd
import pywt


def engineer_features(df: pd.DataFrame, rolling_window_size: int = 12) -> pd.DataFrame:
    """
    Add engineered features to a labeled CloudWatch dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_and_label_file — must have 'timestamp' and 'value' columns.
    rolling_window_size : int
        Short rolling window in steps (default 12 = 60 min at 5-min granularity).

    Returns
    -------
    pd.DataFrame with all original columns + feature columns.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Short-window statistical deviation
    rolling_mean = df["value"].rolling(window=rolling_window_size, min_periods=1).mean()
    rolling_std = (
        df["value"]
        .rolling(window=rolling_window_size, min_periods=1)
        .std()
        .replace(0, 1e-4)
        .fillna(1e-4)
    )
    df["z_score"] = (df["value"] - rolling_mean) / rolling_std

    rolling_median = df["value"].rolling(window=rolling_window_size, min_periods=1).median()
    df["median_deviation"] = df["value"] - rolling_median

    # Suppress tiny deviations
    chomp_threshold = rolling_median * 0.05
    df["clean_deviation"] = np.where(
        np.abs(df["median_deviation"]) < chomp_threshold, 0.0, df["median_deviation"]
    )

    # Momentum (velocity and acceleration)
    df["value_diff1"] = df["value"].diff().fillna(0)
    df["value_diff2"] = df["value_diff1"].diff().fillna(0)

    #  EMA crossover
    df["ema_fast"] = df["value"].ewm(span=5, adjust=False).mean()
    df["ema_slow"] = df["value"].ewm(span=20, adjust=False).mean()
    df["ema_cross"] = df["ema_fast"] - df["ema_slow"]

    # 2-hour window:
    df["rolling_std_24"] = (
        df["value"].rolling(window=24, min_periods=1).std().fillna(0)
    )
    # 4-hour window
    df["rolling_std_48"] = (
        df["value"].rolling(window=48, min_periods=1).std().fillna(0)
    )

    # 4-hour mean drift
    rolling_mean_48 = df["value"].rolling(window=48, min_periods=12).mean()
    df["long_trend"] = rolling_mean_48.diff(12).fillna(0)  # change over last 12 steps

    # Haar wavelet jitter (high-frequency noise) ────────────────────────────
    def get_haar_detail(window_data: np.ndarray) -> float:
        if len(window_data) < 2:
            return 0.0
        _, cD = pywt.dwt(window_data, "haar")
        return float(np.max(np.abs(cD)))

    df["wavelet_jitter"] = (
        df["value"]
        .rolling(window=4, min_periods=2)
        .apply(get_haar_detail, raw=True)
        .fillna(0)
    )

    # Cyclical time encoding
    time_in_hours = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * time_in_hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * time_in_hours / 24.0)

    df["value_norm"] = df["value"]

    return df


FEATURE_COLS = [
    "z_score",
    # "clean_deviation",
    "wavelet_jitter",
    "hour_sin",
    "hour_cos",
    # "value_diff1",
    # "value_diff2",
    "ema_cross",
    "value_norm",
    "rolling_std_24",
    "rolling_std_48",
    "long_trend",
]


def create_sliding_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    W: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    df_clean = df[df["exclude"] == 0].reset_index(drop=True)

    X, y = [], []
    for i in range(W, len(df_clean)):
        window_data = df_clean.loc[i - W : i - 1, feature_cols].values
        if window_data.shape[0] < W:
            continue
        X.append(window_data.flatten())
        y.append(df_clean.loc[i, "is_incident"])

    return np.array(X), np.array(y)
