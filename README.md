# Predictive Alerting for CloudWatch Metrics

A predictive alerting system that detects whether an EC2 CPU utilization metric
is currently in a pre-incident warning zone, trained on the
[Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB).

---

## Results

| Metric | Value |
|---|---|
| Per-incident recall | **92%** (11/12 incidents caught) |
| False-positive events/day | **1.26** (aggregated across 8 servers) |
| Median lead time | **450 min** (~7.5 hours before crash) |
| Operating threshold | 0.3745 |
| Evaluation method | Leave-one-server-out cross-validation |

A random baseline achieving the same recall requires threshold=0.95 and
produces 16.58 FP events/day, 13× more false alarms,  confirming the
model has learned a genuine signal rather than firing indiscriminately.

---

## Problem Formulation

**Task**: Given the previous W timesteps of a CPU utilization metric,
predict whether the system is currently in a pre-incident warning zone.

**Sliding-window binary classification**: Each sample is a flattened
`(W × features)` vector. The label is `1` if the current timestep falls
between the anomaly window start and the exact crash timestamp, `0` otherwise.

**Why this labeling boundary matters**: Each NAB anomaly window spans
`[window_start, window_end]` with the exact crash roughly in the middle.
Labeling the full window as positive conflates two structurally different
signal types,pre-crash behavior and post-crash recovery. The correct
formulation labels only `[window_start, crash)` as positive and excludes
`[crash, window_end]` from training entirely. This is the most
important design decision in the project.

---

## Modeling Choices

### Model: Random Forest with balanced class weights

**Why Random Forest over alternatives:**
- `class_weight='balanced'` handles the ~5% positive rate without
  additional hyperparameter tuning
- No feature scaling required, robust to heterogeneous server baselines
  (mean CPU ranges from 0.09% to 89.79% across servers)
- Feature importances are interpretable and reveal which signals matter
- With only 12 anomaly windows, the marginal gain from gradient boosting
  (LightGBM, XGBoost) is not statistically detectable ⇒ RF is the right
  complexity level for this dataset size

**Why not LSTM/forecasting approach:**
The task provides explicit incident labels, making supervised
classification the natural fit. Unsupervised forecasting + residual
detection would require defining "anomaly" without labels, we already
have that information.

### Features (W=24 steps = 120-minute look-back)

| Feature | Description | Why it matters |
|---|---|---|
| `rolling_std_48` | 4-hour rolling std | Dominant signal (40% importance)  instability builds over hours before crashes |
| `rolling_std_24` | 2-hour rolling std | Shorter-window instability (17% importance) |
| `hour_sin`, `hour_cos` | Cyclical time encoding | Daily patterns in incident probability (23% combined) |
| `long_trend` | 4-hour mean drift | Is the baseline slowly shifting? |
| `value_norm` | Raw CPU value | Absolute level matters near-zero is a distinct regime |
| `ema_cross` | Fast/slow EMA difference | Momentum signal |
| `wavelet_jitter` | Haar detail coefficient | High-frequency noise bursts |
| `z_score` | Local deviation | Point-level anomaly score |

**W=24 was chosen over W=12 and W=36** after empirical comparison.
W=24 outperforms W=12 because the dominant signals operate over
multi-hour windows. W=36 without variance features degrades recall
and produces 16-hour continuous alarm bursts on some servers.

Short-term features (`value_diff1`, `value_diff2`, `clean_deviation`) contribute
near-zero importance, they would be more fit for basic anomaly detection "is this
an anomaly?" and not "will an anomaly happen in the future?". That is why they are
not included in the final model

---

## Evaluation Setup

### Why leave-one-server-out CV

Each server has a distinct operating regime. Random train/test splitting
would leak server-specific patterns into the test set, inflating results.
LOSO CV ensures the model is always evaluated on a server it has never
seen during training.

### Metrics

**Per-incident recall** — for each anomaly window, did the model raise
at least one alert between `window_start` and the exact crash time?
This is the operationally meaningful metric. Per-step accuracy is
misleading: a model predicting everything as positive gets 100% step-recall
but is useless as an alerting system.

**FP events/day** — false-positive alert clusters per day, aggregated
across all servers. Consecutive alert ticks above threshold count as one
event. 1.26 FP/day across 8 servers corresponds to ~0.16 FP/server/day,
or roughly one false alarm per server every 6 days.

**Median lead time** — minutes between the first alert and the exact crash.
450-minute median means the model typically warns ~7.5 hours before a crash.

**Alert threshold** — the probability cutoff above which the model fires.
A full threshold sweep is reported, allowing the operating point to be
adjusted based on the precision/recall trade-off required for a given
deployment context.

---

## Analysis of Results

### What works

Four servers show selective, precise behaviour (8–14% firing rate, short
bursts under 2 hours): `fe7f93`, `24ae8d`, `825cc2`, `77c1ca`. On these,
the model fires in compact bursts around genuine anomaly zones and stays
quiet during normal operation. `c6585a`, the only server with no labeled
anomalies produces **0% firing rate** => evidence the model
has learned a real signal.

### What doesn't work

Three servers (`5f5533`, `53ea38`, `ac20cd`) show high firing rates
(33–44%) with long bursts averaging 4–6 hours.

### Missed incident

The one missed incident is on `24ae8d` (mean CPU 0.13%). This server's
anomaly manifests as isolated rare spikes from a near-zero baseline —
structurally different from the sustained variance increase the model
learned from higher-utilization servers. Per-server threshold tuning
would likely recover this incident.

---

## Limitations

**Dataset size**: 12 anomaly windows across 8 servers is a small sample.
Results should be interpreted as a proof-of-concept. Statistical confidence
in the 92% recall estimate is limited — one additional missed incident
would drop it to 83%.

**FP event counting**: Consecutive alerts are grouped into one event.
A 6-hour continuous alarm counts as a single FP event, understating the
operational impact of regime-shift servers. A cooldown-based counter
would be more honest.

**Single metric**: Only CPU utilization is used. In practice, crashes are
often preceded by correlated changes across multiple metrics (memory,
disk I/O, network). Multi-metric models would likely improve both
recall and FP rates.

**Time-of-day correlation**: `hour_sin`/`hour_cos` contribute 23% of
feature importance. With only 12 incidents, some of this may reflect
spurious temporal clustering rather than genuine daily patterns.

---

**Production improvements over this prototype:**
- Per-server threshold tuning to handle heterogeneous baselines
- Alert cooldown (suppress re-alerting for 30 minutes after firing)
- Drift detection to trigger retraining when server baseline shifts
- Multi-metric input (CPU + memory + network) for richer signal

---

## Project Structure

```
├── PredictiveAlert.ipynb     ← main notebook
├── visualize_data.ipynb      ← data visualization
├── src/
│   ├── labeling.py           ← load_and_label_file, get_anomaly_windows
│   ├── features.py           ← engineer_features, create_sliding_windows, FEATURE_COLS
│   ├── model.py              ← build_model, train, predict_proba, get_feature_importances
│   └── evaluate.py           ← threshold_sweep, find_operating_point, per_incident_recall
├── data/                     ← 8 CloudWatch CSV files
├── labels/
│   ├── combined_windows.json
│   └── combined_labels.json
└── requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook PredictiveAlert.ipynb
```