"""
model.py
--------
Training and prediction wrappers for the predictive alerting model.

Model choice: Random Forest with balanced class weights.

Why Random Forest over alternatives:
  - Handles the extreme class imbalance (~5% positive rate) via class_weight='balanced'
  - No feature scaling required — robust to the heterogeneous value ranges
    across servers (0.09% mean vs 89.79% mean)
  - Feature importances are interpretable — useful for the analysis section
  - Deterministic enough to reproduce results across CV folds

Alternatives considered:
  - LightGBM: marginally better on imbalanced data via scale_pos_weight,
    but adds a dependency and the improvement is unlikely to be decisive
    given the dataset size (~20k samples, ~500 positives)
  - LSTM: appropriate for sequential forecasting, but overkill for Task #1
    which frames this as binary classification with a flat feature vector
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def build_model(n_estimators: int = 200, random_state: int = 42) -> RandomForestClassifier:
    """
    Return an untrained RandomForestClassifier configured for imbalanced alerting data.

    Parameters
    ----------
    n_estimators : int
        Number of trees. 200 is a reasonable ceiling for this dataset size.
    random_state : int
        Seed for reproducibility.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",   # up-weights the minority (positive) class
        max_depth=12,              # prevents deep trees from memorising noise
        min_samples_leaf=5,        # requires at least 5 samples per leaf
        random_state=random_state,
        n_jobs=-1,
    )


def train(
    model: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """Fit the model and return it."""
    model.fit(X_train, y_train)
    return model


def predict_proba(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """Return the probability of the positive class for each sample."""
    return model.predict_proba(X)[:, 1]


def get_feature_importances(
    model: RandomForestClassifier,
    feature_cols: list[str],
    W: int,
) -> dict[str, float]:
    """
    Return mean impurity-based importance per feature (aggregated across time steps).

    Because the model receives a flattened (W × n_features) vector, each
    named feature appears W times. This function sums across those W copies
    and returns one importance value per named feature.

    Parameters
    ----------
    model : trained RandomForestClassifier
    feature_cols : list of feature names (in the order passed to create_sliding_windows)
    W : look-back window size used when building X

    Returns
    -------
    dict mapping feature name → aggregated importance (sums to 1.0)
    """
    raw = model.feature_importances_  # length W * n_features
    n_features = len(feature_cols)
    importances = {}
    for j, name in enumerate(feature_cols):
        # Indices of this feature across all W time steps
        indices = [j + k * n_features for k in range(W)]
        importances[name] = float(raw[indices].sum())

    # Normalise so values sum to 1
    total = sum(importances.values())
    return {k: v / total for k, v in sorted(importances.items(), key=lambda x: -x[1])}
