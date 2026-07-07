"""Global feature-importance explainers.

Two complementary views of *which features matter*:

1. ``tree_feature_importance`` — the random forest's built-in impurity-based
   importance. Fast, but biased toward high-cardinality / continuous features.
2. ``permutation_feature_importance`` — model-agnostic. Measures how much a
   metric drops when a single feature's values are shuffled. Slower but more
   trustworthy, and it reflects performance on data the model didn't train on.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance


def tree_feature_importance(model, feature_names) -> pd.Series:
    """Return impurity-based importances from the fitted tree model.

    Works with a bare estimator or a Pipeline whose final step is named
    ``"clf"``.
    """
    estimator = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    importances = pd.Series(
        estimator.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    return importances


def permutation_feature_importance(
    model, X, y, n_repeats: int = 20, random_state: int = 42
) -> pd.Series:
    """Return permutation importances (mean over repeats)."""
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )
    importances = pd.Series(
        result.importances_mean, index=X.columns
    ).sort_values(ascending=False)
    return importances


def plot_importance(importances: pd.Series, top_n: int = 15, title: str = "Feature importance"):
    """Horizontal bar chart of the top-N features. Returns the Matplotlib figure."""
    top = importances.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(top) + 1))
    ax.barh(top.index, top.values, color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig
