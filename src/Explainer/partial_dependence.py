"""Partial Dependence Plots (PDP).

A PDP shows the *marginal* effect of one (or two) features on the model's
predicted output, averaging over all other features. It answers: "as this
feature increases, how does the average prediction change?"
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay


def partial_dependence_plot(model, X, features, target=None):
    """Draw partial dependence for the given features.

    Parameters
    ----------
    model:
        A fitted estimator or pipeline.
    X:
        The (training) feature frame the PDP is computed over.
    features:
        List of feature names or column indices. Use a tuple like
        ``("a", "b")`` inside the list to request a 2-D interaction plot.
    target:
        Class index for multiclass problems (ignored for binary).

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(features)
    fig, ax = plt.subplots(figsize=(5 * min(n, 3), 4 * ((n + 2) // 3)))
    kwargs = {}
    if target is not None:
        kwargs["target"] = target
    PartialDependenceDisplay.from_estimator(model, X, features, ax=ax, **kwargs)
    fig.tight_layout()
    return fig
