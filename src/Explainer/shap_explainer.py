"""SHAP explainers (optional dependency).

SHAP (SHapley Additive exPlanations) attributes a model's prediction to each
feature using a game-theoretic idea: how much each feature contributes,
averaged over all possible orderings. It gives both global (summary) and
local (single-prediction) explanations that are additive and consistent.

The ``shap`` package is imported lazily so importing this module never fails
when SHAP is not installed. Install it with ``pip install shap``.
"""

from __future__ import annotations


def _require_shap():
    try:
        import shap  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'shap' package is required for SHAP explanations. "
            "Install it with:  pip install shap"
        ) from exc
    return shap


def _unwrap(model):
    """Return (preprocessing_transform, tree_estimator) from a pipeline.

    TreeExplainer needs the raw tree model, so we split a Scaler+clf pipeline
    into the scaler transform and the underlying estimator.
    """
    if hasattr(model, "named_steps"):
        estimator = model.named_steps["clf"]
        transform = lambda X: model[:-1].transform(X)  # noqa: E731
        return transform, estimator
    return (lambda X: X), model


def shap_summary(model, X, max_display: int = 15):
    """Compute SHAP values and draw a global summary (beeswarm) plot.

    Returns the SHAP ``Explanation`` object so callers can reuse it.
    """
    shap = _require_shap()
    transform, estimator = _unwrap(model)
    X_trans = transform(X)

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer(X_trans)

    shap.summary_plot(
        shap_values, features=X, max_display=max_display, show=False
    )
    return shap_values


def shap_explain_instance(model, X, index: int = 0):
    """Explain a single prediction with a SHAP waterfall plot.

    Parameters
    ----------
    index:
        Row position in ``X`` to explain.
    """
    shap = _require_shap()
    transform, estimator = _unwrap(model)
    X_trans = transform(X)

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer(X_trans)

    # For multiclass, shap_values has an extra axis; pick the predicted class.
    sv = shap_values[index]
    if sv.values.ndim > 1:
        pred_class = int(model.predict(X.iloc[[index]])[0])
        sv = sv[..., pred_class]

    shap.plots.waterfall(sv, show=False)
    return sv
