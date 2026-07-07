"""Explainability methods.

Each explainer is a thin, well-documented wrapper. The SHAP and LIME
wrappers import their heavy dependencies lazily so the rest of the package
still works if those optional libraries are not installed.
"""

from .feature_importance import (
    permutation_feature_importance,
    tree_feature_importance,
)
from .partial_dependence import partial_dependence_plot
from .shap_explainer import shap_summary, shap_explain_instance
from .lime_explainer import lime_explain_instance

__all__ = [
    "tree_feature_importance",
    "permutation_feature_importance",
    "partial_dependence_plot",
    "shap_summary",
    "shap_explain_instance",
    "lime_explain_instance",
]
