"""LIME explainer (optional dependency).

LIME (Local Interpretable Model-agnostic Explanations) explains a single
prediction by fitting a simple, interpretable model (a weighted linear model)
to the black-box model's behaviour in the local neighbourhood of that one
instance.

The ``lime`` package is imported lazily. Install it with ``pip install lime``.
"""

from __future__ import annotations

import numpy as np


def _require_lime():
    try:
        from lime.lime_tabular import LimeTabularExplainer  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'lime' package is required for LIME explanations. "
            "Install it with:  pip install lime"
        ) from exc
    from lime.lime_tabular import LimeTabularExplainer

    return LimeTabularExplainer


def lime_explain_instance(
    model,
    X_train,
    instance,
    feature_names,
    class_names,
    num_features: int = 10,
):
    """Explain a single instance with LIME.

    Parameters
    ----------
    model:
        Fitted estimator/pipeline exposing ``predict_proba``.
    X_train:
        Training frame used to characterise the feature distributions.
    instance:
        A single row (1-D array or Series) to explain.
    feature_names, class_names:
        Metadata used for readable output.

    Returns
    -------
    lime.explanation.Explanation
        Call ``.as_list()`` for text or ``.as_pyplot_figure()`` for a chart.
    """
    LimeTabularExplainer = _require_lime()

    explainer = LimeTabularExplainer(
        training_data=np.asarray(X_train),
        feature_names=list(feature_names),
        class_names=list(class_names),
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    explanation = explainer.explain_instance(
        data_row=np.asarray(instance).ravel(),
        predict_fn=model.predict_proba,
        num_features=num_features,
    )
    return explanation
