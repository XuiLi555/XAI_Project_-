"""Basic smoke tests. Run with:  pytest -q

These cover the parts of the pipeline that depend only on scikit-learn, so
they pass without SHAP or LIME installed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xai import load_dataset, train_model, evaluate_model
from xai.explainers import (
    tree_feature_importance,
    permutation_feature_importance,
)


def test_load_dataset():
    data = load_dataset("breast_cancer", test_size=0.25)
    assert len(data.feature_names) == 30
    assert len(data.X_train) > len(data.X_test)
    assert set(data.y_train) <= {0, 1}


def test_train_and_evaluate():
    data = load_dataset("wine")
    model = train_model(data.X_train, data.y_train, n_estimators=50)
    result = evaluate_model(model, data.X_test, data.y_test, data.target_names)
    # A random forest should comfortably beat chance on wine.
    assert result.accuracy > 0.85
    assert result.confusion.shape == (3, 3)


def test_feature_importance_shapes():
    data = load_dataset("breast_cancer")
    model = train_model(data.X_train, data.y_train, n_estimators=50)

    tree_imp = tree_feature_importance(model, data.feature_names)
    assert len(tree_imp) == len(data.feature_names)

    perm_imp = permutation_feature_importance(
        model, data.X_test, data.y_test, n_repeats=3
    )
    assert len(perm_imp) == len(data.feature_names)
