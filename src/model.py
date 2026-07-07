"""Model training and evaluation.

A deliberately simple ``RandomForestClassifier`` wrapped in a scaling
pipeline. Random forests are a good teaching model for explainability
because they expose native feature importances *and* work well with
model-agnostic methods like SHAP and LIME.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class EvalResult:
    """Holds evaluation metrics for a fitted model."""

    accuracy: float
    f1_macro: float
    confusion: np.ndarray
    report: str

    def summary(self) -> str:
        return (
            f"Accuracy : {self.accuracy:.4f}\n"
            f"F1 (macro): {self.f1_macro:.4f}\n\n"
            f"{self.report}"
        )


def train_model(
    X_train,
    y_train,
    n_estimators: int = 300,
    max_depth: int | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Train a StandardScaler + RandomForest pipeline.

    Scaling is not strictly required for tree models, but keeping it in the
    pipeline means the exact same object works if you later swap in a
    distance- or gradient-based estimator.
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Pipeline, X_test, y_test, target_names=None) -> EvalResult:
    """Evaluate a fitted model and return a metrics bundle."""
    y_pred = model.predict(X_test)
    return EvalResult(
        accuracy=float(accuracy_score(y_test, y_pred)),
        f1_macro=float(f1_score(y_test, y_pred, average="macro")),
        confusion=confusion_matrix(y_test, y_pred),
        report=classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        ),
    )
