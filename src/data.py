"""Dataset loading and train/test splitting.

Uses scikit-learn's bundled datasets so the project runs with **no internet
connection and no data downloads**. Two datasets are supported out of the box:

- ``"breast_cancer"`` : binary classification (569 samples, 30 features)
- ``"wine"``          : multiclass classification (178 samples, 13 features)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

_LOADERS = {
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
}


@dataclass
class Dataset:
    """Container bundling the split data and useful metadata."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    name: str

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Dataset(name={self.name!r}, "
            f"n_train={len(self.X_train)}, n_test={len(self.X_test)}, "
            f"n_features={len(self.feature_names)}, "
            f"classes={self.target_names})"
        )


def load_dataset(
    name: str = "breast_cancer",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dataset:
    """Load a bundled dataset and return a train/test split.

    Parameters
    ----------
    name:
        One of ``"breast_cancer"`` or ``"wine"``.
    test_size:
        Fraction of the data held out for testing.
    random_state:
        Seed for a reproducible split.

    Returns
    -------
    Dataset
        A dataclass with train/test frames and metadata.
    """
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset {name!r}. Choose from {sorted(_LOADERS)}."
        )

    raw = _LOADERS[name]()
    feature_names = list(raw.feature_names)
    X = pd.DataFrame(raw.data, columns=feature_names)
    y = raw.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return Dataset(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        target_names=list(raw.target_names),
        name=name,
    )
