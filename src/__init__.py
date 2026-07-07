"""
Explainable AI (XAI) toolkit.

A small, well-documented project demonstrating how to interpret machine
learning models using several complementary explainability techniques:

- Built-in / permutation feature importance (global)
- Partial Dependence Plots (global)
- SHAP values (global + local)   [optional dependency]
- LIME (local)                   [optional dependency]

The goal is to be readable and educational rather than exhaustive.
"""

from .data import load_dataset
from .model import train_model, evaluate_model

__version__ = "0.1.0"

__all__ = [
    "load_dataset",
    "train_model",
    "evaluate_model",
    "__version__",
]
