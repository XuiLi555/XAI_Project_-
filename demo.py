"""End-to-end Explainable AI demo.

Run from the project root:

    python examples/demo.py --dataset breast_cancer

This will:
  1. Load a bundled dataset (no download needed)
  2. Train a random forest
  3. Report accuracy / F1
  4. Produce global explanations (feature importance, permutation, PDP)
  5. Produce local explanations (SHAP + LIME) if those libraries are installed

All figures are written to ``outputs/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

# Allow running the script directly without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xai import load_dataset, train_model, evaluate_model  # noqa: E402
from xai.explainers import (  # noqa: E402
    tree_feature_importance,
    permutation_feature_importance,
    partial_dependence_plot,
    shap_summary,
    shap_explain_instance,
    lime_explain_instance,
)
from xai.explainers.feature_importance import plot_importance  # noqa: E402

OUT = ROOT / "outputs"


def _save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explainable AI demo")
    parser.add_argument(
        "--dataset", default="breast_cancer", choices=["breast_cancer", "wine"]
    )
    args = parser.parse_args()

    print(f"\n[1/5] Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    print(f"      {data}")

    print("\n[2/5] Training random forest ...")
    model = train_model(data.X_train, data.y_train)

    print("\n[3/5] Evaluating ...")
    result = evaluate_model(
        model, data.X_test, data.y_test, target_names=data.target_names
    )
    print(result.summary())

    print("\n[4/5] Global explanations")
    tree_imp = tree_feature_importance(model, data.feature_names)
    _save(plot_importance(tree_imp, title="Impurity-based importance"),
          "01_tree_importance.png")

    perm_imp = permutation_feature_importance(model, data.X_test, data.y_test)
    _save(plot_importance(perm_imp, title="Permutation importance"),
          "02_permutation_importance.png")

    top2 = list(tree_imp.head(2).index)
    print(f"      partial dependence for top-2 features: {top2}")
    _save(partial_dependence_plot(model, data.X_train, top2),
          "03_partial_dependence.png")

    print("\n[5/5] Local explanations (optional libraries)")
    # --- SHAP ---
    try:
        shap_summary(model, data.X_test)
        _save(plt.gcf(), "04_shap_summary.png")

        shap_explain_instance(model, data.X_test, index=0)
        _save(plt.gcf(), "05_shap_waterfall_instance0.png")
    except ImportError as exc:
        print(f"      [skip SHAP] {exc}")

    # --- LIME ---
    try:
        explanation = lime_explain_instance(
            model,
            data.X_train,
            data.X_test.iloc[0],
            data.feature_names,
            data.target_names,
        )
        print("      LIME explanation for test instance 0:")
        for feat, weight in explanation.as_list():
            print(f"        {weight:+.4f}  {feat}")
        _save(explanation.as_pyplot_figure(), "06_lime_instance0.png")
    except ImportError as exc:
        print(f"      [skip LIME] {exc}")

    print("\nDone. See the outputs/ folder for figures.\n")


if __name__ == "__main__":
    main()
