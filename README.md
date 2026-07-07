# XAI_Project

# Explainable AI (XAI) Toolkit

[![CI](https://github.com/your-username/explainable-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/explainable-ai/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A small, readable project that shows how to **interpret machine learning
models** using several complementary explainability techniques. It trains a
random forest on a bundled dataset and explains its predictions both
*globally* (which features matter overall) and *locally* (why one specific
prediction was made).

Built to be educational: every module is short and heavily commented.

---

## Why explainability?

Modern ML models are accurate but opaque. Explainable AI helps you answer:

- **Which features drive the model?** → feature importance, permutation importance
- **How does a feature affect predictions?** → partial dependence plots (PDP)
- **Why did the model predict *this* for *this* sample?** → SHAP, LIME

| Method | Scope | Model-agnostic? | What it tells you |
|---|---|---|---|
| Tree importance | Global | No (trees only) | Impurity-based feature ranking |
| Permutation importance | Global | Yes | Drop in performance when a feature is shuffled |
| Partial Dependence (PDP) | Global | Yes | Marginal effect of a feature on predictions |
| SHAP | Global + Local | Yes | Additive, game-theoretic feature attributions |
| LIME | Local | Yes | Local linear approximation around one instance |

---
