"""
app_utils.py — Minimal shim for Streamlit deployment.
====================================================
Only contains ChurnFeatureEngineer so that joblib can unpickle
the saved pipeline. Does NOT import optuna, imblearn, catboost,
xgboost, seaborn, or any training-only dependencies.

The full helpers.py (with all training utilities) lives in the
notebook's GitHub repo and is only needed during training.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that derives domain features.

    Base derived features:
        BalanceSalaryRatio, IsActive_by_CreditCard,
        ProductsPerYear, AgeGroup.
    """

    def __init__(self, extra_features=None):
        self.extra_features = extra_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["BalanceSalaryRatio"] = X["Balance"] / (X["EstimatedSalary"] + 1)
        X["IsActive_by_CreditCard"] = X["IsActiveMember"] * X["HasCrCard"]
        X["ProductsPerYear"] = X["NumOfProducts"] / (X["Tenure"] + 1)
        X["AgeGroup"] = pd.cut(
            X["Age"],
            bins=[0, 35, 55, 100],
            labels=["Young", "Middle", "Senior"],
        ).astype(str)
        if self.extra_features is not None:
            for fn in self.extra_features:
                X = fn(X)
        return X
