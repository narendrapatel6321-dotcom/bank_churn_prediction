import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN

# =============================================================================
# 1. FEATURE ENGINEERING
# =============================================================================

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that derives domain features from raw columns.

    The four base features are always derived. An optional extra_features
    list can be passed from the notebook to add experimental features
    without touching helpers.py.

    Base derived features
    ---------------------
    BalanceSalaryRatio     : Balance / (EstimatedSalary + 1)
    IsActive_by_CreditCard : IsActiveMember × HasCrCard
    ProductsPerYear        : NumOfProducts / (Tenure + 1)
    AgeGroup               : Age bucketed into Young / Middle / Senior.

    Parameters
    ----------
    extra_features : list[Callable] | None
                     Each callable receives the DataFrame after base features
                     are added and returns it with one or more columns appended.
                     Applied in order. Define in the notebook, e.g.:
                     [lambda X: X.assign(AgeBalance=X["Age"] * X["Balance"])]
    """

    def __init__(self, extra_features: list | None = None):
        self.extra_features = extra_features

    def fit(self, X: pd.DataFrame, y=None) -> "ChurnFeatureEngineer":
        """No fitting required — all transforms are stateless."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply base feature derivations, then any extra_features, in order.

        Parameters
        ----------
        X : Raw feature DataFrame (post-drop of ID columns).

        Returns
        -------
        pd.DataFrame with original columns plus all derived features.
        """
        X = X.copy()
        X["BalanceSalaryRatio"]     = X["Balance"] / (X["EstimatedSalary"] + 1)
        X["IsActive_by_CreditCard"] = X["IsActiveMember"] * X["HasCrCard"]
        X["ProductsPerYear"]        = X["NumOfProducts"] / (X["Tenure"] + 1)
        X["AgeGroup"] = pd.cut(
            X["Age"],
            bins=[0, 35, 55, 100],
            labels=["Young", "Middle", "Senior"],
        ).astype(str)
        if self.extra_features is not None:
            for fn in self.extra_features:
                X = fn(X)
        return X

# =============================================================================
# 2. PIPELINE CONSTRUCTION
# =============================================================================

def _make_preprocessor(
    num_features: list[str],
    cat_features: list[str],
    passthrough_features: list[str],
) -> ColumnTransformer:
    """
    Instantiate a fresh ColumnTransformer from caller-supplied column lists.

    Called inside build_pipeline so each pipeline owns an independent,
    unfitted preprocessor — prevents accidental sharing of fitted state
    across multiple pipeline instances during cross-validation.

    Parameters
    ----------
    num_features         : Numeric columns to scale with StandardScaler.
    cat_features         : Categorical columns to one-hot encode.
    passthrough_features : Binary columns passed through unchanged.
    """
    return ColumnTransformer(
        transformers=[
            ("num",  StandardScaler(), num_features),
            ("cat",  OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_features),
            ("pass", "passthrough", passthrough_features),
        ],
        remainder="drop",
    )

def build_pipeline(
    classifier,
    num_features: list[str],
    cat_features: list[str],
    passthrough_features: list[str],
    use_smote: bool = False,
    use_adasyn: bool = False,
    extra_features: list | None = None,
) -> Pipeline | ImbPipeline:
    """
    Assemble the full modelling pipeline for a given classifier.

    Pipeline order
    --------------
    1. ChurnFeatureEngineer  — derives base + any extra features.
    2. ColumnTransformer     — scales numerics, encodes categoricals.
    3. SMOTE or ADASYN (optional) — oversamples minority class on training folds only.
    4. classifier            — any sklearn-compatible estimator.

    Parameters
    ----------
    classifier           : Unfitted sklearn-compatible estimator.
    num_features         : Numeric columns to pass to StandardScaler.
    cat_features         : Categorical columns to pass to OneHotEncoder.
    passthrough_features : Binary columns passed through unchanged.
    use_smote            : If True, inserts SMOTE after preprocessing.
    use_adasyn           : If True, inserts ADASYN after preprocessing.
                           use_smote and use_adasyn are mutually exclusive.
    extra_features       : list[Callable] | None — passed to ChurnFeatureEngineer.
                           Each callable adds one or more experimental columns.

    Returns
    -------
    Fitted-ready Pipeline or ImbPipeline.

    Raises
    ------
    ValueError if both use_smote and use_adasyn are True.
    """
    if use_smote and use_adasyn:
        raise ValueError("use_smote and use_adasyn are mutually exclusive. Pick one.")

    steps = [
        ("engineer",     ChurnFeatureEngineer(extra_features=extra_features)),
        ("preprocessor", _make_preprocessor(num_features, cat_features, passthrough_features)),
    ]
    if use_smote:
        steps.append(("sampler", SMOTE(random_state=42)))
    elif use_adasyn:
        steps.append(("sampler", ADASYN(random_state=42)))
    steps.append(("classifier", classifier))

    PipelineClass = ImbPipeline if (use_smote or use_adasyn) else Pipeline
    return PipelineClass(steps)
    