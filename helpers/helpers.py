# =============================================================================
# helpers.py — Bank Churn Prediction
# =============================================================================
# All reusable functions, transformers, and plotting utilities.
# Import this module at the top of the notebook to keep cells clean.
# =============================================================================

from __future__ import annotations

import os
import warnings
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import joblib
import optuna

from imblearn.over_sampling import SMOTE
from imblearn.pipeline      import Pipeline as ImbPipeline
from sklearn.base           import BaseEstimator, TransformerMixin
from sklearn.calibration    import calibration_curve
from sklearn.compose        import ColumnTransformer
from sklearn.metrics        import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import (cross_val_score,
                                     cross_val_predict)
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

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
    extra_features: list | None = None,
) -> Pipeline | ImbPipeline:
    """
    Assemble the full modelling pipeline for a given classifier.

    Pipeline order
    --------------
    1. ChurnFeatureEngineer  — derives base + any extra features.
    2. ColumnTransformer     — scales numerics, encodes categoricals.
    3. SMOTE (optional)      — oversamples minority class on training folds only.
    4. classifier            — any sklearn-compatible estimator.

    Parameters
    ----------
    classifier           : Unfitted sklearn-compatible estimator.
    num_features         : Numeric columns to pass to StandardScaler.
    cat_features         : Categorical columns to pass to OneHotEncoder.
    passthrough_features : Binary columns passed through unchanged.
    use_smote            : If True, inserts SMOTE after preprocessing.
    extra_features       : list[Callable] | None — passed to ChurnFeatureEngineer.
                           Each callable adds one or more experimental columns.

    Returns
    -------
    Fitted-ready Pipeline or ImbPipeline.
    """
    steps = [
        ("engineer",     ChurnFeatureEngineer(extra_features=extra_features)),
        ("preprocessor", _make_preprocessor(num_features, cat_features, passthrough_features)),
    ]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))
    steps.append(("classifier", classifier))

    PipelineClass = ImbPipeline if use_smote else Pipeline
    return PipelineClass(steps)
    
# =============================================================================
# 3. HYPERPARAMETER TUNING
# =============================================================================

def run_optuna_study(
    objective_fn: Callable,
    n_trials: int,
    best_params_update: dict,
    use_pruner: bool = True,
    show_progress_bar: bool = False,
    verbose: bool = True,
) -> tuple[optuna.Study, dict]:
    """
    Create and run an Optuna maximisation study.

    Parameters
    ----------
    objective_fn       : Callable(trial) → float score to maximise.
    n_trials           : Number of trials. TPE meaningfully outperforms
                         random search only after ~25 trials.
    best_params_update : Fixed params to merge into best_params after
                         optimisation (e.g. random_state, verbosity).
    use_pruner         : If True, uses MedianPruner to cut unpromising
                         trials early. Recommended for iterative models.
    show_progress_bar  : If True, shows tqdm progress bar. Disabled by
                         default since verbose=True already prints per trial.
    verbose            : If True, prints trial number, F1, and chosen params
                         after every completed trial.

    Returns
    -------
    study       : Completed Optuna Study object.
    best_params : Merged dict of optimised + fixed parameters.
    """
    def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(
                f"  Trial {trial.number:>3} | F1: {trial.value:.4f} | "
                f"Params: { {k: round(v, 4) if isinstance(v, float) else v for k, v in trial.params.items()} }"
            )

    sampler   = optuna.samplers.TPESampler(seed=42)
    pruner    = optuna.pruners.MedianPruner() if use_pruner else optuna.pruners.NopPruner()
    study     = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    callbacks = [_trial_callback] if verbose else []
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
        callbacks=callbacks,
    )
    best_params = {**study.best_params, **best_params_update}
    return study, best_params

def make_objective(
    model_fn: Callable,
    param_space: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    skf,
    best_strategy: str,
    num_features: list[str],
    cat_features: list[str],
    passthrough_features: list[str],
    extra_features: list | None = None,
) -> Callable:
    """
    Build a generic Optuna objective from a model factory and a param space.

    This keeps helpers.py experiment-agnostic — all model choices and search
    space definitions live in the notebook, not here.

    Parameters
    ----------
    model_fn             : Callable(**params) → unfitted sklearn estimator.
                           Define this in the notebook, e.g.:
                           lambda **p: LGBMClassifier(**p, verbosity=-1, random_state=SEED)
    param_space          : Dict of {param_name: Callable(trial) → value}.
                           Each value is a lambda wrapping a trial.suggest_* call, e.g.:
                           {"n_estimators": lambda t: t.suggest_int("n_estimators", 100, 1000)}
    X_train              : Training features.
    y_train              : Training labels.
    skf                  : StratifiedKFold splitter.
    best_strategy        : 'SMOTE' or 'class_weight' — controls pipeline construction.
    num_features         : Numeric feature list — passed to build_pipeline.
    cat_features         : Categorical feature list — passed to build_pipeline.
    passthrough_features : Passthrough feature list — passed to build_pipeline.
    extra_features       : Optional list of feature engineering callables.

    Returns
    -------
    objective : Callable(trial) → float, ready to pass to run_optuna_study.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {k: v(trial) for k, v in param_space.items()}
        pipe   = build_pipeline(
            classifier           = model_fn(**params),
            num_features         = num_features,
            cat_features         = cat_features,
            passthrough_features = passthrough_features,
            use_smote            = (best_strategy == "SMOTE"),
            extra_features       = extra_features,
        )
        return cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1", n_jobs=-1).mean()
    return objective
    
# =============================================================================
# 4. THRESHOLD SELECTION
# =============================================================================
 
def find_optimal_threshold(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    skf,
    recall_floor: float = 0.60,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the decision threshold that maximises F1 subject to a recall floor,
    using out-of-fold probability estimates to avoid optimistic bias.
 
    Strategy
    --------
    1. Generate OOF probabilities via cross_val_predict.
    2. Sweep the precision-recall curve.
    3. Among all thresholds where recall ≥ recall_floor, pick the one with
       the highest F1. Falls back to global F1-max if the floor is unachievable.
 
    Parameters
    ----------
    pipeline     : Fitted or unfitted pipeline (will be CV-predicted, not fitted).
    X_train      : Training features.
    y_train      : Training labels.
    skf          : StratifiedKFold cross-validator.
    recall_floor : Minimum acceptable recall on the positive class.
 
    Returns
    -------
    best_threshold : float
    precisions     : np.ndarray
    recalls        : np.ndarray
    thresholds     : np.ndarray
    f1_arr         : np.ndarray
    """   
    y_probas = cross_val_predict(pipeline, X_train, y_train, cv=skf, method="predict_proba",n_jobs=-1)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_probas)
    f1_arr    = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    valid_idx = np.where(recalls[:-1] >= recall_floor)[0]
 
    if len(valid_idx) == 0:
        print(f"  No threshold achieves Recall ≥ {recall_floor}. Using global F1 max.")
        best_idx = np.argmax(f1_arr[:-1])
    else:
        best_idx = valid_idx[np.argmax(f1_arr[valid_idx])]
 
    best_threshold = thresholds[best_idx]
 
    print(f"  Recall floor  : {recall_floor}")
    print(f"  Best threshold: {best_threshold:.4f}")
    print(f"  Precision     : {precisions[best_idx]:.3f}")
    print(f"  Recall        : {recalls[best_idx]:.3f}")
    print(f"  F1            : {f1_arr[best_idx]:.3f}")
 
    return best_threshold, precisions, recalls, thresholds, f1_arr
 
# =============================================================================
# 5. SHAP UTILITIES
# =============================================================================

def build_shap_explainer(classifier, background_data: pd.DataFrame, model_type: str):
    """
    Return the appropriate SHAP explainer for the given model type.

    TreeExplainer is exact and fast for tree-based models.
    KernelExplainer is model-agnostic but slower — uses a k-means summarised
    background sample to keep compute tractable.

    Parameters
    ----------
    classifier      : Fitted sklearn-compatible estimator (extracted from pipeline).
    background_data : Transformed test data used as the SHAP background.
    model_type      : 'tree' or 'linear'. Caller declares this in the notebook
                      next to the model definition — helpers never needs to know
                      the model's name.

    Returns
    -------
    shap.Explainer instance ready to call .shap_values() on.

    Raises
    ------
    ValueError if model_type is not 'tree' or 'linear'.
    """
    if model_type == "tree":
        return shap.TreeExplainer(classifier)
    elif model_type == "linear":
        background = shap.kmeans(background_data, k=50)
        return shap.KernelExplainer(classifier.predict_proba, background)
    else:
        raise ValueError(
            f"model_type must be 'tree' or 'linear', got '{model_type}'. "
            f"Set MODEL_TYPE in the notebook next to your model definition."
        )

def get_shap_values_class1(shap_values, index: int | None = None) -> np.ndarray:
    """
    Extract SHAP values for the positive class (churn = 1).

    Handles all output formats returned by different SHAP explainer types:
      - list of arrays  : [class_0_array, class_1_array]
      - 3-D array       : (n_samples, n_features, n_classes)
      - 2-D array       : (n_samples, n_features)  — binary output

    Parameters
    ----------
    shap_values : Raw output from explainer.shap_values().
    index       : Sample index to extract. Pass 0 when computing values for
                  a single row reshaped as (1, n_features).

    Returns
    -------
    np.ndarray of shape (n_features,) if index given, else (n_samples, n_features).
    """
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
    return sv if index is None else sv[index]


def get_transformed_test_data(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    num_features: list[str],
    cat_features: list[str],
    passthrough_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Extract the fully-transformed test set and reconstructed feature names
    from a fitted pipeline.

    Needed to pass a named DataFrame to SHAP so that plots display
    human-readable feature labels instead of column indices.

    Parameters
    ----------
    pipeline             : Fitted pipeline with 'engineer' and 'preprocessor' steps.
    X_test               : Raw test features (same format as training input).
    num_features         : Numeric feature list used when building the pipeline.
    cat_features         : Categorical feature list used when building the pipeline.
    passthrough_features : Passthrough feature list used when building the pipeline.

    Returns
    -------
    X_test_df         : Transformed test set as a named DataFrame.
    all_feature_names : Ordered list of feature names matching the columns.
    """
    engineer     = pipeline.named_steps["engineer"]
    preprocessor = pipeline.named_steps["preprocessor"]

    X_engineered  = engineer.transform(X_test)
    X_transformed = preprocessor.transform(X_engineered)

    ohe_features      = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features)
    all_feature_names = num_features + list(ohe_features) + passthrough_features

    return pd.DataFrame(X_transformed, columns=all_feature_names), all_feature_names

# =============================================================================
# 6. EDA PLOTS
# =============================================================================

def plot_class_imbalance(train_df: pd.DataFrame) -> None:
    """
    Plot a side-by-side count bar and pie chart showing the churn class split.

    Parameters
    ----------
    train_df : Training DataFrame containing an 'Exited' column.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    palette   = sns.color_palette("muted")

    counts = train_df["Exited"].value_counts()
    axes[0].bar(["Retained", "Churned"], counts.values,
                color=[palette[0], palette[3]], alpha=0.8)
    axes[0].set_title("Customer Count by Churn Status")
    axes[0].set_ylabel("Count")

    pcts = train_df["Exited"].value_counts(normalize=True) * 100
    axes[1].pie(pcts, labels=["Retained", "Churned"],
                colors=[palette[0], palette[3]], autopct="%1.1f%%")
    axes[1].set_title("Churn Proportion")

    plt.suptitle("Class Imbalance Overview", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

def plot_kde_by_churn(train_df: pd.DataFrame, cols: list[str]) -> None:
    """
    Plot KDE distributions of numeric columns split by churn status.

    Parameters
    ----------
    train_df : Training DataFrame with an 'Exited' column.
    cols     : List of numeric column names to plot (one subplot each).
    """
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        sns.kdeplot(data=train_df, x=col, hue="Exited",
                    fill=True, common_norm=False, ax=ax)
        ax.set_title(f"{col} Distribution by Churn")
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ["Retained (0)", "Churned (1)"])

    plt.tight_layout()
    plt.show()

def plot_churn_rate_bar(
    df: pd.DataFrame,
    group_col: str,
    x_labels: list[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
) -> None:
    """
    Plot churn rate (%) per category in group_col with an overall average line.

    Parameters
    ----------
    df        : DataFrame with group_col and 'Exited' columns.
    group_col : Column to group by (e.g. 'Geography', 'NumOfProducts').
    x_labels  : Human-readable tick labels. Defaults to the category values.
    title     : Plot title. Defaults to f'Churn Rate by {group_col}'.
    xlabel    : Optional x-axis label.
    """
    churn_rate = df.groupby(group_col)["Exited"].mean() * 100
    x_ticks    = x_labels if x_labels is not None else churn_rate.index.astype(str)
    palette    = sns.color_palette("muted")

    fig, ax = plt.subplots(figsize=(8, 4))
    bars    = ax.bar(x_ticks, churn_rate.values, color=palette[0], alpha=0.8)

    ax.axhline(df["Exited"].mean() * 100, color="grey",
               linestyle="--", lw=1.5, label="Overall average")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title(title or f"Churn Rate by {group_col}")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.legend()

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{bar.get_height():.1f}%",
            ha="center", fontsize=10,
        )

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(train_df: pd.DataFrame, threshold: float = 0.7) -> None:
    """
    Plot a lower-triangle correlation heatmap and print any high-correlation pairs.

    Parameters
    ----------
    train_df  : Training DataFrame (numeric columns extracted automatically).
    threshold : Correlation magnitude above which pairs are flagged. Default 0.7.
    """
    numeric_cols = train_df.select_dtypes(include="number")
    corr_matrix = numeric_cols.corr()
    mask         = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(numeric_cols.corr(), mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    corr_matrix = corr_matrix.abs()
    high_corr   = [
        (c1, c2, corr_matrix.loc[c1, c2])
        for c1 in corr_matrix.columns
        for c2 in corr_matrix.columns
        if c1 < c2 and corr_matrix.loc[c1, c2] > threshold
    ]

    if high_corr:
        print(f"High correlations (>{threshold}):")
        for c1, c2, val in high_corr:
            print(f"   {c1} <> {c2} : {val:.3f}")
    else:
        print("No redundant features found.")

# =============================================================================
# 7. EVALUATION PLOTS
# =============================================================================

def plot_threshold_curve(
    thresholds: np.ndarray,
    precisions: np.ndarray,
    recalls: np.ndarray,
    f1_arr: np.ndarray,
    best_threshold: float,
    recall_floor: float,
) -> None:
    """
    Plot Precision, Recall, and F1 against decision threshold, marking the
    chosen operating point and the recall floor constraint.

    Parameters
    ----------
    thresholds     : Array of threshold values from precision_recall_curve.
    precisions     : Precision values (aligned to thresholds).
    recalls        : Recall values (aligned to thresholds).
    f1_arr         : F1 values (aligned to thresholds).
    best_threshold : The chosen operating threshold to mark.
    recall_floor   : Minimum acceptable recall — shown as a horizontal guideline.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, f1_arr[:-1],     label="F1",        color="#4C9BE8", lw=2)
    ax.plot(thresholds, precisions[:-1], label="Precision", color="#F0B429", lw=2)
    ax.plot(thresholds, recalls[:-1],    label="Recall",    color="#E8694C", lw=2)
    ax.axvline(best_threshold, color="grey", linestyle="--", lw=1.5,
               label=f"Chosen threshold = {best_threshold:.3f}")
    ax.axhline(recall_floor, color="green", linestyle=":", lw=1.5,
               label=f"Recall floor = {recall_floor}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold vs Precision / Recall / F1", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred_test: np.ndarray,
    best_threshold: float,
    model_name: str,
) -> None:
    """
    Plot the confusion matrix at the chosen decision threshold.

    Parameters
    ----------
    y_test         : True test labels.
    y_pred_test    : Binary predictions at the chosen threshold.
    best_threshold : Operating threshold — shown in the title for context.
    model_name     : Name label for the plot title.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred_test),
        display_labels=["Retained", "Churned"],
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"Confusion Matrix — {model_name}\n(threshold = {best_threshold:.3f})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_roc_curve(
    y_test: pd.Series,
    y_probas_test: np.ndarray,
    auc: float,
    model_name: str,
) -> None:
    """
    Plot the ROC curve with AUC annotated in the legend.

    Parameters
    ----------
    y_test        : True test labels.
    y_probas_test : Predicted probabilities for the positive class.
    auc           : ROC-AUC score.
    model_name    : Name label for the plot legend.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_test, y_probas_test, ax=ax, color="#4C9BE8",
        name=f"{model_name} (AUC = {auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random baseline")
    ax.set_title("ROC Curve", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_pr_curve(
    y_test: pd.Series,
    y_probas_test: np.ndarray,
    best_threshold: float,
    recall: float,
    precision: float,
    f1: float,
) -> None:
    """
    Plot the Precision-Recall curve with the operating point marked.

    Parameters
    ----------
    y_test         : True test labels.
    y_probas_test  : Predicted probabilities for the positive class.
    best_threshold : Operating threshold used for the operating-point marker.
    recall         : Recall at the operating point.
    precision      : Precision at the operating point.
    f1             : F1 at the operating point.
    """
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_probas_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_curve, prec_curve, color="#E8694C", lw=2, label=f"F1 = {f1:.3f}")
    ax.axhline(0.50, color="grey",  linestyle="--", label="Precision floor = 0.50")
    ax.axvline(0.60, color="green", linestyle="--", label="Recall floor = 0.60")
    ax.scatter(
        [recall], [precision], color="black", zorder=5, s=80,
        label=f"Operating point (t = {best_threshold:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_calibration_curve(
    y_test: pd.Series,
    y_probas_test: np.ndarray,
    model_name: str,
    n_bins: int = 10,
) -> None:
    """
    Plot a reliability (calibration) curve to assess whether predicted
    probabilities match observed churn rates.

    A perfectly calibrated model follows the diagonal. Points above the
    diagonal mean the model is under-confident; below means over-confident.
    This is critical to validate before showing probability scores in the
    Streamlit app.

    Parameters
    ----------
    y_test        : True test labels.
    y_probas_test : Predicted probabilities for the positive class.
    model_name    : Name label for the plot legend.
    n_bins        : Number of calibration bins. Default 10.
    """
    prob_true, prob_pred = calibration_curve(y_test, y_probas_test, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, marker="o", color="#4C9BE8",
            lw=2, label=f"{model_name}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.1, color="#4C9BE8")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Actual Churn Rate)")
    ax.set_title("Calibration Curve — Are Probabilities Trustworthy?",
                 fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_error_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred_test: np.ndarray,
    y_probas_test: np.ndarray,
) -> pd.DataFrame:
    """
    Analyse false negatives and false positives by demographic breakdown.

    Shows where the model is systematically wrong — which age groups,
    geographies, or product counts it tends to miss or over-flag.
    Returns the error DataFrame for further inspection.

    Parameters
    ----------
    X_test        : Raw test features (pre-transformation).
    y_test        : True test labels.
    y_pred_test   : Binary predictions at the chosen threshold.
    y_probas_test : Predicted probabilities for the positive class.

    Returns
    -------
    error_df : DataFrame of misclassified customers with error type labelled.
    """
    results_df = X_test.copy().reset_index(drop=True)
    results_df["actual"]      = y_test.values
    results_df["predicted"]   = y_pred_test
    results_df["probability"] = y_probas_test
    results_df["error_type"]  = "Correct"
    results_df.loc[(results_df["actual"] == 1) & (results_df["predicted"] == 0), "error_type"] = "False Negative"
    results_df.loc[(results_df["actual"] == 0) & (results_df["predicted"] == 1), "error_type"] = "False Positive"

    error_df = results_df[results_df["error_type"] != "Correct"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette   = {"False Negative": "#E8694C", "False Positive": "#F0B429"}

    for ax, col in zip(axes, ["Geography", "AgeGroup", "NumOfProducts"]):
        plot_col = col
        if col == "AgeGroup":
            plot_df = error_df.copy()
            plot_df["AgeGroup"] = pd.cut(
                plot_df["Age"], bins=[0, 35, 55, 100],
                labels=["Young", "Middle", "Senior"]
            ).astype(str)
            plot_col = "AgeGroup"
        else:
            plot_df = error_df

        counts = (
            plot_df.groupby([plot_col, "error_type"])
            .size()
            .reset_index(name="count")
        )
        sns.barplot(data=counts, x=plot_col, y="count",
                    hue="error_type", palette=palette, ax=ax)
        ax.set_title(f"Errors by {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend(title="Error Type", fontsize=8)

    plt.suptitle("Error Analysis — Where Is the Model Getting It Wrong?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    n_fn = (error_df["error_type"] == "False Negative").sum()
    n_fp = (error_df["error_type"] == "False Positive").sum()
    print(f"\nTotal errors       : {len(error_df):,} / {len(results_df):,} ({len(error_df)/len(results_df):.1%})")
    print(f"False Negatives    : {n_fn}  (missed churners — cost: lost customer)")
    print(f"False Positives    : {n_fp}  (wrong alarms   — cost: wasted retention spend)")

    return error_df


# =============================================================================
# 8. SHAP PLOTS
# =============================================================================

def plot_shap_summary(
    sv: np.ndarray,
    X_test_df: pd.DataFrame,
    save_dir: str = "reports/figures",
) -> None:
    """
    Plot the SHAP beeswarm summary and bar importance chart, and print
    the top 10 features by mean absolute SHAP value.

    Parameters
    ----------
    sv        : SHAP values array for class 1, shape (n_samples, n_features).
    X_test_df : Transformed test set as a named DataFrame.
    save_dir  : Directory to save figures. Created if it doesn't exist.
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    shap.summary_plot(sv, X_test_df, show=False)
    plt.title("SHAP Summary — Feature Impact on Churn", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_summary.png", bbox_inches="tight")
    plt.show()

    plt.figure()
    shap.summary_plot(sv, X_test_df, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_bar.png", bbox_inches="tight")
    plt.show()

    mean_shap = pd.Series(np.abs(sv).mean(axis=0), index=X_test_df.columns)
    print("\nTop 10 features by mean |SHAP|:")
    print(mean_shap.sort_values(ascending=False).head(10).to_string())


def plot_shap_waterfall(
    explainer,
    customer: pd.Series,
    all_feature_names: list[str],
    y_probas_test: np.ndarray,
    y_test,
    idx: int,
) -> None:
    """
    Plot a SHAP waterfall chart explaining a single prediction.
 
    Shows how each feature pushes the model's output above or below the
    base rate for one specific customer, making the prediction auditable.
 
    Parameters
    ----------
    explainer         : Fitted SHAP explainer.
    customer          : Single-row pd.Series of transformed features.
    all_feature_names : Feature names matching the transformed columns.
    y_probas_test     : Predicted probabilities for the full test set.
    y_test            : True test labels.
    idx               : Index of the customer in the test set.
    """
    s_vals   = explainer.shap_values(customer.values.reshape(1, -1))
    wf_values = get_shap_values_class1(s_vals, index=0)
 
    exp_vals = explainer.expected_value
    if isinstance(exp_vals, (list, np.ndarray)) and np.atleast_1d(exp_vals).shape[0] > 1:
        wf_base = float(np.atleast_1d(exp_vals)[1])
    else:
        wf_base = float(np.atleast_1d(exp_vals)[0])
 
    print(f"  Customer index        : {idx}")
    print(f"  Predicted probability : {y_probas_test[idx]:.2%}")
    print(f"  Actual label          : {'CHURN' if y_test.iloc[idx] == 1 else 'RETAIN'}")
 
    shap.plots.waterfall(
        shap.Explanation(
            values        = wf_values,
            base_values   = wf_base,
            data          = customer.values,
            feature_names = all_feature_names,
        ),
        show=False,
    )
    plt.title("SHAP Waterfall — Why this customer was flagged",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/figures/shap_waterfall.png", bbox_inches="tight")
    plt.show()
 

# =============================================================================
# 9. PERSISTENCE
# =============================================================================

def save_pipeline_and_results(
    final_pipeline: Pipeline,
    model_name: str,
    best_threshold: float,
    y_test: pd.Series,
    y_pred_test: np.ndarray,
    y_probas_test: np.ndarray,
    model_dir: str = "models",
) -> None:
    """
    Serialise the final pipeline and decision threshold to disk, and save
    test predictions as CSV.

    Both the pipeline and the threshold are required by the Streamlit app
    at inference time — they are saved together so nothing is lost between
    the notebook run and deployment.

    Parameters
    ----------
    final_pipeline : Fitted sklearn Pipeline.
    model_name     : Used to name the saved files.
    best_threshold : Optimal decision threshold from find_optimal_threshold.
                     Saved as a separate .joblib so the app can load it
                     independently of the pipeline.
    y_test         : True test labels.
    y_pred_test    : Binary predictions at the chosen threshold.
    y_probas_test  : Predicted probabilities for the positive class.
    model_dir      : Directory for saved files. Created if it doesn't exist.
    """
    os.makedirs(model_dir, exist_ok=True)

    model_path = f"{model_dir}/{model_name}_final_pipeline.joblib"
    joblib.dump(final_pipeline, model_path)
    print(f"Pipeline saved  -> {model_path}")

    threshold_path = f"{model_dir}/{model_name}_threshold.joblib"
    joblib.dump(best_threshold, threshold_path)
    print(f"Threshold saved -> {threshold_path}  (value: {best_threshold:.4f})")

    results_df = pd.DataFrame({
        "actual"     : y_test.values,
        "predicted"  : y_pred_test,
        "probability": y_probas_test,
        "correct"    : y_test.values == y_pred_test,
    })
    csv_path = f"{model_dir}/test_predictions.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"Predictions saved → {csv_path}")
    print(f"   Accuracy : {results_df['correct'].mean():.2%}")
    print(f"   Rows     : {len(results_df):,}")
