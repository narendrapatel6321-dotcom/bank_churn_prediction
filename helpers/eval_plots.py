
from __future__ import annotations
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration    import calibration_curve
from sklearn.metrics        import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_curve,
    )
warnings.filterwarnings("ignore")


def plot_threshold_curve(
    thresholds: np.ndarray,
    precisions: np.ndarray,
    recalls: np.ndarray,
    f1_arr: np.ndarray,
    best_threshold: float,
    recall_floor: float,
    save_path : str | None = None
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
    save_path      : Path to save plot.
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
    if save_path:              
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred_test: np.ndarray,
    best_threshold: float,
    model_name: str,
    save_path : str | None = None
) -> None:
    """
    Plot the confusion matrix at the chosen decision threshold.

    Parameters
    ----------
    y_test         : True test labels.
    y_pred_test    : Binary predictions at the chosen threshold.
    best_threshold : Operating threshold — shown in the title for context.
    model_name     : Name label for the plot title.
    save_path      : Path to save plot.
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
    if save_path:              
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curve(
    y_test: pd.Series,
    y_probas_test: np.ndarray,
    auc: float,
    model_name: str,
    save_path : str | None = None
) -> None:
    """
    Plot the ROC curve with AUC annotated in the legend.

    Parameters
    ----------
    y_test        : True test labels.
    y_probas_test : Predicted probabilities for the positive class.
    auc           : ROC-AUC score.
    model_name    : Name label for the plot legend.
    save_path     : Path to save plot.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_test, y_probas_test, ax=ax, color="#4C9BE8",
        name=f"{model_name}",
    )
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random baseline")
    ax.set_title("ROC Curve", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:              
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_pr_curve(
    y_test: pd.Series,
    y_probas_test: np.ndarray,
    best_threshold: float,
    recall: float,
    precision: float,
    f1: float,
    save_path : str | None = None
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
    save_path      : Path to save plot.
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
    if save_path:              
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_calibration_curve(
    y_test: pd.Series,
    y_probas_test: np.ndarray,
    model_name: str,
    n_bins: int = 10,
    save_path : str | None = None
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
    save_path     : Path to save plot.
    """
    prob_true, prob_pred = calibration_curve(y_test, y_probas_test, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, marker="o", color="#4C9BE8",
            lw=2, label=f"{model_name}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.1, color="#4C9BE8")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives ")
    ax.set_title("Calibration Curve ",
                 fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save_path:              
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_error_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred_test: np.ndarray,
    y_probas_test: np.ndarray,
    save_path : str | None = None
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
    save_path     : Path to save plot.

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

    plt.suptitle("Error Analysis",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:              
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    n_fn = (error_df["error_type"] == "False Negative").sum()
    n_fp = (error_df["error_type"] == "False Positive").sum()
    print(f"\nTotal errors       : {len(error_df):,} / {len(results_df):,} ({len(error_df)/len(results_df):.1%})")
    print(f"False Negatives    : {n_fn}  (missed churners) ")
    print(f"False Positives    : {n_fp}  (wrong alarms) ")

    return error_df


