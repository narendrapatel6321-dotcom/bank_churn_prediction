import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

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

def get_shap_values_class1(shap_values, index: int | None = None,) -> np.ndarray:
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
    save_path = str | None = None,
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
    save_path         : Path to save plot.
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
    if save_path:         
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
