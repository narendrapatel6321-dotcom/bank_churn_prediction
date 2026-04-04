
from sklearn.pipeline import Pipeline
import joblib
import os
import pandas as pd
import numpy as np


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

    print(f"Predictions saved -> {csv_path}")
    print(f"   Accuracy : {results_df['correct'].mean():.2%}")
    print(f"   Rows     : {len(results_df):,}")
