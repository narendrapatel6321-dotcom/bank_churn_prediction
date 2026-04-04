import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve

# =============================================================================
# 1. THRESHOLD SELECTION
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
 