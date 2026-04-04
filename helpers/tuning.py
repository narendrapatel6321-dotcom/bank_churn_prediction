import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from typing import Callable
from helper.feature_engineering import build_pipeline
# =============================================================================
# 1. HYPERPARAMETER TUNING
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
                f"  Trial {trial.number:>3} | Score: {trial.value:.4f} | "
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
    score_fn: Callable | None = None,
) -> Callable:
    """
    Build a generic Optuna objective from a model factory and a param space.

    This keeps helpers.py experiment-agnostic — all model choices, search
    spaces, and scoring logic live in the notebook, not here.

    Parameters
    ----------
    model_fn             : Callable(**params) → unfitted sklearn estimator.
                           e.g. lambda **p: LGBMClassifier(**p, verbosity=-1)
    param_space          : Dict of {param_name: Callable(trial) → value}.
                           e.g. {"n_estimators": lambda t: t.suggest_int("n_estimators", 100, 1000)}
    X_train              : Training features.
    y_train              : Training labels.
    skf                  : StratifiedKFold splitter.
    best_strategy        : 'SMOTE', 'ADASYN', or 'class_weight'.
    num_features         : Numeric feature list — passed to build_pipeline.
    cat_features         : Categorical feature list — passed to build_pipeline.
    passthrough_features : Passthrough feature list — passed to build_pipeline.
    extra_features       : Optional list of feature engineering callables.
    score_fn             : Callable(y_true, y_pred, y_proba) → float, or None.
                           If provided, uses cross_val_predict to generate OOF
                           predictions and passes them to score_fn.
                           If None, falls back to fast cross_val_score with F1.

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
            use_adasyn           = (best_strategy == "ADASYN"),
            extra_features       = extra_features,
        )

        if score_fn is not None:
            y_proba = cross_val_predict(
                pipe, X_train, y_train, cv=skf,
                method="predict_proba", n_jobs=-1,
            )[:, 1]
            y_pred  = (y_proba >= 0.5).astype(int)
            return score_fn(y_train, y_pred, y_proba)
        else:
            return cross_val_score(
                pipe, X_train, y_train, cv=skf,
                scoring="f1", n_jobs=-1,
            ).mean()

    return objective
   