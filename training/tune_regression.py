"""
tune_regression.py — Optuna Bayesian Optimization for TSB Regressors

Runs 100 trials to find LightGBM hyperparameters minimizing MAE on the neonatal dataset.
"""

import logging
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.INFO)

DATA_PATH = "__data__/neo/out/training.csv"
TSB_COL = "blood_mg_dl"
SHAP_CUTOFF = 0.01
N_TRIALS = 100

df = pd.read_csv(DATA_PATH)
df = df[df[TSB_COL] > 0.5].copy()

df["postnatal_age_hours"] = df["postnatal_age_days"] * 24
df["zone1_to_zone3_Y_ratio"] = df["zone1_Y_mean"] / (df["zone3_Y_mean"] + 1e-5)
df["zone1_to_zone3_b_ratio"] = df["zone1_Lab_b_mean"] / (df["zone3_Lab_b_mean"] + 1e-5)

BASE_COLOR = [c for c in df.columns if c.startswith("zone") and "ratio" not in c]
ENG_COLOR = ["zone1_to_zone3_Y_ratio", "zone1_to_zone3_b_ratio"]
META_FEATURES = ["gestational_age", "postnatal_age_hours", "weight"]
FEATURES = BASE_COLOR + ENG_COLOR + META_FEATURES

patients = df[~df["is_augmented"]]["patient_id"].unique()
train_p, temp_p = train_test_split(patients, test_size=0.30, random_state=42)
val_p, _ = train_test_split(temp_p, test_size=0.50, random_state=42)

train_df = df[df["patient_id"].isin(train_p)].copy()
val_df = df[df["patient_id"].isin(val_p) & ~df["is_augmented"]].copy()

logging.info("Running baseline model for SHAP feature selection...")
baseline_model = lgb.LGBMRegressor(boosting_type="gbdt", random_state=42, verbose=-1)
baseline_model.fit(train_df[FEATURES], train_df[TSB_COL])

explainer = shap.TreeExplainer(baseline_model)
sv = explainer.shap_values(val_df[FEATURES])
if isinstance(sv, list):
    sv = sv[1]
mean_shap = np.abs(sv).mean(axis=0)
selected_features = [
    f for f, s in zip(FEATURES, mean_shap) if s >= SHAP_CUTOFF * mean_shap.max()
]
logging.info("%d features selected for tuning.", len(selected_features))

X_train = train_df[selected_features]
y_train = train_df[TSB_COL]
X_val = val_df[selected_features]
y_val = val_df[TSB_COL]


def objective(trial: optuna.Trial) -> float:
    params = {
        "boosting_type": "gbdt",
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 120),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "random_state": 42,
        "verbose": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return float(mean_absolute_error(y_val, model.predict(X_val)))  # type: ignore


logging.info("Starting Optuna optimization (%d trials)...", N_TRIALS)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS)

logging.info("Best validation MAE: %.4f mg/dL", study.best_value)
logging.info("Best hyperparameters:")
for key, value in study.best_params.items():
    logging.info("  '%s': %s,", key, value)
