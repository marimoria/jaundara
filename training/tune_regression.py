"""
tune_regression.py — Optuna Bayesian Optimization for TSB Regressor (Model 2)

Runs N_TRIALS trials to find LightGBM hyperparameters minimizing MAE.

Feature set mirrors train_models.py exactly: all features with |Spearman r| >= 0.10
on original rows, plus engineered cross-zone and gradient features.
SHAP selection uses the cumulative-95% method before the Optuna search,
so hyperparameters are tuned on the same reduced feature set used in production.
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

DATA_PATH  = "__data__/neo/out/training_engineered.csv"
TSB_COL    = "blood_mg_dl"
N_TRIALS   = 100

# Must match train_models.py's SHAP_CUMULATIVE_THRESHOLD
SHAP_CUMULATIVE_THRESHOLD = 0.95


# ── feature engineering (mirrors train_models.py:build_features) ──────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    zones = ["zone1", "zone2", "zone3"]

    h_cols = [f"{z}_H_mean" for z in zones]
    out["mean_zones_H_mean"] = out[h_cols].mean(axis=1)

    for ch in ["Lab_b_mean", "Cb_mean", "H_mean"]:
        out[f"grad_z3z1_{ch}"] = out[f"zone3_{ch}"] - out[f"zone1_{ch}"]

    return out


def _define_feature_sets(df: pd.DataFrame) -> list:
    EXCLUDE_PATTERNS = (
        "_R_mean", "_G_mean", "_Y_mean",
        "_Lab_L_mean", "_Lab_a_mean",
        "_ITA", "log1p_",
    )
    EXCLUDE_EXACT = {"weight", "zone2_Cr_mean"}

    def _keep(col: str) -> bool:
        if col in EXCLUDE_EXACT:
            return False
        for pat in EXCLUDE_PATTERNS:
            if pat in col:
                return False
        return True

    color_features = [
        c for c in df.columns
        if (c.startswith("zone") or c.startswith("mean_zones") or
            c.startswith("grad_z3z1"))
        and _keep(c)
    ]
    meta_features = [
        f for f in ["gestational_age", "postnatal_age_days"]
        if f in df.columns
    ]
    return color_features + meta_features


# ── load + split ──────────────────────────────────────────────────────────────

df_raw = pd.read_csv(DATA_PATH)
df = build_features(df_raw)
df = df[df[TSB_COL] > 0.5].copy()

ALL_FEATURES = _define_feature_sets(df)

logging.info("Features: total=%d", len(ALL_FEATURES))

patients = df[~df["is_augmented"]]["patient_id"].unique()
train_p, temp_p = train_test_split(patients, test_size=0.30, random_state=42)
val_p, _        = train_test_split(temp_p,   test_size=0.50, random_state=42)

train_df = df[df["patient_id"].isin(train_p)].copy()
val_df   = df[df["patient_id"].isin(val_p) & ~df["is_augmented"]].copy()

logging.info("train rows=%d  val rows=%d (original only)", len(train_df), len(val_df))


# ── SHAP selection (cumulative 95%) ───────────────────────────────────────────

logging.info("Running baseline model for SHAP feature selection...")
baseline = lgb.LGBMRegressor(boosting_type="gbdt", random_state=42, verbose=-1)
baseline.fit(train_df[ALL_FEATURES], train_df[TSB_COL])

sv = shap.TreeExplainer(baseline).shap_values(val_df[ALL_FEATURES])
if isinstance(sv, list):
    sv = sv[1]

mean_shap  = np.abs(sv).mean(axis=0)
total_shap = mean_shap.sum()
order      = np.argsort(mean_shap)[::-1]
cumulative = np.cumsum(mean_shap[order]) / total_shap
n_keep     = int(np.searchsorted(cumulative, SHAP_CUMULATIVE_THRESHOLD)) + 1

selected_features = [ALL_FEATURES[i] for i in order[:n_keep]]
dropped_features  = [ALL_FEATURES[i] for i in order[n_keep:]]

logging.info("SHAP selection: kept %d / %d features (cumulative %.0f%% SHAP mass)",
             len(selected_features), len(ALL_FEATURES),
             SHAP_CUMULATIVE_THRESHOLD * 100)
logging.info("Top feature mean|SHAP|=%.4f  total mass=%.4f",
             mean_shap.max(), total_shap)
logging.info("Dropped: %s", dropped_features)

# log full SHAP table
shap_table = sorted(zip(ALL_FEATURES, mean_shap), key=lambda x: -x[1])
logging.info("%-45s %10s %10s %s", "feature", "mean|SHAP|", "% of top", "kept")
for feat, ms in shap_table:
    kept = "✓" if feat in selected_features else "✗"
    logging.info("  %-43s %10.4f %9.1f%%  %s",
                 feat, ms, ms / mean_shap.max() * 100, kept)

X_train = train_df[selected_features]
y_train = train_df[TSB_COL]
X_val   = val_df[selected_features]
y_val   = val_df[TSB_COL]


# ── Optuna objective ──────────────────────────────────────────────────────────

def objective(trial: optuna.Trial) -> float:
    params = {
        "boosting_type":    "gbdt",
        "objective":        "regression_l1",
        "metric":           "mae",
        "n_estimators":     trial.suggest_int("n_estimators", 300, 2000, step=100),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 15, 120),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "random_state":     42,
        "verbose":          -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return float(mean_absolute_error(y_val, model.predict(X_val)))  # type: ignore


# ── run ───────────────────────────────────────────────────────────────────────

logging.info("Starting Optuna optimization (%d trials)...", N_TRIALS)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS)

logging.info("Best validation MAE: %.4f mg/dL", study.best_value)
logging.info("Best hyperparameters (paste into LGB_REGRESSION_PARAMS in train_models.py):")
for key, value in study.best_params.items():
    logging.info("  '%s': %s,", key, value)