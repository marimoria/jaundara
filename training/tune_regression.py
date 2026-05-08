"""
tune_regression.py — Optuna Bayesian Optimization for TSB Regressors
====================================================================
Runs 100 trials to find the absolute best LightGBM hyperparameters 
for minimizing Mean Absolute Error (MAE) on the neonatal dataset.
"""

import warnings
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.INFO)

DATA_PATH = "__data__/neo/out/training.csv"
TSB_COL = "blood_mg_dl"
SHAP_CUTOFF = 0.01

print("Loading and cleaning data...")
# 1. Load Data
df = pd.read_csv(DATA_PATH)

# 2. Clean Target (Crucial for Regression)
df = df[df[TSB_COL] > 0.5].copy()

# 3. Feature Engineering
df['postnatal_age_hours'] = df['postnatal_age_days'] * 24
df['zone1_to_zone3_Y_ratio'] = df['zone1_Y_mean'] / (df['zone3_Y_mean'] + 1e-5)
df['zone1_to_zone3_b_ratio'] = df['zone1_Lab_b_mean'] / (df['zone3_Lab_b_mean'] + 1e-5)

BASE_COLOR = [c for c in df.columns if c.startswith("zone") and "ratio" not in c]
ENG_COLOR = ['zone1_to_zone3_Y_ratio', 'zone1_to_zone3_b_ratio']
META_FEATURES = ["gestational_age", "postnatal_age_hours", "weight"]

# Let's tune Model 2A (Color + Meta)
FEATURES = BASE_COLOR + ENG_COLOR + META_FEATURES

# 4. Split Data (Matching your pipeline)
patients = df[~df["is_augmented"]]["patient_id"].unique()
train_p, temp_p = train_test_split(patients, test_size=0.30, random_state=42)
val_p, _ = train_test_split(temp_p, test_size=0.50, random_state=42)

train_df = df[df["patient_id"].isin(train_p)].copy()
val_df = df[df["patient_id"].isin(val_p) & ~df["is_augmented"]].copy()

# 5. Baseline SHAP Selection (Get the best features first)
print("Running baseline model to extract SHAP features...")
baseline_params = {"boosting_type": "gbdt", "random_state": 42, "verbose": -1}
baseline_model = lgb.LGBMRegressor(**baseline_params).fit(train_df[FEATURES], train_df[TSB_COL])

explainer = shap.TreeExplainer(baseline_model)
sv = explainer.shap_values(val_df[FEATURES])
if isinstance(sv, list): sv = sv[1]
mean_shap = np.abs(sv).mean(axis=0)
cutoff = SHAP_CUTOFF * mean_shap.max()
selected_features = [f for f, s in zip(FEATURES, mean_shap) if s >= cutoff]

print(f"Selected {len(selected_features)} features for tuning.")

X_train = train_df[selected_features]
y_train = train_df[TSB_COL]
X_val = val_df[selected_features]
y_val = val_df[TSB_COL]

# 6. Optuna Objective Function
def objective(trial):
    # These are the exact parameters Optuna will tweak mathematically
    param = {
        "boosting_type": "gbdt",
        "objective": "regression_l1",  # Optimizing strictly for MAE
        "metric": "mae",
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 120),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "random_state": 42,
        "verbose": -1
    }

    model = lgb.LGBMRegressor(**param)
    
    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # Predict and evaluate
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds) # type: ignore
    
    return mae

# 7. Run the Optimization
print("\n--- Starting Optuna Optimization (100 Trials) ---")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print("\n==================================================")
print("OPTIMIZATION COMPLETE")
print(f"Best Validation MAE: {study.best_value:.4f} mg/dL")
print("Best Hyperparameters to paste into your main script:")
for key, value in study.best_params.items():
    print(f"    '{key}': {value},")
print("==================================================")