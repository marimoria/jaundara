"""
train_models_v4.py — Neonatal Jaundice Detection (Plan v4)
===========================================================
ARCHITECTURE CHANGE vs Plan v3
────────────────────────────────────────────────────────────
Plan v3 used a unified 3-class classifier:
  Model 2A/2B → predicts Normal / Mild / Severe (class labels)
  Problem: the Mild/Severe boundary is a hard threshold learned by the model,
           not the clinically meaningful Bhutani Nomogram curve.

Plan v4 replaces Model 2 with a LightGBM REGRESSOR:
  Model 1A/1B  → binary detection (unchanged): jaundiced vs normal
  Model 2A/2B  → TSB regression: predicts blood_mg_dl as a continuous value

Inference pipeline:
  1. Model 1A/2B says "Jaundice Detected" (binary gate, fast).
  2. Model 2A/2B predicts "Estimated Bilirubin = 14.2 mg/dL" (the number).
  3. Flutter app receives (tsb_estimate, postnatal_age_hours), evaluates the
     Bhutani Nomogram thresholds in code, and surfaces the exact risk zone
     and recommended action to the mother.

Why regression beats classification here:
  • The model's job is to read color → estimate a continuous quantity.
    Phototherapy thresholds depend on BOTH TSB AND age — the model cannot
    learn that interaction without age as a feature, but the app always has it.
  • Regression gives the app a real number it can compare against any
    nomogram curve, without baking a fixed threshold into the model weights.
  • MAE / RMSE are interpretable in clinical terms (± mg/dL).

Models saved to: ./__models__/
  model_1A.pkl  — Binary detection (color + metadata)    [unchanged]
  model_1B.pkl  — Binary detection (color only)           [unchanged]
  model_2A.pkl  — TSB regressor    (color + metadata)    [NEW]
  model_2B.pkl  — TSB regressor    (color only)          [NEW]

Regression is trained on jaundiced patients only (blood_mg_dl is meaningful
only when jaundice_label == 1).  Model 1 handles the detection gate.

Usage:
  python train_models_v4.py
  python train_models_v4.py --log
  python train_models_v4.py --log --log-dir path/to/dir
"""

import os
import sys
import json
import pickle
import logging
import argparse
import warnings
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
# CLI ARGUMENTS
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train neonatal jaundice models (Plan v4).")
parser.add_argument("--log", action="store_true", default=False,
                    help="Write detailed logs and artifacts to __data__/models_log/<timestamp>/")
parser.add_argument("--log-dir", type=str, default=None,
                    help="Override the log output directory (implies --log).")
args = parser.parse_args()
LOGGING_ENABLED = args.log or (args.log_dir is not None)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH    = "__data__/neo/out/training.csv"
MODELS_DIR   = "__models__"
DETECT_LABEL = "jaundice_label"
TSB_COL      = "blood_mg_dl"       # regression target
SHAP_CUTOFF  = 0.01                # drop features below 1% of top SHAP

# Binary detection models (Model 1A / 1B) — identical to v3
LGB_BINARY_PARAMS = dict(
    boosting_type="gbdt",
    objective="binary",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# TSB regression models (Model 2A / 2B)
LGB_REGRESSION_PARAMS = dict(
    boosting_type="gbdt", 
    objective="regression_l1",
    metric="mae",
    n_estimators=1400,
    learning_rate=0.05093543755558586,
    num_leaves=15,
    min_child_samples=13,
    subsample=0.581256713045798,
    colsample_bytree=0.5647836560380153,
    random_state=42, 
    verbose=-1,
)

os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if LOGGING_ENABLED:
    LOG_DIR = Path(args.log_dir) if args.log_dir else Path("__data__") / "models_log" / RUN_TIMESTAMP
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = LOG_DIR / "run.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("train_models_v4")
    log.info(f"Log directory  : {LOG_DIR.resolve()}")
    log.info(f"Run timestamp  : {RUN_TIMESTAMP}")
    log.info(f"LGB_BINARY_PARAMS    : {json.dumps(LGB_BINARY_PARAMS, indent=2)}")
    log.info(f"LGB_REGRESSION_PARAMS: {json.dumps(LGB_REGRESSION_PARAMS, indent=2)}")
    log.info(f"SHAP_CUTOFF    : {SHAP_CUTOFF}")
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("train_models_v4")
    LOG_DIR = None


def _save_json(obj, filename: str):
    if not LOGGING_ENABLED:
        return
    out = LOG_DIR / filename  # type: ignore
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    log.debug(f"  [artifact] Saved JSON → {out.name}")


def _save_csv(df: pd.DataFrame, filename: str):
    if not LOGGING_ENABLED:
        return
    out = LOG_DIR / filename  # type: ignore
    df.to_csv(out, index=False)
    log.debug(f"  [artifact] Saved CSV  → {out.name}")


# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────
log.info("=" * 70)
log.info("STEP 1 — LOADING DATA")
log.info("=" * 70)

df = pd.read_csv(DATA_PATH)
log.info(f"  Total rows     : {len(df)}")
log.info(f"  Total columns  : {len(df.columns)}")

vc_detect = df[DETECT_LABEL].value_counts()
log.info(f"  Detection label distribution:")
for val, cnt in vc_detect.items():
    log.info(f"    {val}: {cnt}  ({cnt/len(df)*100:.1f}%)")

COLOR_FEATURES = [c for c in df.columns if c.startswith("zone")]
META_FEATURES  = ["gestational_age", "postnatal_age_days", "weight"]
ALL_FEATURES   = COLOR_FEATURES + META_FEATURES

log.info(f"  Color features : {len(COLOR_FEATURES)}")
log.info(f"  Meta  features : {len(META_FEATURES)}")
log.info(f"  Total features : {len(ALL_FEATURES)}")

tsb_stats = df[TSB_COL]
log.info(f"  TSB stats (all patients): min={tsb_stats.min():.2f}  max={tsb_stats.max():.2f}"
         f"  mean={tsb_stats.mean():.2f}  std={tsb_stats.std():.2f}")

# ─────────────────────────────────────────────
# STEP 2 — PATIENT-LEVEL SPLIT (70 / 15 / 15)
# ─────────────────────────────────────────────
log.info("\n" + "─" * 70)
log.info("STEP 2 — PATIENT-LEVEL TRAIN/VAL/TEST SPLIT (70 / 15 / 15)")
log.info("─" * 70)
log.info("  Val and Test use original (non-augmented) rows only.")
log.info("  Train includes augmented rows for all train patients.")

patients        = df[~df["is_augmented"]]["patient_id"].unique()
train_p, temp_p = train_test_split(patients, test_size=0.30, random_state=42)
val_p,  test_p  = train_test_split(temp_p,   test_size=0.50, random_state=42)

train_df = df[df["patient_id"].isin(train_p)].copy()
val_df   = df[df["patient_id"].isin(val_p)  & ~df["is_augmented"]].copy()
test_df  = df[df["patient_id"].isin(test_p) & ~df["is_augmented"]].copy()

log.info(f"  Patients — train: {len(train_p)}  val: {len(val_p)}  test: {len(test_p)}")
log.info(f"  Rows     — train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

# All patients used for regression training (normal babies have bilirubin too)
reg_train_df = train_df
reg_val_df   = val_df
reg_test_df  = test_df

log.info(f"  Regression subsets (all patients):")
log.info(f"    train: {len(reg_train_df)}  val: {len(reg_val_df)}  test: {len(reg_test_df)}")

_save_json(
    {
        "n_patients": {"train": len(train_p), "val": len(val_p), "test": len(test_p)},
        "n_rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "detect_dist": {
            "train": train_df[DETECT_LABEL].value_counts().to_dict(),
            "val":   val_df[DETECT_LABEL].value_counts().to_dict(),
            "test":  test_df[DETECT_LABEL].value_counts().to_dict(),
        },
    },
    "01_split_summary.json",
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _compute_class_weights(y_series: pd.Series) -> np.ndarray:
    """Balanced class weights for binary models."""
    classes   = sorted(y_series.unique())
    n_total   = len(y_series)
    n_classes = len(classes)
    cw = {c: n_total / (n_classes * (y_series == c).sum()) for c in classes}
    return y_series.map(cw).values  # type: ignore


def train_binary_clf(X_tr, y_tr, X_val, y_val, model_tag: str = ""):
    """Train a LightGBM binary classifier with early stopping."""
    log.info(f"    [{model_tag}] Training LGBMClassifier (binary)")
    log.info(f"    [{model_tag}] Train: {X_tr.shape}  pos_rate={y_tr.mean()*100:.1f}%")
    log.info(f"    [{model_tag}] Val:   {X_val.shape}  pos_rate={y_val.mean()*100:.1f}%")

    sample_w = _compute_class_weights(y_tr)
    m = lgb.LGBMClassifier(**LGB_BINARY_PARAMS)  # type: ignore
    m.fit(
        X_tr, y_tr,
        sample_weight=sample_w,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    log.info(f"    [{model_tag}] Best iteration: {m.best_iteration_}")

    fi_gain = dict(zip(X_tr.columns, m.booster_.feature_importance(importance_type="gain")))
    log.info(f"    [{model_tag}] Top-10 feature importance (gain):")
    for feat, val in sorted(fi_gain.items(), key=lambda x: -x[1])[:10]:
        log.info(f"        {feat:<35s}  gain={val:.4f}")

    if LOGGING_ENABLED:
        _save_json({
            "model_tag": model_tag, "best_iteration": m.best_iteration_,
            "train_shape": list(X_tr.shape), "val_shape": list(X_val.shape),
            "feature_importance_gain": fi_gain, "features": list(X_tr.columns),
        }, f"{model_tag}_training_detail.json")

    return m


def train_tsb_regressor(X_tr, y_tr, X_val, y_val, model_tag: str = ""):
    """
    Train a LightGBM regressor to predict TSB (blood_mg_dl).
    Uses MAE (regression_l1) loss — more robust to outlier bilirubin values.
    """
    log.info(f"    [{model_tag}] Training LGBMRegressor (TSB, MAE loss)")
    log.info(f"    [{model_tag}] Train: {X_tr.shape}  TSB mean={y_tr.mean():.2f} std={y_tr.std():.2f}")
    log.info(f"    [{model_tag}] Val:   {X_val.shape}  TSB mean={y_val.mean():.2f} std={y_val.std():.2f}")

    m = lgb.LGBMRegressor(**LGB_REGRESSION_PARAMS)  # type: ignore
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    log.info(f"    [{model_tag}] Best iteration: {m.best_iteration_}")

    fi_gain = dict(zip(X_tr.columns, m.booster_.feature_importance(importance_type="gain")))
    log.info(f"    [{model_tag}] Top-10 feature importance (gain):")
    for feat, val in sorted(fi_gain.items(), key=lambda x: -x[1])[:10]:
        log.info(f"        {feat:<35s}  gain={val:.4f}")

    if LOGGING_ENABLED:
        _save_json({
            "model_tag": model_tag, "best_iteration": m.best_iteration_,
            "train_shape": list(X_tr.shape), "val_shape": list(X_val.shape),
            "feature_importance_gain": fi_gain, "features": list(X_tr.columns),
        }, f"{model_tag}_training_detail.json")

    return m


def evaluate_binary(model, X, y_true, split_name: str, model_tag: str = "") -> dict:
    """Binary evaluation: acc, AUC, F1, confusion matrix."""
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    acc  = accuracy_score(y_true, preds)
    auc  = roc_auc_score(y_true, proba)
    f1   = f1_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    cm   = confusion_matrix(y_true, preds)
    cr   = classification_report(y_true, preds, zero_division=0)

    fpr, tpr, roc_thresh = roc_curve(y_true, proba)
    j_scores  = tpr - fpr
    best_thresh = float(roc_thresh[np.argmax(j_scores)])

    log.info(f"    [{model_tag}|{split_name}]  Acc={acc*100:.2f}%  AUC={auc*100:.2f}%"
             f"  F1={f1*100:.2f}%  Prec={prec*100:.2f}%  Recall={rec*100:.2f}%")
    log.info(f"    [{model_tag}|{split_name}]  Optimal threshold (Youden J): {best_thresh:.4f}")
    log.info(f"    [{model_tag}|{split_name}]  CM: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    log.info(f"    [{model_tag}|{split_name}]  Report:\n{cr}")

    result = {
        "split": split_name, "model_tag": model_tag,
        "accuracy": float(acc), "auc_roc": float(auc), "f1": float(f1),
        "precision": float(prec), "recall": float(rec),
        "confusion_matrix": cm.tolist(),
        "optimal_threshold_youden": best_thresh,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresh.tolist()},
    }
    if LOGGING_ENABLED:
        _save_json(result, f"{model_tag}_eval_{split_name}.json")
    return result


def evaluate_regression(model, X, y_true, split_name: str, model_tag: str = "") -> dict:
    """
    Regression evaluation for TSB prediction.
    Reports: MAE, RMSE, R², and %-within-2 mg/dL (clinically meaningful).
    Also logs TSB distribution of predictions vs ground truth.
    """
    preds = model.predict(X)
    # Clip to plausible TSB range to avoid extreme outliers being served to Flutter
    preds_clipped = np.clip(preds, 0.0, 40.0)

    mae  = mean_absolute_error(y_true, preds_clipped)
    rmse = float(np.sqrt(mean_squared_error(y_true, preds_clipped)))
    r2   = r2_score(y_true, preds_clipped)
    within_2  = float(np.mean(np.abs(y_true - preds_clipped) <= 2.0) * 100)
    within_3  = float(np.mean(np.abs(y_true - preds_clipped) <= 3.0) * 100)
    within_5  = float(np.mean(np.abs(y_true - preds_clipped) <= 5.0) * 100)
    bias      = float(np.mean(preds_clipped - y_true))   # systematic over/under

    log.info(f"    [{model_tag}|{split_name}]  MAE={mae:.3f} mg/dL  RMSE={rmse:.3f}  R²={r2:.4f}")
    log.info(f"    [{model_tag}|{split_name}]  Within ±2 mg/dL: {within_2:.1f}%"
             f"  ±3: {within_3:.1f}%  ±5: {within_5:.1f}%")
    log.info(f"    [{model_tag}|{split_name}]  Bias (mean pred-true): {bias:+.3f} mg/dL")
    log.info(f"    [{model_tag}|{split_name}]  Pred  — mean={preds_clipped.mean():.2f}"
             f"  std={preds_clipped.std():.2f}  min={preds_clipped.min():.2f}"
             f"  max={preds_clipped.max():.2f}")
    log.info(f"    [{model_tag}|{split_name}]  True  — mean={y_true.mean():.2f}"
             f"  std={y_true.std():.2f}  min={y_true.min():.2f}  max={y_true.max():.2f}")

    result = {
        "split": split_name, "model_tag": model_tag,
        "mae": float(mae), "rmse": rmse, "r2": float(r2),
        "bias_mean": bias,
        "within_2_mgdl_pct": within_2,
        "within_3_mgdl_pct": within_3,
        "within_5_mgdl_pct": within_5,
        "pred_mean": float(preds_clipped.mean()), "pred_std": float(preds_clipped.std()),
        "true_mean": float(y_true.mean()),         "true_std": float(y_true.std()),
    }
    if LOGGING_ENABLED:
        pred_df = pd.DataFrame({"y_true": y_true.values, "y_pred": preds_clipped,
                                 "abs_error": np.abs(y_true.values - preds_clipped)})
        _save_csv(pred_df, f"{model_tag}_predictions_{split_name}.csv")
        _save_json(result, f"{model_tag}_eval_{split_name}.json")
    return result


def shap_select_binary(model, X_val: pd.DataFrame, features: list, model_tag: str = "") -> list:
    """SHAP feature selection for binary classifiers."""
    log.info(f"    [{model_tag}] Computing SHAP values ({X_val.shape[0]} × {len(features)})")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_val)
    if isinstance(sv, list):
        sv = sv[1]
    mean_shap = np.abs(sv).mean(axis=0)
    max_shap  = mean_shap.max()
    cutoff    = SHAP_CUTOFF * max_shap
    selected  = [f for f, s in zip(features, mean_shap) if s >= cutoff]
    dropped   = [f for f, s in zip(features, mean_shap) if s < cutoff]
    log.info(f"    [{model_tag}] Kept {len(selected)}/{len(features)}  Dropped: {dropped}")

    if LOGGING_ENABLED:
        summary = pd.DataFrame({
            "feature": features, "mean_abs_shap": mean_shap.tolist(),
            "pct_of_top": (mean_shap / max_shap * 100).tolist(),
            "selected": [s >= cutoff for s in mean_shap],
        }).sort_values("mean_abs_shap", ascending=False)
        _save_csv(summary, f"{model_tag}_shap_summary.csv")
        _save_json({
            "model_tag": model_tag, "n_in": len(features), "n_kept": len(selected),
            "cutoff_value": float(cutoff), "max_value": float(max_shap),
            "selected": selected, "dropped": dropped,
        }, f"{model_tag}_shap_selection.json")
    return selected


def shap_select_regressor(model, X_val: pd.DataFrame, features: list, model_tag: str = "") -> list:
    """SHAP feature selection for regression models."""
    log.info(f"    [{model_tag}] Computing SHAP values ({X_val.shape[0]} × {len(features)})")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_val)   # shape: (n_samples, n_features)
    mean_shap = np.abs(sv).mean(axis=0)
    max_shap  = mean_shap.max()
    cutoff    = SHAP_CUTOFF * max_shap
    selected  = [f for f, s in zip(features, mean_shap) if s >= cutoff]
    dropped   = [f for f, s in zip(features, mean_shap) if s < cutoff]
    log.info(f"    [{model_tag}] Kept {len(selected)}/{len(features)}  Dropped: {dropped}")

    if LOGGING_ENABLED:
        summary = pd.DataFrame({
            "feature": features, "mean_abs_shap": mean_shap.tolist(),
            "pct_of_top": (mean_shap / max_shap * 100).tolist(),
            "selected": [s >= cutoff for s in mean_shap],
        }).sort_values("mean_abs_shap", ascending=False)
        _save_csv(summary, f"{model_tag}_shap_summary.csv")
        _save_json({
            "model_tag": model_tag, "n_in": len(features), "n_kept": len(selected),
            "cutoff_value": float(cutoff), "max_value": float(max_shap),
            "selected": selected, "dropped": dropped,
        }, f"{model_tag}_shap_selection.json")
    return selected


def save_model(model, features: list, name: str, model_type: str):
    """Pickle model bundle: {model, features, model_type}."""
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    payload = {"model": model, "features": features, "model_type": model_type}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    size_kb = os.path.getsize(path) / 1024
    log.info(f"    [save] {path}  ({size_kb:.1f} KB)  type={model_type}")
    if LOGGING_ENABLED:
        _save_json({"name": name, "path": path, "size_kb": size_kb,
                    "n_features": len(features), "features": features,
                    "model_type": model_type}, f"{name}_saved_model_info.json")


all_results = {}

# ─────────────────────────────────────────────
# MODEL 1A — Binary Detection + Metadata
# ─────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("MODEL 1A — Binary Detection (color + metadata)")
log.info("=" * 70)

m1A_full = train_binary_clf(
    train_df[ALL_FEATURES], train_df[DETECT_LABEL],
    val_df[ALL_FEATURES],   val_df[DETECT_LABEL],
    model_tag="1A_full",
)
feats_1A = shap_select_binary(m1A_full, val_df[ALL_FEATURES], ALL_FEATURES, model_tag="1A")
m1A = train_binary_clf(
    train_df[feats_1A], train_df[DETECT_LABEL],
    val_df[feats_1A],   val_df[DETECT_LABEL],
    model_tag="1A_final",
)
r1A_val  = evaluate_binary(m1A, val_df[feats_1A],  val_df[DETECT_LABEL],  "Val",  model_tag="1A")
r1A_test = evaluate_binary(m1A, test_df[feats_1A], test_df[DETECT_LABEL], "Test", model_tag="1A")
save_model(m1A, feats_1A, "model_1A", model_type="binary")
all_results["1A"] = {"val": r1A_val, "test": r1A_test, "n_feat": len(feats_1A), "features": feats_1A}

# ─────────────────────────────────────────────
# MODEL 1B — Binary Detection, Color Only
# ─────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("MODEL 1B — Binary Detection (color only)")
log.info("=" * 70)

m1B_full = train_binary_clf(
    train_df[COLOR_FEATURES], train_df[DETECT_LABEL],
    val_df[COLOR_FEATURES],   val_df[DETECT_LABEL],
    model_tag="1B_full",
)
feats_1B = shap_select_binary(m1B_full, val_df[COLOR_FEATURES], COLOR_FEATURES, model_tag="1B")
m1B = train_binary_clf(
    train_df[feats_1B], train_df[DETECT_LABEL],
    val_df[feats_1B],   val_df[DETECT_LABEL],
    model_tag="1B_final",
)
r1B_val  = evaluate_binary(m1B, val_df[feats_1B],  val_df[DETECT_LABEL],  "Val",  model_tag="1B")
r1B_test = evaluate_binary(m1B, test_df[feats_1B], test_df[DETECT_LABEL], "Test", model_tag="1B")
save_model(m1B, feats_1B, "model_1B", model_type="binary")
all_results["1B"] = {"val": r1B_val, "test": r1B_test, "n_feat": len(feats_1B), "features": feats_1B}

# ─────────────────────────────────────────────
# MODEL 2A — TSB Regressor + Metadata
# ─────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("MODEL 2A — TSB Regressor (color + metadata)")
log.info("  Trained on JAUNDICED patients only (blood_mg_dl is the target).")
log.info("  Model 1A/1B gates detection; Model 2A/2B estimates the number.")
log.info("=" * 70)

m2A_full = train_tsb_regressor(
    reg_train_df[ALL_FEATURES], reg_train_df[TSB_COL],
    reg_val_df[ALL_FEATURES],   reg_val_df[TSB_COL],
    model_tag="2A_full",
)
feats_2A = shap_select_regressor(m2A_full, reg_val_df[ALL_FEATURES], ALL_FEATURES, model_tag="2A")
m2A = train_tsb_regressor(
    reg_train_df[feats_2A], reg_train_df[TSB_COL],
    reg_val_df[feats_2A],   reg_val_df[TSB_COL],
    model_tag="2A_final",
)
r2A_val  = evaluate_regression(m2A, reg_val_df[feats_2A],  reg_val_df[TSB_COL],  "Val",  model_tag="2A")
r2A_test = evaluate_regression(m2A, reg_test_df[feats_2A], reg_test_df[TSB_COL], "Test", model_tag="2A")
save_model(m2A, feats_2A, "model_2A", model_type="tsb_regressor")
all_results["2A"] = {"val": r2A_val, "test": r2A_test, "n_feat": len(feats_2A), "features": feats_2A}

# ─────────────────────────────────────────────
# MODEL 2B — TSB Regressor, Color Only
# ─────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("MODEL 2B — TSB Regressor (color only)")
log.info("  Used when gestational_age / postnatal_age_days / weight are unavailable.")
log.info("=" * 70)

m2B_full = train_tsb_regressor(
    reg_train_df[COLOR_FEATURES], reg_train_df[TSB_COL],
    reg_val_df[COLOR_FEATURES],   reg_val_df[TSB_COL],
    model_tag="2B_full",
)
feats_2B = shap_select_regressor(m2B_full, reg_val_df[COLOR_FEATURES], COLOR_FEATURES, model_tag="2B")
m2B = train_tsb_regressor(
    reg_train_df[feats_2B], reg_train_df[TSB_COL],
    reg_val_df[feats_2B],   reg_val_df[TSB_COL],
    model_tag="2B_final",
)
r2B_val  = evaluate_regression(m2B, reg_val_df[feats_2B],  reg_val_df[TSB_COL],  "Val",  model_tag="2B")
r2B_test = evaluate_regression(m2B, reg_test_df[feats_2B], reg_test_df[TSB_COL], "Test", model_tag="2B")
save_model(m2B, feats_2B, "model_2B", model_type="tsb_regressor")
all_results["2B"] = {"val": r2B_val, "test": r2B_test, "n_feat": len(feats_2B), "features": feats_2B}

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
log.info("\n\n" + "=" * 85)
log.info("  FINAL RESULTS SUMMARY — Plan v4 (Detection + TSB Regression)")
log.info("=" * 85)
log.info("  Models 1A/1B: binary detection (jaundice vs normal)")
log.info("  Models 2A/2B: TSB regressor → predicts blood_mg_dl (mg/dL)")
log.info("  Risk zone classification is done in the Flutter app via Bhutani Nomogram.")
log.info("=" * 85)

log.info(f"\n{'Model':<6} {'Feat':>5} | {'Val Acc':>8} {'Val AUC':>8} {'Val F1':>7}"
         f" | {'Test Acc':>9} {'Test AUC':>9} {'Test F1':>8}")
log.info("-" * 85)
for k in ["1A", "1B"]:
    r  = all_results[k]
    va = r["val"]["accuracy"];  vu = r["val"]["auc_roc"];  vf = r["val"]["f1"]
    ta = r["test"]["accuracy"]; tu = r["test"]["auc_roc"]; tf = r["test"]["f1"]
    log.info(f"  {k:<4} {r['n_feat']:>5} | {va*100:>7.2f}% {vu*100:>7.2f}% {vf*100:>6.2f}%"
             f" | {ta*100:>8.2f}% {tu*100:>8.2f}% {tf*100:>7.2f}%   [binary detection]")

log.info("-" * 85)
log.info(f"  {'Model':<6} {'Feat':>5} | {'Val MAE':>9} {'Val RMSE':>9} {'Val R²':>7} {'±2mgdL%':>8}"
         f" | {'Test MAE':>9} {'Test RMSE':>9} {'Test R²':>7} {'±2mgdL%':>8}")
log.info("-" * 85)
for k in ["2A", "2B"]:
    r  = all_results[k]
    vr = r["val"];  tr = r["test"]
    log.info(f"  {k:<4} {r['n_feat']:>5} | {vr['mae']:>9.3f} {vr['rmse']:>9.3f}"
             f" {vr['r2']:>7.4f} {vr['within_2_mgdl_pct']:>7.1f}%"
             f" | {tr['mae']:>9.3f} {tr['rmse']:>9.3f}"
             f" {tr['r2']:>7.4f} {tr['within_2_mgdl_pct']:>7.1f}%   [TSB regressor]")

log.info("=" * 85)

if LOGGING_ENABLED:
    summary = {k: {"n_features": all_results[k]["n_feat"],
                   "val": all_results[k]["val"], "test": all_results[k]["test"]}
               for k in ["1A", "1B", "2A", "2B"]}
    _save_json({
        "run_timestamp": RUN_TIMESTAMP,
        "plan": "v4",
        "lgb_binary_params": LGB_BINARY_PARAMS,
        "lgb_regression_params": LGB_REGRESSION_PARAMS,
        "shap_cutoff": SHAP_CUTOFF,
        "results": summary,
    }, "00_MASTER_SUMMARY.json")
    log.info(f"\n  All artifacts saved to: {LOG_DIR.resolve()}")  # type: ignore

log.info("\nSemua model tersimpan di folder: ./__models__/")
log.info("Gunakan predict_v4.py untuk inferensi pada data baru.")