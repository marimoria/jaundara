"""
train_models_v4.py — Neonatal Jaundice Detection (Plan v4)

Architecture:
  Model 1A/1B  — binary detection gate (jaundiced vs normal)
  Model 2A/2B  — TSB regressor: predicts blood_mg_dl as a continuous value

Inference pipeline:
  1. Model 1A/1B detects jaundice (binary gate).
  2. Model 2A/2B estimates bilirubin (mg/dL).
  3. Flutter app evaluates the Bhutani Nomogram using (tsb_estimate, postnatal_age_days)
     to surface the risk zone and recommended action.

Why regression over classification:
  The model reads color to estimate a continuous quantity. Phototherapy thresholds
  depend on both TSB and age — an interaction the model cannot learn without age as
  a feature, but which the app always has. Regression also gives the app a real number
  to compare against any nomogram curve, without baking a fixed threshold into weights.

Usage:
  python train_models_v4.py
  python train_models_v4.py --log
  python train_models_v4.py --log --log-dir path/to/dir
"""

import argparse
import datetime
import json
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_PATH = "__data__/neo/out/training.csv"
MODELS_DIR = "__models__"
DETECT_LABEL = "jaundice_label"
TSB_COL = "blood_mg_dl"
SHAP_CUTOFF = 0.01

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

parser = argparse.ArgumentParser(
    description="Train neonatal jaundice models (Plan v4)."
)
parser.add_argument(
    "--log",
    action="store_true",
    default=False,
    help="Write detailed logs and artifacts to __data__/models_log/<timestamp>/",
)
parser.add_argument(
    "--log-dir",
    type=str,
    default=None,
    help="Override the log output directory (implies --log).",
)
args = parser.parse_args()
LOGGING_ENABLED = args.log or (args.log_dir is not None)

RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if LOGGING_ENABLED:
    LOG_DIR = (
        Path(args.log_dir)
        if args.log_dir
        else Path("__data__") / "models_log" / RUN_TIMESTAMP
    )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "run.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
else:
    LOG_DIR = None
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

log = logging.getLogger("train_models_v4")

os.makedirs(MODELS_DIR, exist_ok=True)

if LOGGING_ENABLED:
    log.info("Log directory  : %s", LOG_DIR.resolve())  # type: ignore
    log.info("Run timestamp  : %s", RUN_TIMESTAMP)
    log.info("LGB_BINARY_PARAMS     : %s", json.dumps(LGB_BINARY_PARAMS, indent=2))
    log.info("LGB_REGRESSION_PARAMS : %s", json.dumps(LGB_REGRESSION_PARAMS, indent=2))
    log.info("SHAP_CUTOFF    : %s", SHAP_CUTOFF)


def _save_json(obj, filename: str):
    if not LOGGING_ENABLED:
        return
    out = LOG_DIR / filename  # type: ignore
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    log.debug("[artifact] %s", out.name)


def _save_csv(df: pd.DataFrame, filename: str):
    if not LOGGING_ENABLED:
        return
    out = LOG_DIR / filename  # type: ignore
    df.to_csv(out, index=False)
    log.debug("[artifact] %s", out.name)


df = pd.read_csv(DATA_PATH)
log.info("rows=%d  cols=%d", len(df), len(df.columns))

vc = df[DETECT_LABEL].value_counts()
for val, cnt in vc.items():
    log.info("  label %s: %d (%.1f%%)", val, cnt, cnt / len(df) * 100)

COLOR_FEATURES = [c for c in df.columns if c.startswith("zone")]
META_FEATURES = ["gestational_age", "postnatal_age_days", "weight"]
ALL_FEATURES = COLOR_FEATURES + META_FEATURES

tsb = df[TSB_COL]
log.info(
    "TSB: min=%.2f  max=%.2f  mean=%.2f  std=%.2f",
    tsb.min(),
    tsb.max(),
    tsb.mean(),
    tsb.std(),
)
log.info("color_features=%d  meta_features=%d", len(COLOR_FEATURES), len(META_FEATURES))

patients = df[~df["is_augmented"]]["patient_id"].unique()
train_p, temp_p = train_test_split(patients, test_size=0.30, random_state=42)
val_p, test_p = train_test_split(temp_p, test_size=0.50, random_state=42)

train_df = df[df["patient_id"].isin(train_p)].copy()
val_df = df[df["patient_id"].isin(val_p) & ~df["is_augmented"]].copy()
test_df = df[df["patient_id"].isin(test_p) & ~df["is_augmented"]].copy()

log.info(
    "split — patients: train=%d  val=%d  test=%d", len(train_p), len(val_p), len(test_p)
)
log.info(
    "split — rows:     train=%d  val=%d  test=%d",
    len(train_df),
    len(val_df),
    len(test_df),
)

_save_json(
    {
        "n_patients": {"train": len(train_p), "val": len(val_p), "test": len(test_p)},
        "n_rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "detect_dist": {
            "train": train_df[DETECT_LABEL].value_counts().to_dict(),
            "val": val_df[DETECT_LABEL].value_counts().to_dict(),
            "test": test_df[DETECT_LABEL].value_counts().to_dict(),
        },
    },
    "01_split_summary.json",
)


def _compute_class_weights(y_series: pd.Series) -> np.ndarray:
    classes = sorted(y_series.unique())
    n_total = len(y_series)
    n_classes = len(classes)
    cw = {c: n_total / (n_classes * (y_series == c).sum()) for c in classes}
    return y_series.map(cw).values  # type: ignore


def train_binary_clf(X_tr, y_tr, X_val, y_val, model_tag: str = ""):
    log.info(
        "[%s] training binary clf — train=%s  val=%s  pos=%.1f%%",
        model_tag,
        X_tr.shape,
        X_val.shape,
        y_tr.mean() * 100,
    )
    m = lgb.LGBMClassifier(**LGB_BINARY_PARAMS)  # type: ignore
    m.fit(
        X_tr,
        y_tr,
        sample_weight=_compute_class_weights(y_tr),
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    log.info("[%s] best_iteration=%d", model_tag, m.best_iteration_)
    fi = dict(zip(X_tr.columns, m.booster_.feature_importance(importance_type="gain")))
    for feat, val in sorted(fi.items(), key=lambda x: -x[1])[:10]:
        log.info("  %-35s gain=%.4f", feat, val)
    if LOGGING_ENABLED:
        _save_json(
            {
                "model_tag": model_tag,
                "best_iteration": m.best_iteration_,
                "feature_importance_gain": fi,
                "features": list(X_tr.columns),
            },
            f"{model_tag}_training_detail.json",
        )
    return m


def train_tsb_regressor(X_tr, y_tr, X_val, y_val, model_tag: str = ""):
    log.info(
        "[%s] training TSB regressor — train=%s  val=%s  tsb_mean=%.2f",
        model_tag,
        X_tr.shape,
        X_val.shape,
        y_tr.mean(),
    )
    m = lgb.LGBMRegressor(**LGB_REGRESSION_PARAMS)  # type: ignore
    m.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    log.info("[%s] best_iteration=%d", model_tag, m.best_iteration_)
    fi = dict(zip(X_tr.columns, m.booster_.feature_importance(importance_type="gain")))
    for feat, val in sorted(fi.items(), key=lambda x: -x[1])[:10]:
        log.info("  %-35s gain=%.4f", feat, val)
    if LOGGING_ENABLED:
        _save_json(
            {
                "model_tag": model_tag,
                "best_iteration": m.best_iteration_,
                "feature_importance_gain": fi,
                "features": list(X_tr.columns),
            },
            f"{model_tag}_training_detail.json",
        )
    return m


def evaluate_binary(model, X, y_true, split_name: str, model_tag: str = "") -> dict:
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, proba)
    f1 = f1_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds)
    cr = classification_report(y_true, preds, zero_division=0)
    fpr, tpr, roc_thresh = roc_curve(y_true, proba)
    best_thresh = float(roc_thresh[np.argmax(tpr - fpr)])

    log.info(
        "[%s|%s] acc=%.2f%%  auc=%.2f%%  f1=%.2f%%  prec=%.2f%%  rec=%.2f%%",
        model_tag,
        split_name,
        acc * 100,
        auc * 100,
        f1 * 100,
        prec * 100,
        rec * 100,
    )
    log.info(
        "[%s|%s] optimal_threshold=%.4f  TN=%d FP=%d FN=%d TP=%d",
        model_tag,
        split_name,
        best_thresh,
        cm[0, 0],
        cm[0, 1],
        cm[1, 0],
        cm[1, 1],
    )
    log.info("[%s|%s] report:\n%s", model_tag, split_name, cr)

    result = {
        "split": split_name,
        "model_tag": model_tag,
        "accuracy": float(acc),
        "auc_roc": float(auc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "confusion_matrix": cm.tolist(),
        "optimal_threshold_youden": best_thresh,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresh.tolist(),
        },
    }
    if LOGGING_ENABLED:
        _save_json(result, f"{model_tag}_eval_{split_name}.json")
    return result


def evaluate_regression(model, X, y_true, split_name: str, model_tag: str = "") -> dict:
    preds = np.clip(model.predict(X), 0.0, 40.0)

    mae = mean_absolute_error(y_true, preds)
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    r2 = r2_score(y_true, preds)
    within_2 = float(np.mean(np.abs(y_true - preds) <= 2.0) * 100)
    within_3 = float(np.mean(np.abs(y_true - preds) <= 3.0) * 100)
    within_5 = float(np.mean(np.abs(y_true - preds) <= 5.0) * 100)
    bias = float(np.mean(preds - y_true))

    log.info(
        "[%s|%s] MAE=%.3f  RMSE=%.3f  R²=%.4f  ±2mgdL=%.1f%%  bias=%+.3f",
        model_tag,
        split_name,
        mae,
        rmse,
        r2,
        within_2,
        bias,
    )
    log.info(
        "[%s|%s] pred: mean=%.2f std=%.2f  true: mean=%.2f std=%.2f",
        model_tag,
        split_name,
        preds.mean(),
        preds.std(),
        y_true.mean(),
        y_true.std(),
    )

    result = {
        "split": split_name,
        "model_tag": model_tag,
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "bias_mean": bias,
        "within_2_mgdl_pct": within_2,
        "within_3_mgdl_pct": within_3,
        "within_5_mgdl_pct": within_5,
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
        "true_mean": float(y_true.mean()),
        "true_std": float(y_true.std()),
    }
    if LOGGING_ENABLED:
        _save_csv(
            pd.DataFrame(
                {
                    "y_true": y_true.values,
                    "y_pred": preds,
                    "abs_error": np.abs(y_true.values - preds),
                }
            ),
            f"{model_tag}_predictions_{split_name}.csv",
        )
        _save_json(result, f"{model_tag}_eval_{split_name}.json")
    return result


def _shap_select(
    model,
    X_val: pd.DataFrame,
    features: list,
    model_tag: str,
    is_classifier: bool = True,
) -> list:
    log.info("[%s] SHAP selection — %d × %d", model_tag, X_val.shape[0], len(features))
    sv = shap.TreeExplainer(model).shap_values(X_val)
    if isinstance(sv, list):
        sv = sv[1]
    mean_shap = np.abs(sv).mean(axis=0)
    max_shap = mean_shap.max()
    cutoff = SHAP_CUTOFF * max_shap
    selected = [f for f, s in zip(features, mean_shap) if s >= cutoff]
    dropped = [f for f, s in zip(features, mean_shap) if s < cutoff]
    log.info(
        "[%s] kept %d/%d  dropped: %s", model_tag, len(selected), len(features), dropped
    )
    if LOGGING_ENABLED:
        _save_csv(
            pd.DataFrame(
                {
                    "feature": features,
                    "mean_abs_shap": mean_shap.tolist(),
                    "pct_of_top": (mean_shap / max_shap * 100).tolist(),
                    "selected": [s >= cutoff for s in mean_shap],
                }
            ).sort_values("mean_abs_shap", ascending=False),
            f"{model_tag}_shap_summary.csv",
        )
        _save_json(
            {
                "model_tag": model_tag,
                "n_in": len(features),
                "n_kept": len(selected),
                "cutoff_value": float(cutoff),
                "selected": selected,
                "dropped": dropped,
            },
            f"{model_tag}_shap_selection.json",
        )
    return selected


def save_model(model, features: list, name: str, model_type: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": model, "features": features, "model_type": model_type}, f)
    size_kb = os.path.getsize(path) / 1024
    log.info("[save] %s  (%.1f KB)  type=%s", path, size_kb, model_type)
    if LOGGING_ENABLED:
        _save_json(
            {
                "name": name,
                "path": path,
                "size_kb": size_kb,
                "n_features": len(features),
                "features": features,
                "model_type": model_type,
            },
            f"{name}_saved_model_info.json",
        )


all_results = {}

for tag, feat_set, label, trainer, evaluator in [
    ("1A", ALL_FEATURES, DETECT_LABEL, train_binary_clf, evaluate_binary),
    ("1B", COLOR_FEATURES, DETECT_LABEL, train_binary_clf, evaluate_binary),
    ("2A", ALL_FEATURES, TSB_COL, train_tsb_regressor, evaluate_regression),
    ("2B", COLOR_FEATURES, TSB_COL, train_tsb_regressor, evaluate_regression),
]:
    is_clf = tag.startswith("1")
    model_type = "binary" if is_clf else "tsb_regressor"
    log.info("--- model %s ---", tag)

    m_full = trainer(
        train_df[feat_set],
        train_df[label],
        val_df[feat_set],
        val_df[label],
        model_tag=f"{tag}_full",
    )
    feats = _shap_select(
        m_full, val_df[feat_set], feat_set, model_tag=tag, is_classifier=is_clf
    )
    m_final = trainer(
        train_df[feats],
        train_df[label],
        val_df[feats],
        val_df[label],
        model_tag=f"{tag}_final",
    )

    r_val = evaluator(m_final, val_df[feats], val_df[label], "Val", model_tag=tag)
    r_test = evaluator(m_final, test_df[feats], test_df[label], "Test", model_tag=tag)
    save_model(m_final, feats, f"model_{tag}", model_type=model_type)
    all_results[tag] = {
        "val": r_val,
        "test": r_test,
        "n_feat": len(feats),
        "features": feats,
    }


log.info("--- summary ---")
log.info(
    "%-6s %5s | %8s %8s %7s | %9s %9s %8s",
    "Model",
    "Feat",
    "Val Acc",
    "Val AUC",
    "Val F1",
    "Test Acc",
    "Test AUC",
    "Test F1",
)
for k in ["1A", "1B"]:
    r = all_results[k]
    v, t = r["val"], r["test"]
    log.info(
        "%-6s %5d | %7.2f%% %7.2f%% %6.2f%% | %8.2f%% %8.2f%% %7.2f%%",
        k,
        r["n_feat"],
        v["accuracy"] * 100,
        v["auc_roc"] * 100,
        v["f1"] * 100,
        t["accuracy"] * 100,
        t["auc_roc"] * 100,
        t["f1"] * 100,
    )

log.info(
    "%-6s %5s | %9s %9s %7s %8s | %9s %9s %7s %8s",
    "Model",
    "Feat",
    "Val MAE",
    "Val RMSE",
    "Val R²",
    "±2mgdL%",
    "Test MAE",
    "Test RMSE",
    "Test R²",
    "±2mgdL%",
)
for k in ["2A", "2B"]:
    r = all_results[k]
    v, t = r["val"], r["test"]
    log.info(
        "%-6s %5d | %9.3f %9.3f %7.4f %7.1f%% | %9.3f %9.3f %7.4f %7.1f%%",
        k,
        r["n_feat"],
        v["mae"],
        v["rmse"],
        v["r2"],
        v["within_2_mgdl_pct"],
        t["mae"],
        t["rmse"],
        t["r2"],
        t["within_2_mgdl_pct"],
    )

if LOGGING_ENABLED:
    _save_json(
        {
            "run_timestamp": RUN_TIMESTAMP,
            "plan": "v4",
            "lgb_binary_params": LGB_BINARY_PARAMS,
            "lgb_regression_params": LGB_REGRESSION_PARAMS,
            "shap_cutoff": SHAP_CUTOFF,
            "results": {
                k: {
                    "n_features": all_results[k]["n_feat"],
                    "val": all_results[k]["val"],
                    "test": all_results[k]["test"],
                }
                for k in ["1A", "1B", "2A", "2B"]
            },
        },
        "00_MASTER_SUMMARY.json",
    )
    log.info("artifacts saved to: %s", LOG_DIR.resolve())  # type: ignore

log.info("models saved to: ./%s/", MODELS_DIR)
