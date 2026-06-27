"""
train_models.py — Neonatal Jaundice Detection

Architecture:
  Model 1  — binary detection gate (jaundiced vs normal)
  Model 2  — TSB regressor: predicts blood_mg_dl as a continuous value

Inference pipeline:
  1. Model 1 detects jaundice (binary gate).
  2. Model 2 estimates bilirubin (mg/dL).
  3. Flutter app evaluates the Bhutani Nomogram using (tsb_estimate, postnatal_age_days)
     to surface the risk zone and recommended action.

Feature set (built from training_engineered.csv):
  - All original zone color features with |Spearman r| >= 0.10 vs blood_mg_dl
  - Engineered features: cross-zone means, R/B ratios, G-B differences,
    cross-zone gradients (zone3-zone1), mean_zones_H_mean
  - Clinical: postnatal_age_days, gestational_age
  - Dropped: pure brightness/luminance features (|r| < 0.10), weight (r=0.037),
    log1p_postnatal_age_days (zero gain over raw), all ITA features (weaker than Lab_b),
    all R_mean/G_mean/Y_mean/Lab_L_mean/Lab_a_mean columns

SHAP selection: cumulative importance at 95% of total SHAP mass (self-calibrating).
  Defers redundancy resolution (e.g. Y_mean vs Lab_L_mean r=0.999) to SHAP
  post-training rather than pre-training correlation pruning.

Usage:
  python train_models.py
  python train_models.py --log
  python train_models.py --log --log-dir path/to/dir
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

DATA_PATH  = "__data__/neo/out/training_engineered.csv"
MODELS_DIR = "__models__"
DETECT_LABEL = "jaundice_label"
TSB_COL      = "blood_mg_dl"

# SHAP selection: keep features that together account for this fraction of
# total SHAP mass, sorted descending. Self-calibrates regardless of target scale.
SHAP_CUMULATIVE_THRESHOLD = 0.95

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
    n_estimators=700,
    learning_rate=0.01558230773982519,
    num_leaves=93,
    min_child_samples=17,
    subsample=0.7969181420869593,
    colsample_bytree=0.7886165660038295,
    random_state=42,
    verbose=-1,
)

parser = argparse.ArgumentParser(description="Train neonatal jaundice models.")
parser.add_argument("--log", action="store_true", default=False,
                    help="Write detailed logs and artifacts to __data__/models_log/<timestamp>/")
parser.add_argument("--log-dir", type=str, default=None,
                    help="Override the log output directory (implies --log).")
args = parser.parse_args()
LOGGING_ENABLED = args.log or (args.log_dir is not None)

RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if LOGGING_ENABLED:
    LOG_DIR = (
        Path(args.log_dir) if args.log_dir
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

log = logging.getLogger("train_models")
os.makedirs(MODELS_DIR, exist_ok=True)

if LOGGING_ENABLED:
    log.info("Log directory  : %s", LOG_DIR.resolve())  # type: ignore
    log.info("Run timestamp  : %s", RUN_TIMESTAMP)
    log.info("SHAP_CUMULATIVE_THRESHOLD : %.2f", SHAP_CUMULATIVE_THRESHOLD)
    log.info("LGB_BINARY_PARAMS     : %s", json.dumps(LGB_BINARY_PARAMS, indent=2))
    log.info("LGB_REGRESSION_PARAMS : %s", json.dumps(LGB_REGRESSION_PARAMS, indent=2))


# ── helpers ──────────────────────────────────────────────────────────────────

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


# ── feature engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features that are not already in the CSV, then return
    the dataframe with all columns needed for training.

    Already present in training_engineered.csv (from phase4_exploration.py):
      mean_zones_Lab_b_mean, mean_zones_Cb_mean, mean_zones_B_mean,
      mean_zones_S_mean, zone{1-3}_R_div_B, zone{1-3}_G_minus_B,
      log1p_postnatal_age_days

    Added here (missing from CSV):
      mean_zones_H_mean, grad_z3z1_{Lab_b_mean,Cb_mean,H_mean}

    Dropped / never added:
      log1p_postnatal_age_days  — zero Spearman gain over raw; use postnatal_age_days
      zone{1-3}_ITA             — |r| 0.06–0.17, weaker than their Lab_b components
    """
    out = df.copy()

    zones = ["zone1", "zone2", "zone3"]

    # cross-zone H mean (not in CSV)
    h_cols = [f"{z}_H_mean" for z in zones]
    out["mean_zones_H_mean"] = out[h_cols].mean(axis=1)

    # cross-zone gradients zone3 - zone1 (not in CSV)
    for ch in ["Lab_b_mean", "Cb_mean", "H_mean"]:
        out[f"grad_z3z1_{ch}"] = out[f"zone3_{ch}"] - out[f"zone1_{ch}"]

    return out


def _define_feature_sets(df: pd.DataFrame) -> tuple[list, list]:
    """
    Return (COLOR_FEATURES, META_FEATURES) based on the feature-selection
    analysis (all |Spearman r| >= 0.10 vs blood_mg_dl on original rows).

    Explicitly excluded:
      *_R_mean, *_G_mean, *_Y_mean, *_Lab_L_mean, *_Lab_a_mean  — |r| < 0.10
      *_ITA                                                       — |r| < 0.18
      weight                                                      — |r| = 0.037
      log1p_postnatal_age_days                                    — zero gain
      zone2_Cr_mean                                               — |r| = 0.087
    """
    EXCLUDE_PATTERNS = (
        "_R_mean", "_G_mean", "_Y_mean",
        "_Lab_L_mean", "_Lab_a_mean",
        "_ITA",
        "log1p_",
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
            c.startswith("grad_z3z1") or c.startswith("mean_zones"))
        and _keep(c)
    ]

    meta_features = [
        "gestational_age",
        "postnatal_age_days",
    ]
    meta_features = [f for f in meta_features if f in df.columns]

    return color_features + meta_features # type: ignore


# ── data loading ──────────────────────────────────────────────────────────────

df_raw = pd.read_csv(DATA_PATH)
log.info("Loaded  rows=%d  cols=%d", len(df_raw), len(df_raw.columns))

df = build_features(df_raw)
ALL_FEATURES = _define_feature_sets(df)

log.info("total_features=%d", len(ALL_FEATURES))
log.info("ALL_FEATURES : %s", ALL_FEATURES)

tsb = df[TSB_COL]
log.info("TSB: min=%.2f  max=%.2f  mean=%.2f  std=%.2f",
         tsb.min(), tsb.max(), tsb.mean(), tsb.std())

vc = df[DETECT_LABEL].value_counts()
for val, cnt in vc.items():
    log.info("  label %s: %d (%.1f%%)", val, cnt, cnt / len(df) * 100)


# ── patient-level split ───────────────────────────────────────────────────────

patients = df[~df["is_augmented"]]["patient_id"].unique()
train_p, temp_p = train_test_split(patients, test_size=0.30, random_state=42)
val_p, test_p   = train_test_split(temp_p,  test_size=0.50, random_state=42)

train_df = df[df["patient_id"].isin(train_p)].copy()
val_df   = df[df["patient_id"].isin(val_p)  & ~df["is_augmented"]].copy()
test_df  = df[df["patient_id"].isin(test_p) & ~df["is_augmented"]].copy()

log.info("split — patients: train=%d  val=%d  test=%d",
         len(train_p), len(val_p), len(test_p))
log.info("split — rows:     train=%d  val=%d  test=%d",
         len(train_df), len(val_df), len(test_df))

_save_json({
    "n_patients": {"train": len(train_p), "val": len(val_p), "test": len(test_p)},
    "n_rows":     {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    "detect_dist": {
        "train": train_df[DETECT_LABEL].value_counts().to_dict(),
        "val":   val_df[DETECT_LABEL].value_counts().to_dict(),
        "test":  test_df[DETECT_LABEL].value_counts().to_dict(),
    },
}, "01_split_summary.json")


# ── training helpers ──────────────────────────────────────────────────────────

def _compute_class_weights(y_series: pd.Series) -> np.ndarray:
    classes  = sorted(y_series.unique())
    n_total  = len(y_series)
    n_classes = len(classes)
    cw = {c: n_total / (n_classes * (y_series == c).sum()) for c in classes}
    return y_series.map(cw).values  # type: ignore


def train_binary_clf(X_tr, y_tr, X_val, y_val, model_tag: str = ""):
    log.info("[%s] training binary clf — train=%s  val=%s  pos=%.1f%%",
             model_tag, X_tr.shape, X_val.shape, y_tr.mean() * 100)
    m = lgb.LGBMClassifier(**LGB_BINARY_PARAMS)  # type: ignore
    m.fit(
        X_tr, y_tr,
        sample_weight=_compute_class_weights(y_tr),
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    log.info("[%s] best_iteration=%d", model_tag, m.best_iteration_)
    fi = dict(zip(X_tr.columns, m.booster_.feature_importance(importance_type="gain")))
    for feat, val in sorted(fi.items(), key=lambda x: -x[1])[:10]:
        log.info("  %-40s gain=%.4f", feat, val)
    if LOGGING_ENABLED:
        _save_json({
            "model_tag": model_tag, "best_iteration": m.best_iteration_,
            "feature_importance_gain": fi, "features": list(X_tr.columns),
        }, f"{model_tag}_training_detail.json")
    return m


def train_tsb_regressor(X_tr, y_tr, X_val, y_val, model_tag: str = ""):
    log.info("[%s] training TSB regressor — train=%s  val=%s  tsb_mean=%.2f",
             model_tag, X_tr.shape, X_val.shape, y_tr.mean())
    m = lgb.LGBMRegressor(**LGB_REGRESSION_PARAMS)  # type: ignore
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    log.info("[%s] best_iteration=%d", model_tag, m.best_iteration_)
    fi = dict(zip(X_tr.columns, m.booster_.feature_importance(importance_type="gain")))
    for feat, val in sorted(fi.items(), key=lambda x: -x[1])[:10]:
        log.info("  %-40s gain=%.4f", feat, val)
    if LOGGING_ENABLED:
        _save_json({
            "model_tag": model_tag, "best_iteration": m.best_iteration_,
            "feature_importance_gain": fi, "features": list(X_tr.columns),
        }, f"{model_tag}_training_detail.json")
    return m


# ── evaluation helpers ────────────────────────────────────────────────────────

def evaluate_binary(model, X, y_true, split_name: str, model_tag: str = "") -> dict:
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
    best_thresh = float(roc_thresh[np.argmax(tpr - fpr)])

    log.info("[%s|%s] acc=%.2f%%  auc=%.2f%%  f1=%.2f%%  prec=%.2f%%  rec=%.2f%%",
             model_tag, split_name,
             acc*100, auc*100, f1*100, prec*100, rec*100)
    log.info("[%s|%s] optimal_threshold=%.4f  TN=%d FP=%d FN=%d TP=%d",
             model_tag, split_name, best_thresh,
             cm[0,0], cm[0,1], cm[1,0], cm[1,1])
    log.info("[%s|%s] report:\n%s", model_tag, split_name, cr)

    result = {
        "split": split_name, "model_tag": model_tag,
        "accuracy": float(acc), "auc_roc": float(auc),
        "f1": float(f1), "precision": float(prec), "recall": float(rec),
        "confusion_matrix": cm.tolist(),
        "optimal_threshold_youden": best_thresh,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                      "thresholds": roc_thresh.tolist()},
    }
    if LOGGING_ENABLED:
        _save_json(result, f"{model_tag}_eval_{split_name}.json")
    return result


def evaluate_regression(model, X, y_true, split_name: str, model_tag: str = "") -> dict:
    preds = np.clip(model.predict(X), 0.0, 40.0)

    mae     = mean_absolute_error(y_true, preds)
    rmse    = float(np.sqrt(mean_squared_error(y_true, preds)))
    r2      = r2_score(y_true, preds)
    within_2 = float(np.mean(np.abs(y_true - preds) <= 2.0) * 100)
    within_3 = float(np.mean(np.abs(y_true - preds) <= 3.0) * 100)
    within_5 = float(np.mean(np.abs(y_true - preds) <= 5.0) * 100)
    bias    = float(np.mean(preds - y_true))

    log.info("[%s|%s] MAE=%.3f  RMSE=%.3f  R²=%.4f  ±2mgdL=%.1f%%  bias=%+.3f",
             model_tag, split_name, mae, rmse, r2, within_2, bias)
    log.info("[%s|%s] pred: mean=%.2f std=%.2f  true: mean=%.2f std=%.2f",
             model_tag, split_name,
             preds.mean(), preds.std(), y_true.mean(), y_true.std())

    result = {
        "split": split_name, "model_tag": model_tag,
        "mae": float(mae), "rmse": rmse, "r2": float(r2),
        "bias_mean": bias,
        "within_2_mgdl_pct": within_2,
        "within_3_mgdl_pct": within_3,
        "within_5_mgdl_pct": within_5,
        "pred_mean": float(preds.mean()), "pred_std": float(preds.std()),
        "true_mean": float(y_true.mean()), "true_std": float(y_true.std()),
    }
    if LOGGING_ENABLED:
        _save_csv(
            pd.DataFrame({
                "y_true": y_true.values, "y_pred": preds,
                "abs_error": np.abs(y_true.values - preds),
            }),
            f"{model_tag}_predictions_{split_name}.csv",
        )
        _save_json(result, f"{model_tag}_eval_{split_name}.json")
    return result


# ── SHAP selection (cumulative 95%) ───────────────────────────────────────────

def _shap_select(model, X_val: pd.DataFrame, features: list,
                 model_tag: str, is_classifier: bool = True) -> list:
    """
    Select features whose cumulative mean |SHAP| accounts for
    SHAP_CUMULATIVE_THRESHOLD (95%) of total SHAP mass.

    This is self-calibrating: it does not depend on the absolute scale of
    the target variable and automatically handles redundancy by letting the
    model assign low SHAP to whichever of two correlated features it used
    less consistently.

    Expected outcome for this dataset: 15–25 features kept from ~40 inputs.
    """
    log.info("[%s] SHAP selection (cumulative %.0f%%) — %d × %d",
             model_tag, SHAP_CUMULATIVE_THRESHOLD * 100,
             X_val.shape[0], len(features))

    sv = shap.TreeExplainer(model).shap_values(X_val)
    if isinstance(sv, list):
        sv = sv[1]  # positive class for classifiers

    mean_shap = np.abs(sv).mean(axis=0)
    total_shap = mean_shap.sum()

    # sort descending, accumulate, cut at threshold
    order      = np.argsort(mean_shap)[::-1]
    cumulative = np.cumsum(mean_shap[order]) / total_shap
    n_keep     = int(np.searchsorted(cumulative, SHAP_CUMULATIVE_THRESHOLD)) + 1

    selected_idx = order[:n_keep]
    selected = [features[i] for i in selected_idx]
    dropped  = [features[i] for i in order[n_keep:]]

    # log the full SHAP table
    shap_df = pd.DataFrame({
        "feature":       features,
        "mean_abs_shap": mean_shap.tolist(),
        "pct_of_top":    (mean_shap / mean_shap.max() * 100).tolist(),
        "cumulative_pct": (
            pd.Series(mean_shap)
            .rank(ascending=False)
            .map(lambda r: cumulative[int(r)-1] * 100) # type: ignore
        ).tolist(),
        "selected": [f in selected for f in features],
    }).sort_values("mean_abs_shap", ascending=False)

    log.info("[%s] top feature mean|SHAP|=%.4f  total SHAP mass=%.4f",
             model_tag, mean_shap.max(), total_shap)
    log.info("[%s] kept %d/%d features at cumulative %.0f%% SHAP mass",
             model_tag, len(selected), len(features),
             SHAP_CUMULATIVE_THRESHOLD * 100)
    log.info("[%s] dropped (%d): %s", model_tag, len(dropped), dropped)

    if LOGGING_ENABLED:
        _save_csv(shap_df, f"{model_tag}_shap_summary.csv")
        _save_json({
            "model_tag": model_tag,
            "cumulative_threshold": SHAP_CUMULATIVE_THRESHOLD,
            "n_in": len(features), "n_kept": len(selected),
            "total_shap_mass": float(total_shap),
            "top_feature_shap": float(mean_shap.max()),
            "selected": selected, "dropped": dropped,
        }, f"{model_tag}_shap_selection.json")

    return selected


# ── model save ────────────────────────────────────────────────────────────────

def save_model(model, features: list, name: str, model_type: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": model, "features": features, "model_type": model_type}, f)
    size_kb = os.path.getsize(path) / 1024
    log.info("[save] %s  (%.1f KB)  type=%s  features=%d", path, size_kb, model_type, len(features))
    if LOGGING_ENABLED:
        _save_json({
            "name": name, "path": path, "size_kb": size_kb,
            "n_features": len(features), "features": features, "model_type": model_type,
        }, f"{name}_saved_model_info.json")


# ── training loop ─────────────────────────────────────────────────────────────

all_results = {}

for tag, label, trainer, evaluator, is_clf in [
    ("1", DETECT_LABEL, train_binary_clf,    evaluate_binary,     True),
    ("2", TSB_COL,      train_tsb_regressor, evaluate_regression, False),
]:
    model_type = "binary" if is_clf else "tsb_regressor"
    log.info("─── model %s ───", tag)

    # pass 1: full feature set → SHAP selection
    m_full = trainer(
        train_df[ALL_FEATURES], train_df[label],
        val_df[ALL_FEATURES],   val_df[label],
        model_tag=f"{tag}_full",
    )
    feats = _shap_select(
        m_full, val_df[ALL_FEATURES], ALL_FEATURES, # type: ignore
        model_tag=tag, is_classifier=is_clf,
    )

    # pass 2: retrain on SHAP-selected features only
    m_final = trainer(
        train_df[feats], train_df[label],
        val_df[feats],   val_df[label],
        model_tag=f"{tag}_final",
    )

    r_val  = evaluator(m_final, val_df[feats],  val_df[label],  "Val",  model_tag=tag)
    r_test = evaluator(m_final, test_df[feats], test_df[label], "Test", model_tag=tag)
    save_model(m_final, feats, f"model_{tag}", model_type=model_type)

    all_results[tag] = {
        "val": r_val, "test": r_test,
        "n_feat": len(feats), "features": feats,
    }


# ── summary table ─────────────────────────────────────────────────────────────

log.info("─── summary ───")
log.info("%-6s %5s | %8s %8s %7s | %9s %9s %8s",
         "Model", "Feat", "Val Acc", "Val AUC", "Val F1",
         "Test Acc", "Test AUC", "Test F1")
r = all_results["1"]
v, t = r["val"], r["test"]
log.info("%-6s %5d | %7.2f%% %7.2f%% %6.2f%% | %8.2f%% %8.2f%% %7.2f%%",
         "1", r["n_feat"],
         v["accuracy"]*100, v["auc_roc"]*100, v["f1"]*100,
         t["accuracy"]*100, t["auc_roc"]*100, t["f1"]*100)

log.info("%-6s %5s | %9s %9s %7s %8s | %9s %9s %7s %8s",
         "Model", "Feat", "Val MAE", "Val RMSE", "Val R²", "±2mgdL%",
         "Test MAE", "Test RMSE", "Test R²", "±2mgdL%")
r = all_results["2"]
v, t = r["val"], r["test"]
log.info("%-6s %5d | %9.3f %9.3f %7.4f %7.1f%% | %9.3f %9.3f %7.4f %7.1f%%",
         "2", r["n_feat"],
         v["mae"], v["rmse"], v["r2"], v["within_2_mgdl_pct"],
         t["mae"], t["rmse"], t["r2"], t["within_2_mgdl_pct"])

if LOGGING_ENABLED:
    _save_json({
        "run_timestamp": RUN_TIMESTAMP,
        "shap_cumulative_threshold": SHAP_CUMULATIVE_THRESHOLD,
        "lgb_binary_params": LGB_BINARY_PARAMS,
        "lgb_regression_params": LGB_REGRESSION_PARAMS,
        "results": {
            k: {
                "n_features": all_results[k]["n_feat"],
                "features": all_results[k]["features"],
                "val":  all_results[k]["val"],
                "test": all_results[k]["test"],
            }
            for k in ["1", "2"]
        },
    }, "00_MASTER_SUMMARY.json")
    log.info("artifacts saved to: %s", LOG_DIR.resolve())  # type: ignore

log.info("models saved to: ./%s/", MODELS_DIR)