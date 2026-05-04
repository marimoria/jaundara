"""
train_models.py — Neonatal Jaundice Detection (Plan v2)
========================================================
Trains 4 LightGBM models with SHAP feature selection.
Place this file at project root. Run: python train_models.py

Models saved to: ./models/
  model_1A.pkl  — Detection + metadata
  model_1B.pkl  — Detection, color only
  model_2A.pkl  — Severity + metadata
  model_2B.pkl  — Severity, color only

Logging:
  python train_models.py              # no log file
  python train_models.py --log        # writes to __data__/models_log/<timestamp>/
  python train_models.py --log --log-dir path/to/custom/dir
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
    precision_recall_curve,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
# CLI ARGUMENTS
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train neonatal jaundice detection models.")
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

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH      = "__data__/neo/out/training.csv"
MODELS_DIR     = "__models__"
DETECT_LABEL   = "jaundice_label"
TSB_COL        = "blood_mg_dl"
SEV_THRESHOLD  = 15.0    # mg/dL — Mild vs Severe boundary
SHAP_CUTOFF    = 0.01    # drop features below 1% of top SHAP value

LGB_PARAMS = dict(
    boosting_type="gbdt",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if LOGGING_ENABLED:
    if args.log_dir:
        LOG_DIR = Path(args.log_dir)
    else:
        LOG_DIR = Path("__data__") / "models_log" / RUN_TIMESTAMP
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Main text log — every print also goes here
    log_file_path = LOG_DIR / "run.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("train_models")
    log.info(f"Log directory  : {LOG_DIR.resolve()}")
    log.info(f"Run timestamp  : {RUN_TIMESTAMP}")
    log.info(f"Python argv    : {sys.argv}")
    log.info(f"LGB_PARAMS     : {json.dumps(LGB_PARAMS, indent=2)}")
    log.info(f"SHAP_CUTOFF    : {SHAP_CUTOFF}")
    log.info(f"SEV_THRESHOLD  : {SEV_THRESHOLD} mg/dL")
else:
    # Fallback: logging to stdout only, no file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("train_models")
    LOG_DIR = None

def _save_json(obj, filename: str):
    """Save a JSON-serialisable object to LOG_DIR if logging is enabled."""
    if not LOGGING_ENABLED:
        return
    out = LOG_DIR / filename # type: ignore
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    log.debug(f"  [artifact] Saved JSON → {out.name}")

def _save_csv(df: pd.DataFrame, filename: str):
    """Save a DataFrame to LOG_DIR as CSV if logging is enabled."""
    if not LOGGING_ENABLED:
        return
    out = LOG_DIR / filename # type: ignore
    df.to_csv(out, index=False)
    log.debug(f"  [artifact] Saved CSV  → {out.name}")

# ─────────────────────────────────────────────
# LOAD & SPLIT (patient-level 70/15/15)
# ─────────────────────────────────────────────
log.info("=" * 70)
log.info("STEP 1 — LOADING DATA")
log.info("=" * 70)
log.info(f"  Reading: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

log.info(f"  Total rows          : {len(df)}")
log.info(f"  Total columns       : {len(df.columns)}")
log.info(f"  Columns             : {list(df.columns)}")
log.info(f"  Dtypes summary      :\n{df.dtypes.to_string()}")
log.info(f"  Missing values      :\n{df.isnull().sum().to_string()}")
log.info(f"  Label distribution  ({DETECT_LABEL}):")
vc = df[DETECT_LABEL].value_counts()
for val, cnt in vc.items():
    log.info(f"    {val}: {cnt}  ({cnt/len(df)*100:.1f}%)")

_save_json(
    {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "label_distribution": vc.to_dict(),
    },
    "01_data_load_summary.json",
)

COLOR_FEATURES = [c for c in df.columns if c.startswith("zone")]
META_FEATURES  = ["gestational_age", "postnatal_age_days", "weight"]
ALL_FEATURES   = COLOR_FEATURES + META_FEATURES

log.info(f"\n  Color features ({len(COLOR_FEATURES)}): {COLOR_FEATURES}")
log.info(f"  Meta  features ({len(META_FEATURES)}): {META_FEATURES}")
log.info(f"  ALL   features  total: {len(ALL_FEATURES)}")

log.info("\n" + "─" * 70)
log.info("STEP 2 — PATIENT-LEVEL TRAIN/VAL/TEST SPLIT (70 / 15 / 15)")
log.info("─" * 70)

patients        = df[~df["is_augmented"]]["patient_id"].unique()
train_p, temp_p = train_test_split(patients, test_size=0.30, random_state=42)
val_p,  test_p  = train_test_split(temp_p,   test_size=0.50, random_state=42)

train_df = df[df["patient_id"].isin(train_p)].copy()
val_df   = df[df["patient_id"].isin(val_p)  & ~df["is_augmented"]].copy()
test_df  = df[df["patient_id"].isin(test_p) & ~df["is_augmented"]].copy()

log.info(f"  Original (non-augmented) patients  : {len(patients)}")
log.info(f"  Train patients : {len(train_p)}  → {len(train_df)} rows (augmented included)")
log.info(f"  Val   patients : {len(val_p)}   → {len(val_df)} rows (original only)")
log.info(f"  Test  patients : {len(test_p)}   → {len(test_df)} rows (original only)")
log.info(f"\n  Train label distribution:")
for val, cnt in train_df[DETECT_LABEL].value_counts().items():
    log.info(f"    {val}: {cnt}  ({cnt/len(train_df)*100:.1f}%)")
log.info(f"  Val label distribution:")
for val, cnt in val_df[DETECT_LABEL].value_counts().items():
    log.info(f"    {val}: {cnt}  ({cnt/len(val_df)*100:.1f}%)")
log.info(f"  Test label distribution:")
for val, cnt in test_df[DETECT_LABEL].value_counts().items():
    log.info(f"    {val}: {cnt}  ({cnt/len(test_df)*100:.1f}%)")

_save_json(
    {
        "n_original_patients": len(patients),
        "train_patients": len(train_p),
        "val_patients": len(val_p),
        "test_patients": len(test_p),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_label_dist": train_df[DETECT_LABEL].value_counts().to_dict(),
        "val_label_dist":   val_df[DETECT_LABEL].value_counts().to_dict(),
        "test_label_dist":  test_df[DETECT_LABEL].value_counts().to_dict(),
    },
    "02_split_summary.json",
)

# ─────────────────────────────────────────────
# SEVERITY SUBSET
# ─────────────────────────────────────────────
log.info("\n" + "─" * 70)
log.info("STEP 3 — BUILD SEVERITY SUBSETS (jaundiced patients only)")
log.info("─" * 70)

sev_train = train_df[train_df[DETECT_LABEL] == 1].copy()
sev_val   = val_df[val_df[DETECT_LABEL]     == 1].copy()
sev_test  = test_df[test_df[DETECT_LABEL]   == 1].copy()

for s in [sev_train, sev_val, sev_test]:
    s["sev_label"] = (s[TSB_COL] >= SEV_THRESHOLD).astype(int)

log.info(f"  Severity train rows : {len(sev_train)}")
log.info(f"  Severity val   rows : {len(sev_val)}")
log.info(f"  Severity test  rows : {len(sev_test)}")
log.info(f"  Severity threshold  : TSB >= {SEV_THRESHOLD} mg/dL  →  label=1 (Severe)")
log.info(f"  Sev_train label dist: {sev_train['sev_label'].value_counts().to_dict()}")
log.info(f"  Sev_val   label dist: {sev_val['sev_label'].value_counts().to_dict()}")
log.info(f"  Sev_test  label dist: {sev_test['sev_label'].value_counts().to_dict()}")
log.info(f"\n  TSB statistics (all jaundiced patients in train):")
log.info(f"    min={sev_train[TSB_COL].min():.2f}  max={sev_train[TSB_COL].max():.2f}  "
         f"mean={sev_train[TSB_COL].mean():.2f}  std={sev_train[TSB_COL].std():.2f}")

_save_json(
    {
        "sev_threshold_mgdl": SEV_THRESHOLD,
        "sev_train_rows": len(sev_train),
        "sev_val_rows":   len(sev_val),
        "sev_test_rows":  len(sev_test),
        "sev_train_label_dist": sev_train["sev_label"].value_counts().to_dict(),
        "sev_val_label_dist":   sev_val["sev_label"].value_counts().to_dict(),
        "sev_test_label_dist":  sev_test["sev_label"].value_counts().to_dict(),
        "tsb_stats_train": {
            "min":  float(sev_train[TSB_COL].min()),
            "max":  float(sev_train[TSB_COL].max()),
            "mean": float(sev_train[TSB_COL].mean()),
            "std":  float(sev_train[TSB_COL].std()),
        },
    },
    "03_severity_subset_summary.json",
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def train_clf(X_tr, y_tr, X_val, y_val, model_tag: str = ""):
    """Train a LightGBM classifier with early stopping, logging every detail."""
    log.info(f"    [{model_tag}] Training LGBMClassifier")
    log.info(f"    [{model_tag}] Train shape : {X_tr.shape}  |  positive rate: {y_tr.mean()*100:.1f}%")
    log.info(f"    [{model_tag}] Val   shape : {X_val.shape}  |  positive rate: {y_val.mean()*100:.1f}%")
    log.info(f"    [{model_tag}] Features    : {list(X_tr.columns)}")
    log.info(f"    [{model_tag}] LGB params  : {LGB_PARAMS}")

    m = lgb.LGBMClassifier(**LGB_PARAMS)  # type: ignore
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    log.info(f"    [{model_tag}] Best iteration : {m.best_iteration_}")
    log.info(f"    [{model_tag}] Best val score : {m.best_score_}")
    log.info(f"    [{model_tag}] N estimators   : {m.n_estimators_}")

    # Feature importances (gain + split)
    fi_gain  = dict(zip(X_tr.columns, m.booster_.feature_importance(importance_type="gain")))
    fi_split = dict(zip(X_tr.columns, m.booster_.feature_importance(importance_type="split")))
    log.info(f"    [{model_tag}] Feature importance (gain, top 10):")
    for feat, val in sorted(fi_gain.items(), key=lambda x: -x[1])[:10]:
        log.info(f"        {feat:<30s}  gain={val:.4f}  split={fi_split[feat]}")

    if LOGGING_ENABLED:
        _save_json(
            {
                "model_tag": model_tag,
                "best_iteration": m.best_iteration_,
                "best_score": m.best_score_,
                "n_estimators": m.n_estimators_,
                "train_shape": list(X_tr.shape),
                "val_shape": list(X_val.shape),
                "feature_importance_gain": fi_gain,
                "feature_importance_split": fi_split,
                "features": list(X_tr.columns),
            },
            f"{model_tag}_training_detail.json",
        )

    return m


def evaluate(model, X, y_true, split_name: str, model_tag: str = "") -> dict:
    """
    Full evaluation: accuracy, AUC, F1, precision, recall,
    confusion matrix, classification report, ROC curve points.
    Returns a dict of all metrics.
    """
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    acc  = accuracy_score(y_true, preds)
    auc  = roc_auc_score(y_true, proba)
    f1   = f1_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    cm   = confusion_matrix(y_true, preds)
    cr   = classification_report(y_true, preds, zero_division=0)

    # ROC curve
    fpr, tpr, roc_thresh = roc_curve(y_true, proba)
    # Precision-Recall curve
    prec_curve, rec_curve, pr_thresh = precision_recall_curve(y_true, proba)

    # Optimal threshold by Youden's J statistic
    j_scores = tpr - fpr
    best_thresh_idx = np.argmax(j_scores)
    best_thresh = float(roc_thresh[best_thresh_idx])

    log.info(f"    [{model_tag}|{split_name}]  Accuracy={acc*100:.2f}%  "
             f"AUC={auc*100:.2f}%  F1={f1*100:.2f}%  "
             f"Prec={prec*100:.2f}%  Recall={rec*100:.2f}%")
    log.info(f"    [{model_tag}|{split_name}]  Optimal threshold (Youden J): {best_thresh:.4f}")
    log.info(f"    [{model_tag}|{split_name}]  Confusion matrix:\n"
             f"        TN={cm[0,0]}  FP={cm[0,1]}\n"
             f"        FN={cm[1,0]}  TP={cm[1,1]}")
    log.info(f"    [{model_tag}|{split_name}]  Classification report:\n{cr}")

    result = {
        "split": split_name,
        "model_tag": model_tag,
        "accuracy":  float(acc),
        "auc_roc":   float(auc),
        "f1":        float(f1),
        "precision": float(prec),
        "recall":    float(rec),
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
        "optimal_threshold_youden": best_thresh,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresh.tolist(),
        },
        "pr_curve": {
            "precision": prec_curve.tolist(),
            "recall":    rec_curve.tolist(),
            "thresholds": pr_thresh.tolist(),
        },
    }

    if LOGGING_ENABLED:
        _save_json(result, f"{model_tag}_eval_{split_name.strip()}.json")

    return result


def shap_select(model, X_val: pd.DataFrame, features: list, model_tag: str = "") -> list:
    """
    Compute SHAP values on the validation set.
    Drop features whose mean |SHAP| < SHAP_CUTOFF * max(mean |SHAP|).
    Logs every feature's SHAP value and saves full SHAP matrix as CSV.
    Returns list of selected feature names.
    """
    log.info(f"    [{model_tag}] Computing SHAP values on val set ({X_val.shape[0]} rows × {len(features)} features)")

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_val)

    # For binary classifier lgb, shap_values returns shape (n_samples, n_features)
    # or a list of 2 arrays [class_0, class_1]
    if isinstance(sv, list):
        log.info(f"    [{model_tag}] shap_values is list of length {len(sv)}, using index [1] (positive class)")
        sv = sv[1]
    else:
        log.info(f"    [{model_tag}] shap_values shape: {sv.shape}")

    mean_shap = np.abs(sv).mean(axis=0)
    max_shap  = mean_shap.max()
    cutoff    = SHAP_CUTOFF * max_shap

    log.info(f"    [{model_tag}] SHAP max value     : {max_shap:.6f}")
    log.info(f"    [{model_tag}] SHAP cutoff (1%)   : {cutoff:.6f}")
    log.info(f"    [{model_tag}] SHAP values per feature (sorted desc):")

    shap_per_feature = sorted(
        zip(features, mean_shap.tolist()),
        key=lambda x: -x[1],
    )
    for rank, (feat, val) in enumerate(shap_per_feature, 1):
        status = "KEEP" if val >= cutoff else "DROP"
        log.info(f"        #{rank:02d}  {feat:<30s}  mean|SHAP|={val:.6f}  [{status}]")

    selected = [f for f, s in zip(features, mean_shap) if s >= cutoff]
    dropped  = [f for f, s in zip(features, mean_shap) if s < cutoff]

    log.info(f"    [{model_tag}] Kept  : {len(selected)}/{len(features)} features")
    log.info(f"    [{model_tag}] Dropped ({len(dropped)}): {dropped}")

    if LOGGING_ENABLED:
        # Save full SHAP matrix (rows = validation samples, cols = features)
        shap_df = pd.DataFrame(sv, columns=features)
        shap_df.index.name = "val_sample_idx"
        _save_csv(shap_df, f"{model_tag}_shap_matrix.csv")

        # Save per-feature SHAP summary
        shap_summary = pd.DataFrame(
            {
                "feature":        features,
                "mean_abs_shap":  mean_shap.tolist(),
                "pct_of_top":     (mean_shap / max_shap * 100).tolist(),
                "selected":       [s >= cutoff for s in mean_shap],
            }
        ).sort_values("mean_abs_shap", ascending=False)
        _save_csv(shap_summary, f"{model_tag}_shap_summary.csv")

        _save_json(
            {
                "model_tag":    model_tag,
                "n_features_in":  len(features),
                "n_features_kept": len(selected),
                "n_features_dropped": len(dropped),
                "shap_cutoff_value": float(cutoff),
                "shap_max_value":    float(max_shap),
                "selected_features": selected,
                "dropped_features":  dropped,
                "shap_per_feature_sorted": [
                    {"rank": i+1, "feature": f, "mean_abs_shap": float(v)}
                    for i, (f, v) in enumerate(shap_per_feature)
                ],
            },
            f"{model_tag}_shap_selection.json",
        )

    return selected


def save_model(model, features: list, name: str):
    """Pickle model + feature list, log file size."""
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    payload = {"model": model, "features": features}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    size_kb = os.path.getsize(path) / 1024
    log.info(f"    [save] Model → {path}  ({size_kb:.1f} KB)")
    log.info(f"    [save] Features ({len(features)}): {features}")

    if LOGGING_ENABLED:
        _save_json(
            {
                "model_name": name,
                "pkl_path": path,
                "pkl_size_kb": size_kb,
                "n_features": len(features),
                "features": features,
            },
            f"{name}_saved_model_info.json",
        )


all_results = {}

# ─────────────────────────────────────────────
# MODEL 1A — Detection + Metadata
# ─────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("MODEL 1A — Detection (color + metadata)")
log.info("=" * 70)

log.info("  [1A] Phase 1: Full model with all 42+3 features")
m1A_full = train_clf(
    train_df[ALL_FEATURES], train_df[DETECT_LABEL],
    val_df[ALL_FEATURES],   val_df[DETECT_LABEL],
    model_tag="1A_full",
)

log.info("  [1A] Phase 2: SHAP feature selection")
feats_1A = shap_select(m1A_full, val_df[ALL_FEATURES], ALL_FEATURES, model_tag="1A")

log.info(f"  [1A] Phase 3: Retrain on {len(feats_1A)} SHAP-selected features")
m1A = train_clf(
    train_df[feats_1A], train_df[DETECT_LABEL],
    val_df[feats_1A],   val_df[DETECT_LABEL],
    model_tag="1A_final",
)

log.info("  [1A] Phase 4: Evaluation")
r1A_val  = evaluate(m1A, val_df[feats_1A],  val_df[DETECT_LABEL],  "Val",  model_tag="1A")
r1A_test = evaluate(m1A, test_df[feats_1A], test_df[DETECT_LABEL], "Test", model_tag="1A")

log.info(f"  [1A] Val→Test AUC gap: {(r1A_val['auc_roc'] - r1A_test['auc_roc'])*100:.2f}%")

save_model(m1A, feats_1A, "model_1A")
all_results["1A"] = {
    "val":    r1A_val,
    "test":   r1A_test,
    "n_feat": len(feats_1A),
    "features": feats_1A,
}

# ─────────────────────────────────────────────
# MODEL 1B — Detection, Color Only
# ─────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("MODEL 1B — Detection (color only)")
log.info("=" * 70)

log.info("  [1B] Phase 1: Full model with all 42 color features")
m1B_full = train_clf(
    train_df[COLOR_FEATURES], train_df[DETECT_LABEL],
    val_df[COLOR_FEATURES],   val_df[DETECT_LABEL],
    model_tag="1B_full",
)

log.info("  [1B] Phase 2: SHAP feature selection")
feats_1B = shap_select(m1B_full, val_df[COLOR_FEATURES], COLOR_FEATURES, model_tag="1B")

log.info(f"  [1B] Phase 3: Retrain on {len(feats_1B)} SHAP-selected features")
m1B = train_clf(
    train_df[feats_1B], train_df[DETECT_LABEL],
    val_df[feats_1B],   val_df[DETECT_LABEL],
    model_tag="1B_final",
)

log.info("  [1B] Phase 4: Evaluation")
r1B_val  = evaluate(m1B, val_df[feats_1B],  val_df[DETECT_LABEL],  "Val",  model_tag="1B")
r1B_test = evaluate(m1B, test_df[feats_1B], test_df[DETECT_LABEL], "Test", model_tag="1B")

log.info(f"  [1B] Val→Test AUC gap: {(r1B_val['auc_roc'] - r1B_test['auc_roc'])*100:.2f}%")

save_model(m1B, feats_1B, "model_1B")
all_results["1B"] = {
    "val":    r1B_val,
    "test":   r1B_test,
    "n_feat": len(feats_1B),
    "features": feats_1B,
}

# ─────────────────────────────────────────────
# MODEL 2A — Severity + Metadata
# ─────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("MODEL 2A — Severity (color + metadata)")
log.info("=" * 70)

log.info("  [2A] Phase 1: Full model with all 42+3 features (jaundiced only)")
m2A_full = train_clf(
    sev_train[ALL_FEATURES], sev_train["sev_label"],
    sev_val[ALL_FEATURES],   sev_val["sev_label"],
    model_tag="2A_full",
)

log.info("  [2A] Phase 2: SHAP feature selection")
feats_2A = shap_select(m2A_full, sev_val[ALL_FEATURES], ALL_FEATURES, model_tag="2A")

log.info(f"  [2A] Phase 3: Retrain on {len(feats_2A)} SHAP-selected features")
m2A = train_clf(
    sev_train[feats_2A], sev_train["sev_label"],
    sev_val[feats_2A],   sev_val["sev_label"],
    model_tag="2A_final",
)

log.info("  [2A] Phase 4: Evaluation")
r2A_val  = evaluate(m2A, sev_val[feats_2A],  sev_val["sev_label"],  "Val",  model_tag="2A")
r2A_test = evaluate(m2A, sev_test[feats_2A], sev_test["sev_label"], "Test", model_tag="2A")

log.info(f"  [2A] Val→Test AUC gap: {(r2A_val['auc_roc'] - r2A_test['auc_roc'])*100:.2f}%")

save_model(m2A, feats_2A, "model_2A")
all_results["2A"] = {
    "val":    r2A_val,
    "test":   r2A_test,
    "n_feat": len(feats_2A),
    "features": feats_2A,
}

# ─────────────────────────────────────────────
# MODEL 2B — Severity, Color Only
# ─────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("MODEL 2B — Severity (color only)")
log.info("=" * 70)

log.info("  [2B] Phase 1: Full model with all 42 color features (jaundiced only)")
m2B_full = train_clf(
    sev_train[COLOR_FEATURES], sev_train["sev_label"],
    sev_val[COLOR_FEATURES],   sev_val["sev_label"],
    model_tag="2B_full",
)

log.info("  [2B] Phase 2: SHAP feature selection")
feats_2B = shap_select(m2B_full, sev_val[COLOR_FEATURES], COLOR_FEATURES, model_tag="2B")

log.info(f"  [2B] Phase 3: Retrain on {len(feats_2B)} SHAP-selected features")
m2B = train_clf(
    sev_train[feats_2B], sev_train["sev_label"],
    sev_val[feats_2B],   sev_val["sev_label"],
    model_tag="2B_final",
)

log.info("  [2B] Phase 4: Evaluation")
r2B_val  = evaluate(m2B, sev_val[feats_2B],  sev_val["sev_label"],  "Val",  model_tag="2B")
r2B_test = evaluate(m2B, sev_test[feats_2B], sev_test["sev_label"], "Test", model_tag="2B")

log.info(f"  [2B] Val→Test AUC gap: {(r2B_val['auc_roc'] - r2B_test['auc_roc'])*100:.2f}%")

save_model(m2B, feats_2B, "model_2B")
all_results["2B"] = {
    "val":    r2B_val,
    "test":   r2B_test,
    "n_feat": len(feats_2B),
    "features": feats_2B,
}

# ─────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────
log.info("\n\n" + "=" * 80)
log.info("  FINAL RESULTS SUMMARY — Plan v2 (SHAP-selected, Binary Classifier)")
log.info("=" * 80)
log.info(f"{'Model':<26} {'Feat':>5} | {'Val Acc':>8} {'Val AUC':>8} {'Val F1':>7} | "
         f"{'Test Acc':>9} {'Test AUC':>9} {'Test F1':>8}")
log.info("-" * 80)

labels = {
    "1A": "1A  (detection + meta) ",
    "1B": "1B  (detection only)  ",
    "2A": "2A  (severity + meta) ",
    "2B": "2B  (severity only)   ",
}
for k, desc in labels.items():
    r  = all_results[k]
    va = r["val"]["accuracy"];  vu = r["val"]["auc_roc"];  vf = r["val"]["f1"]
    ta = r["test"]["accuracy"]; tu = r["test"]["auc_roc"]; tf = r["test"]["f1"]
    log.info(
        f"{desc}  {r['n_feat']:>3} | "
        f"{va*100:>7.2f}% {vu*100:>7.2f}% {vf*100:>6.2f}% | "
        f"{ta*100:>8.2f}% {tu*100:>8.2f}% {tf*100:>7.2f}%"
    )

log.info("=" * 80)

# ─────────────────────────────────────────────
# SAVE MASTER SUMMARY TO LOG DIR
# ─────────────────────────────────────────────
if LOGGING_ENABLED:
    # Compact summary JSON (without full ROC/PR arrays to keep it readable)
    summary = {}
    for k in ["1A", "1B", "2A", "2B"]:
        r = all_results[k]
        summary[k] = {
            "n_features": r["n_feat"],
            "features_selected": r["features"],
            "val": {
                "accuracy":  r["val"]["accuracy"],
                "auc_roc":   r["val"]["auc_roc"],
                "f1":        r["val"]["f1"],
                "precision": r["val"]["precision"],
                "recall":    r["val"]["recall"],
                "confusion_matrix": r["val"]["confusion_matrix"],
            },
            "test": {
                "accuracy":  r["test"]["accuracy"],
                "auc_roc":   r["test"]["auc_roc"],
                "f1":        r["test"]["f1"],
                "precision": r["test"]["precision"],
                "recall":    r["test"]["recall"],
                "confusion_matrix": r["test"]["confusion_matrix"],
            },
            "val_test_auc_gap": round(r["val"]["auc_roc"] - r["test"]["auc_roc"], 4),
        }

    _save_json(
        {
            "run_timestamp": RUN_TIMESTAMP,
            "lgb_params": LGB_PARAMS,
            "shap_cutoff": SHAP_CUTOFF,
            "sev_threshold_mgdl": SEV_THRESHOLD,
            "results": summary,
        },
        "00_MASTER_SUMMARY.json",
    )

    log.info(f"\n  All log artifacts saved to: {LOG_DIR.resolve()}") # type: ignore
    log.info(f"  Files in log dir:")
    for f in sorted(LOG_DIR.iterdir()): # type: ignore
        size = f.stat().st_size / 1024
        log.info(f"    {f.name:<50s}  {size:>8.1f} KB")

log.info("\nSemua model tersimpan di folder: ./models/")
log.info("Gunakan predict.py untuk inferensi pada data baru.")
log.info("Jalankan dengan --log untuk menyimpan semua artefak log ke __data__/models_log/")