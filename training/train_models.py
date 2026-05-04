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
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH      = "__data__/neo/out/training.csv"
MODELS_DIR     = "models"
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
# LOAD & SPLIT (patient-level 70/15/15)
# ─────────────────────────────────────────────
print("=" * 60)
print("LOADING DATA")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
print(f"  Total rows    : {len(df)}")
print(f"  Total columns : {len(df.columns)}")

COLOR_FEATURES = [c for c in df.columns if c.startswith("zone")]
META_FEATURES  = ["gestational_age", "postnatal_age_days", "weight"]
ALL_FEATURES   = COLOR_FEATURES + META_FEATURES

print(f"  Color features: {len(COLOR_FEATURES)}")
print(f"  Meta  features: {len(META_FEATURES)}")

patients        = df[~df["is_augmented"]]["patient_id"].unique()
train_p, temp_p = train_test_split(patients, test_size=0.30, random_state=42)
val_p,  test_p  = train_test_split(temp_p,   test_size=0.50, random_state=42)

train_df = df[df["patient_id"].isin(train_p)].copy()
val_df   = df[df["patient_id"].isin(val_p)  & ~df["is_augmented"]].copy()
test_df  = df[df["patient_id"].isin(test_p) & ~df["is_augmented"]].copy()

print(f"\n  Train: {len(train_p)} patients → {len(train_df)} rows (aug included)")
print(f"  Val  : {len(val_p)}  patients → {len(val_df)} rows (original only)")
print(f"  Test : {len(test_p)}  patients → {len(test_df)} rows (original only)")

# Severity subset — jaundiced patients only
sev_train = train_df[train_df[DETECT_LABEL] == 1].copy()
sev_val   = val_df[val_df[DETECT_LABEL]     == 1].copy()
sev_test  = test_df[test_df[DETECT_LABEL]   == 1].copy()

for s in [sev_train, sev_val, sev_test]:
    s["sev_label"] = (s[TSB_COL] >= SEV_THRESHOLD).astype(int)

print(f"\n  Severity train: {len(sev_train)} rows")
print(f"  Severity val  : {len(sev_val)}  rows")
print(f"  Severity test : {len(sev_test)}  rows")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def train_clf(X_tr, y_tr, X_val, y_val):
    m = lgb.LGBMClassifier(**LGB_PARAMS) # type: ignore
    m.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(-1)])
    return m

def evaluate(model, X, y_true, split_name):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, proba)
    f1  = f1_score(y_true, preds, zero_division=0)
    print(f"    [{split_name}]  Accuracy={acc*100:.2f}%  AUC={auc*100:.2f}%  F1={f1*100:.2f}%")
    return acc, auc, f1

def shap_select(model, X_val, features):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_val)
    if isinstance(sv, list):
        sv = sv[1]
    mean_shap = np.abs(sv).mean(axis=0)
    cutoff    = SHAP_CUTOFF * mean_shap.max()
    selected  = [f for f, s in zip(features, mean_shap) if s >= cutoff]
    dropped   = [f for f, s in zip(features, mean_shap) if s < cutoff]
    print(f"      SHAP: kept {len(selected)}/{len(features)} features, "
          f"dropped {len(dropped)}: {dropped}")
    return selected

def save_model(model, features, name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"model": model, "features": features}, f)
    print(f"      Saved → {path}")

all_results = {}

# ─────────────────────────────────────────────
# MODEL 1A — Detection + Metadata
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 1A — Detection (color + metadata)")
print("=" * 60)

m1A_full = train_clf(train_df[ALL_FEATURES], train_df[DETECT_LABEL],
                     val_df[ALL_FEATURES],   val_df[DETECT_LABEL])
feats_1A = shap_select(m1A_full, val_df[ALL_FEATURES], ALL_FEATURES)

m1A = train_clf(train_df[feats_1A], train_df[DETECT_LABEL],
                val_df[feats_1A],   val_df[DETECT_LABEL])

r1A_val  = evaluate(m1A, val_df[feats_1A],  val_df[DETECT_LABEL],  "Val ")
r1A_test = evaluate(m1A, test_df[feats_1A], test_df[DETECT_LABEL], "Test")
save_model(m1A, feats_1A, "model_1A")
all_results["1A"] = {"val": r1A_val, "test": r1A_test, "n_feat": len(feats_1A)}

# ─────────────────────────────────────────────
# MODEL 1B — Detection, Color Only
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 1B — Detection (color only)")
print("=" * 60)

m1B_full = train_clf(train_df[COLOR_FEATURES], train_df[DETECT_LABEL],
                     val_df[COLOR_FEATURES],   val_df[DETECT_LABEL])
feats_1B = shap_select(m1B_full, val_df[COLOR_FEATURES], COLOR_FEATURES)

m1B = train_clf(train_df[feats_1B], train_df[DETECT_LABEL],
                val_df[feats_1B],   val_df[DETECT_LABEL])

r1B_val  = evaluate(m1B, val_df[feats_1B],  val_df[DETECT_LABEL],  "Val ")
r1B_test = evaluate(m1B, test_df[feats_1B], test_df[DETECT_LABEL], "Test")
save_model(m1B, feats_1B, "model_1B")
all_results["1B"] = {"val": r1B_val, "test": r1B_test, "n_feat": len(feats_1B)}

# ─────────────────────────────────────────────
# MODEL 2A — Severity + Metadata
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 2A — Severity (color + metadata)")
print("=" * 60)

m2A_full = train_clf(sev_train[ALL_FEATURES], sev_train["sev_label"],
                     sev_val[ALL_FEATURES],   sev_val["sev_label"])
feats_2A = shap_select(m2A_full, sev_val[ALL_FEATURES], ALL_FEATURES)

m2A = train_clf(sev_train[feats_2A], sev_train["sev_label"],
                sev_val[feats_2A],   sev_val["sev_label"])

r2A_val  = evaluate(m2A, sev_val[feats_2A],  sev_val["sev_label"],  "Val ")
r2A_test = evaluate(m2A, sev_test[feats_2A], sev_test["sev_label"], "Test")
save_model(m2A, feats_2A, "model_2A")
all_results["2A"] = {"val": r2A_val, "test": r2A_test, "n_feat": len(feats_2A)}

# ─────────────────────────────────────────────
# MODEL 2B — Severity, Color Only
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 2B — Severity (color only)")
print("=" * 60)

m2B_full = train_clf(sev_train[COLOR_FEATURES], sev_train["sev_label"],
                     sev_val[COLOR_FEATURES],   sev_val["sev_label"])
feats_2B = shap_select(m2B_full, sev_val[COLOR_FEATURES], COLOR_FEATURES)

m2B = train_clf(sev_train[feats_2B], sev_train["sev_label"],
                sev_val[feats_2B],   sev_val["sev_label"])

r2B_val  = evaluate(m2B, sev_val[feats_2B],  sev_val["sev_label"],  "Val ")
r2B_test = evaluate(m2B, sev_test[feats_2B], sev_test["sev_label"], "Test")
save_model(m2B, feats_2B, "model_2B")
all_results["2B"] = {"val": r2B_val, "test": r2B_test, "n_feat": len(feats_2B)}

# ─────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  FINAL RESULTS SUMMARY — Plan v2 (SHAP-selected, Binary Classifier)")
print("=" * 70)
print(f"{'Model':<8} {'Feat':>5} | {'Val Acc':>8} {'Val AUC':>8} {'Val F1':>7} | "
      f"{'Test Acc':>9} {'Test AUC':>9} {'Test F1':>8}")
print("-" * 70)
labels = {
    "1A": "1A  (detection + meta)",
    "1B": "1B  (detection only) ",
    "2A": "2A  (severity + meta)",
    "2B": "2B  (severity only)  ",
}
for k, desc in labels.items():
    r = all_results[k]
    va, vu, vf = r["val"]
    ta, tu, tf = r["test"]
    print(f"{desc}  {r['n_feat']:>3} | "
          f"{va*100:>7.2f}% {vu*100:>7.2f}% {vf*100:>6.2f}% | "
          f"{ta*100:>8.2f}% {tu*100:>8.2f}% {tf*100:>7.2f}%")
print("=" * 70)
print("\nSemua model tersimpan di folder: ./models/")
print("Gunakan predict.py untuk inferensi pada data baru.")