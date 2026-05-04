"""
evaluate.py — Full Evaluation + Plots
======================================
Loads all 4 saved models, runs full evaluation on test set,
prints classification reports, and saves ROC curve plots.

Run AFTER train_models.py.
Usage: python evaluate.py
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_PATH     = "__data__/neo/out/training.csv"
MODELS_DIR    = "models"
PLOTS_DIR     = "plots"
SEV_THRESHOLD = 15.0
DETECT_LABEL  = "jaundice_label"
TSB_COL       = "blood_mg_dl"

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load data & recreate test split ───────────────────────────
df = pd.read_csv(DATA_PATH)
COLOR_FEATURES = [c for c in df.columns if c.startswith("zone")]
META_FEATURES  = ["gestational_age", "postnatal_age_days", "weight"]
ALL_FEATURES   = COLOR_FEATURES + META_FEATURES

patients        = df[~df["is_augmented"]]["patient_id"].unique()
train_p, temp_p = train_test_split(patients, test_size=0.30, random_state=42)
val_p,  test_p  = train_test_split(temp_p,   test_size=0.50, random_state=42)

val_df  = df[df["patient_id"].isin(val_p)  & ~df["is_augmented"]].copy()
test_df = df[df["patient_id"].isin(test_p) & ~df["is_augmented"]].copy()

sev_val  = val_df[val_df[DETECT_LABEL]  == 1].copy()
sev_test = test_df[test_df[DETECT_LABEL] == 1].copy()
sev_val["sev_label"]  = (sev_val[TSB_COL]  >= SEV_THRESHOLD).astype(int)
sev_test["sev_label"] = (sev_test[TSB_COL] >= SEV_THRESHOLD).astype(int)

def load_model(name):
    with open(f"{MODELS_DIR}/{name}.pkl", "rb") as f:
        return pickle.load(f)

def safe_feats(feats, df):
    return [f for f in feats if f in df.columns]

configs = {
    "1A": (test_df,  DETECT_LABEL,  "Detection + Meta"),
    "1B": (test_df,  DETECT_LABEL,  "Detection Only"),
    "2A": (sev_test, "sev_label",   "Severity + Meta"),
    "2B": (sev_test, "sev_label",   "Severity Only"),
}

# ── Evaluate all models ────────────────────────────────────────
print("=" * 65)
print("FULL EVALUATION — Test Set")
print("=" * 65)

roc_data = {}
for key, (df_eval, label_col, desc) in configs.items():
    bundle = load_model(f"model_{key}")
    model  = bundle["model"]
    feats  = safe_feats(bundle["features"], df_eval)

    X = df_eval[feats]
    y = df_eval[label_col]

    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, proba)
    f1  = f1_score(y, preds, zero_division=0)
    fpr, tpr, _ = roc_curve(y, proba)
    roc_data[key] = (fpr, tpr, auc, desc)

    print(f"\n── Model {key}: {desc} ──")
    print(f"   Accuracy : {acc*100:.2f}%")
    print(f"   AUC      : {auc*100:.2f}%")
    print(f"   F1       : {f1*100:.2f}%")
    print(classification_report(y, preds,
          target_names=["Negative","Positive"], zero_division=0))

# ── Plot ROC curves ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ROC Curves — Neonatal Jaundice LightGBM Models (Plan v2)",
             fontsize=13, fontweight="bold")

for ax, (pair, title) in zip(axes, [
    (["1A", "1B"], "Detection Models"),
    (["2A", "2B"], "Severity Models"),
]):
    for key in pair:
        fpr, tpr, auc, desc = roc_data[key]
        ax.plot(fpr, tpr, lw=2, label=f"Model {key} — {desc} (AUC={auc*100:.1f}%)")
    ax.plot([0,1],[0,1], "k--", lw=1, label="Random (AUC=50%)")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
roc_path = os.path.join(PLOTS_DIR, "roc_curves.png")
plt.savefig(roc_path, dpi=150)
print(f"\nROC plot saved → {roc_path}")

# ── Plot Confusion Matrices ────────────────────────────────────
fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4))
fig2.suptitle("Confusion Matrices — Test Set", fontsize=13, fontweight="bold")

for ax, (key, (df_eval, label_col, desc)) in zip(axes2, configs.items()):
    bundle = load_model(f"model_{key}")
    feats  = safe_feats(bundle["features"], df_eval)
    preds  = bundle["model"].predict(df_eval[feats])
    y      = df_eval[label_col]
    cm     = confusion_matrix(y, preds)
    disp   = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Model {key}\n{desc}", fontsize=9)

plt.tight_layout()
cm_path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
plt.savefig(cm_path, dpi=150)
print(f"Confusion matrix saved → {cm_path}")

plt.show()
print("\nEvaluasi selesai.")
