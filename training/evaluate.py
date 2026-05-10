"""
evaluate_v4.py — Full Evaluation + Plots (Plan v4)
====================================================
Loads all 4 saved models:
  model_1A / 1B  — binary detection classifiers
  model_2A / 2B  — TSB regressors (predict blood_mg_dl)

Run AFTER train_models_v4.py.
Usage: python evaluate_v4.py
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
    mean_absolute_error, mean_squared_error, r2_score,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_PATH    = "__data__/neo/out/training.csv"
MODELS_DIR   = "__models__"
PLOTS_DIR    = "__plots__"
DETECT_LABEL = "jaundice_label"
TSB_COL      = "blood_mg_dl"

os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
COLOR_FEATURES = [c for c in df.columns if c.startswith("zone")]
META_FEATURES  = ["gestational_age", "postnatal_age_days", "weight"]
ALL_FEATURES   = COLOR_FEATURES + META_FEATURES

# Patient-level split — must match train_models_v4.py exactly
patients        = df[~df["is_augmented"]]["patient_id"].unique()
train_p, temp_p = train_test_split(patients, test_size=0.30, random_state=42)
val_p,  test_p  = train_test_split(temp_p,   test_size=0.50, random_state=42)

test_df     = df[df["patient_id"].isin(test_p) & ~df["is_augmented"]].copy()
reg_test_df = test_df


def load_model(name):
    with open(f"{MODELS_DIR}/{name}.pkl", "rb") as f:
        return pickle.load(f)


def safe_feats(feats, df):
    return [f for f in feats if f in df.columns]


print("=" * 70)
print("FULL EVALUATION — Test Set (Plan v4)")
print("=" * 70)

roc_data = {}

# ── Binary models 1A / 1B ───────────────────────────────────────────────
for key, desc in [("1A", "Detection + Meta"), ("1B", "Detection Only")]:
    bundle = load_model(f"model_{key}")
    feats  = safe_feats(bundle["features"], test_df)
    X, y   = test_df[feats], test_df[DETECT_LABEL]

    preds = bundle["model"].predict(X)
    proba = bundle["model"].predict_proba(X)[:, 1]

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
          target_names=["Normal", "Jaundiced"], zero_division=0))

# ── TSB Regression models 2A / 2B ───────────────────────────────────────
reg_results = {}
for key, desc in [("2A", "TSB Regressor + Meta"), ("2B", "TSB Regressor Only")]:
    bundle = load_model(f"model_{key}")
    feats  = safe_feats(bundle["features"], reg_test_df)
    X      = reg_test_df[feats]
    y_true = reg_test_df[TSB_COL]

    preds = np.clip(bundle["model"].predict(X), 0.0, 40.0)

    mae  = mean_absolute_error(y_true, preds)
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    r2   = r2_score(y_true, preds)
    w2   = float(np.mean(np.abs(y_true - preds) <= 2.0) * 100)
    w3   = float(np.mean(np.abs(y_true - preds) <= 3.0) * 100)
    bias = float(np.mean(preds - y_true))

    print(f"\n── Model {key}: {desc} ──")
    print(f"   MAE         : {mae:.3f} mg/dL")
    print(f"   RMSE        : {rmse:.3f} mg/dL")
    print(f"   R²          : {r2:.4f}")
    print(f"   Within ±2   : {w2:.1f}%")
    print(f"   Within ±3   : {w3:.1f}%")
    print(f"   Bias        : {bias:+.3f} mg/dL")

    reg_results[key] = {"y_true": y_true.values, "preds": preds, "desc": desc,
                         "mae": mae, "rmse": rmse, "r2": r2, "w2": w2, "bias": bias}

# ── FIGURE 1: ROC curves ─────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(8, 5))
fig1.suptitle("ROC Curves — Detection Models 1A / 1B (Plan v4)", fontsize=13, fontweight="bold")
for key in ["1A", "1B"]:
    fpr, tpr, auc, desc = roc_data[key]
    ax.plot(fpr, tpr, lw=2, label=f"Model {key} — {desc} (AUC={auc*100:.1f}%)")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=50%)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(PLOTS_DIR, "roc_curves_v4.png")
plt.savefig(roc_path, dpi=150)
print(f"\nROC plot saved → {roc_path}")

# ── FIGURE 2: Confusion matrices for Models 1A / 1B ─────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
fig2.suptitle("Confusion Matrices — Detection Models (Plan v4)", fontsize=13, fontweight="bold")

for ax, key in zip(axes2, ["1A", "1B"]):
    bundle = load_model(f"model_{key}")
    feats  = safe_feats(bundle["features"], test_df)
    preds  = bundle["model"].predict(test_df[feats])
    cm     = confusion_matrix(test_df[DETECT_LABEL], preds)
    disp   = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Jaundiced"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Model {key}", fontsize=10)

plt.tight_layout()
cm_path = os.path.join(PLOTS_DIR, "confusion_matrices_v4.png")
plt.savefig(cm_path, dpi=150)
print(f"Confusion matrix saved → {cm_path}")

# ── FIGURE 3: Regression scatter + residuals for Models 2A / 2B ─────────
fig3, axes3 = plt.subplots(2, 2, figsize=(13, 10))
fig3.suptitle("TSB Regression Evaluation — Models 2A / 2B (Plan v4)",
              fontsize=13, fontweight="bold")

for col_idx, key in enumerate(["2A", "2B"]):
    rr       = reg_results[key]
    y_true   = rr["y_true"]
    preds    = rr["preds"]
    residuals = preds - y_true

    # Scatter: predicted vs actual
    ax_scatter = axes3[0, col_idx]
    ax_scatter.scatter(y_true, preds, alpha=0.35, s=12, color="steelblue", edgecolors="none")
    lims = [min(y_true.min(), preds.min()) - 1, max(y_true.max(), preds.max()) + 1]
    ax_scatter.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax_scatter.set_xlabel("True TSB (mg/dL)")
    ax_scatter.set_ylabel("Predicted TSB (mg/dL)")
    ax_scatter.set_title(f"Model {key} — {rr['desc']}\n"
                         f"MAE={rr['mae']:.2f}  RMSE={rr['rmse']:.2f}  R²={rr['r2']:.3f}",
                         fontsize=9)
    ax_scatter.legend(fontsize=8)
    ax_scatter.grid(alpha=0.3)

    # Residual distribution
    ax_resid = axes3[1, col_idx]
    ax_resid.hist(residuals, bins=40, color="coral", edgecolor="white", alpha=0.8)
    ax_resid.axvline(0, color="black", lw=1.2, linestyle="--")
    ax_resid.axvline(-2, color="green", lw=1, linestyle=":", label="±2 mg/dL")
    ax_resid.axvline(+2, color="green", lw=1, linestyle=":")
    ax_resid.set_xlabel("Residual (Predicted − True) mg/dL")
    ax_resid.set_ylabel("Count")
    ax_resid.set_title(f"Residuals — Model {key}  (bias={rr['bias']:+.2f}  ±2mgdL: {rr['w2']:.1f}%)",
                        fontsize=9)
    ax_resid.legend(fontsize=8)
    ax_resid.grid(alpha=0.3)

plt.tight_layout()
reg_path = os.path.join(PLOTS_DIR, "regression_eval_v4.png")
plt.savefig(reg_path, dpi=150)
print(f"Regression evaluation plot saved → {reg_path}")

# ── FIGURE 4: Bhutani Nomogram context ──────────────────────────────────
# Shows where the predicted TSB values land relative to the Bhutani risk
# zone thresholds at a typical age (72 hours = 3 days postnatal).

# Approximate Bhutani percentile thresholds at each postnatal hour
# (piecewise linear from published nomogram table, term/near-term infants)
BHUTANI_HOURS = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144]
P95 = [7.5,  10.0, 12.5, 14.5, 16.0, 17.0, 17.5, 17.5, 17.0, 16.5, 16.0, 15.5]
P75 = [5.5,   7.5,  9.5, 11.0, 12.5, 13.5, 14.0, 14.0, 13.5, 13.0, 12.5, 12.0]
P40 = [3.5,   5.5,  7.0,  8.5,  9.5, 10.5, 11.0, 11.0, 10.5, 10.0,  9.5,  9.0]

fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.fill_between(BHUTANI_HOURS, P95, 30, alpha=0.15, color="red",    label="High Risk (>95th)")
ax4.fill_between(BHUTANI_HOURS, P75, P95, alpha=0.15, color="orange", label="High-Intermediate (75–95th)")
ax4.fill_between(BHUTANI_HOURS, P40, P75, alpha=0.15, color="yellow", label="Low-Intermediate (40–75th)")
ax4.fill_between(BHUTANI_HOURS, 0,   P40, alpha=0.15, color="green",  label="Low Risk (<40th)")

ax4.plot(BHUTANI_HOURS, P95, "r-",  lw=1.5)
ax4.plot(BHUTANI_HOURS, P75, "-",  color="orange", lw=1.5)
ax4.plot(BHUTANI_HOURS, P40, "y-", lw=1.5)

# Overlay Model 2A predictions (if postnatal_age_days is available)d
if "postnatal_age_days" in reg_test_df.columns:
    bundle_2A = load_model("model_2A")
    feats_2A  = safe_feats(bundle_2A["features"], reg_test_df)
    pts_pred  = np.clip(bundle_2A["model"].predict(reg_test_df[feats_2A]), 0, 40)
    pts_hours = reg_test_df["postnatal_age_days"].astype(float).values * 24 # type: ignore
    ax4.scatter(pts_hours, pts_pred, s=12, alpha=0.4, color="steelblue",
                edgecolors="none", label="Model 2A Predictions")

ax4.set_xlim(12, 144)
ax4.set_ylim(0, 25)
ax4.set_xlabel("Postnatal Age (hours)", fontsize=11)
ax4.set_ylabel("Estimated TSB (mg/dL)", fontsize=11)
ax4.set_title("Bhutani Nomogram with Model 2A TSB Predictions (Plan v4)", fontsize=12, fontweight="bold")
ax4.legend(fontsize=9, loc="upper right")
ax4.grid(alpha=0.3)
plt.tight_layout()
nom_path = os.path.join(PLOTS_DIR, "bhutani_nomogram_overlay_v4.png")
plt.savefig(nom_path, dpi=150)
print(f"Bhutani nomogram overlay saved → {nom_path}")

plt.show()
print("\nEvaluasi selesai (Plan v4).")