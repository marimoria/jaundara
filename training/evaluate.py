"""
evaluate.py
Loads both saved models (1: detection classifier, 2: TSB regressor)
and produces evaluation plots. Run after train_models.py.

Uses training_engineered.csv and applies the same build_features() engineering
so the test-set feature matrix matches what the models were trained on.
"""

import logging
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_PATH    = "__data__/neo/out/training_engineered.csv"
MODELS_DIR   = "__models__"
PLOTS_DIR    = "__plots__"
DETECT_LABEL = "jaundice_label"
TSB_COL      = "blood_mg_dl"
TSB_CLIP_MAX = 40.0

BHUTANI_HOURS = [12,  24,  36,  48,  60,  72,  84,  96, 108, 120, 132, 144]
P95 = [7.5, 10.0, 12.5, 14.5, 16.0, 17.0, 17.5, 17.5, 17.0, 16.5, 16.0, 15.5]
P75 = [5.5,  7.5,  9.5, 11.0, 12.5, 13.5, 14.0, 14.0, 13.5, 13.0, 12.5, 12.0]
P40 = [3.5,  5.5,  7.0,  8.5,  9.5, 10.5, 11.0, 11.0, 10.5, 10.0,  9.5,  9.0]

os.makedirs(PLOTS_DIR, exist_ok=True)


# ── feature engineering (must mirror train_models.py:build_features) ──────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    zones = ["zone1", "zone2", "zone3"]

    h_cols = [f"{z}_H_mean" for z in zones]
    out["mean_zones_H_mean"] = out[h_cols].mean(axis=1)

    for ch in ["Lab_b_mean", "Cb_mean", "H_mean"]:
        out[f"grad_z3z1_{ch}"] = out[f"zone3_{ch}"] - out[f"zone1_{ch}"]

    return out


# ── load data + reproduce patient-level test split ────────────────────────────

df_raw = pd.read_csv(DATA_PATH)
df = build_features(df_raw)

patients = df[~df["is_augmented"]]["patient_id"].unique()
train_p, temp_p = train_test_split(patients, test_size=0.30, random_state=42)
_, test_p       = train_test_split(temp_p,   test_size=0.50, random_state=42)

# test set: original rows only
test_df = df[df["patient_id"].isin(test_p) & ~df["is_augmented"]].copy()
logging.info("Test set: %d patients, %d rows", len(test_p), len(test_df))


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(name: str) -> dict:
    with open(f"{MODELS_DIR}/{name}.pkl", "rb") as f:
        return pickle.load(f)


def safe_feats(feats: list, df: pd.DataFrame) -> list:
    missing = [f for f in feats if f not in df.columns]
    if missing:
        logging.warning("Features in model bundle missing from test_df: %s", missing)
    return [f for f in feats if f in df.columns]


# ── classification models 1A / 1B ────────────────────────────────────────────

bundle_1 = load_model("model_1")
feats_1  = safe_feats(bundle_1["features"], test_df)
X, y     = test_df[feats_1], test_df[DETECT_LABEL]

preds_1 = bundle_1["model"].predict(X)
proba_1 = bundle_1["model"].predict_proba(X)[:, 1]

acc = accuracy_score(y, preds_1)
auc = roc_auc_score(y, proba_1)
f1  = f1_score(y, preds_1, zero_division=0)
fpr, tpr, _ = roc_curve(y, proba_1)

logging.info("Model 1: Detection  |  features=%d", len(feats_1))
logging.info("  Accuracy : %.2f%%", acc * 100)
logging.info("  AUC      : %.2f%%", auc * 100)
logging.info("  F1       : %.2f%%", f1 * 100)
logging.info(classification_report(y, preds_1,
                                    target_names=["Normal", "Jaundiced"],
                                    zero_division=0))


# ── regression models 2A / 2B ─────────────────────────────────────────────────

bundle_2 = load_model("model_2")
feats_2  = safe_feats(bundle_2["features"], test_df)
y_true   = test_df[TSB_COL]

preds_2 = np.clip(bundle_2["model"].predict(test_df[feats_2]), 0.0, TSB_CLIP_MAX)

mae  = mean_absolute_error(y_true, preds_2)
rmse = float(np.sqrt(mean_squared_error(y_true, preds_2)))
r2   = r2_score(y_true, preds_2)
w2   = float(np.mean(np.abs(y_true - preds_2) <= 2.0) * 100)
w3   = float(np.mean(np.abs(y_true - preds_2) <= 3.0) * 100)
bias = float(np.mean(preds_2 - y_true))

logging.info("Model 2: TSB Regressor  |  features=%d", len(feats_2))
logging.info("  MAE       : %.3f mg/dL", mae)
logging.info("  RMSE      : %.3f mg/dL", rmse)
logging.info("  R²        : %.4f", r2)
logging.info("  Within ±2 : %.1f%%", w2)
logging.info("  Within ±3 : %.1f%%", w3)
logging.info("  Bias      : %+.3f mg/dL", bias)

reg_results = {
    "y_true": y_true.values, "preds": preds_2,
    "mae": mae, "rmse": rmse, "r2": r2, "w2": w2, "bias": bias,
}


# ── plot 1: ROC curves ────────────────────────────────────────────────────────

fig1, ax = plt.subplots(figsize=(8, 5))
fig1.suptitle("ROC Curve — Detection Model 1", fontsize=13, fontweight="bold")
ax.plot(fpr, tpr, lw=2, label=f"Model 1 — Detection (AUC={auc*100:.1f}%)")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=50%)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), dpi=150)
logging.info("roc_curves.png saved")


# ── plot 2: confusion matrices ────────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(5, 4))
fig2.suptitle("Confusion Matrix — Detection Model 1", fontsize=13, fontweight="bold")
cm   = confusion_matrix(test_df[DETECT_LABEL], preds_1)
disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Jaundiced"])
disp.plot(ax=ax2, colorbar=False)
ax2.set_title("Model 1", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"), dpi=150)
logging.info("confusion_matrices.png saved")


# ── plot 3: regression scatter + residuals ────────────────────────────────────

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle("TSB Regression Evaluation — Model 2", fontsize=13, fontweight="bold")

y_true    = reg_results["y_true"]
preds     = reg_results["preds"]
residuals = preds - y_true

ax_scatter = axes3[0]
ax_scatter.scatter(y_true, preds, alpha=0.35, s=12, color="steelblue", edgecolors="none")
lims = [min(y_true.min(), preds.min()) - 1, max(y_true.max(), preds.max()) + 1]
ax_scatter.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
ax_scatter.set_xlabel("True TSB (mg/dL)")
ax_scatter.set_ylabel("Predicted TSB (mg/dL)")
ax_scatter.set_title(
    f"Model 2 — TSB Regressor\nMAE={reg_results['mae']:.2f}  RMSE={reg_results['rmse']:.2f}  R²={reg_results['r2']:.3f}",
    fontsize=9,
)
ax_scatter.legend(fontsize=8)
ax_scatter.grid(alpha=0.3)

ax_resid = axes3[1]
ax_resid.hist(residuals, bins=40, color="coral", edgecolor="white", alpha=0.8)
ax_resid.axvline(0,  color="black", lw=1.2, linestyle="--")
ax_resid.axvline(-2, color="green", lw=1,   linestyle=":", label="±2 mg/dL")
ax_resid.axvline(+2, color="green", lw=1,   linestyle=":")
ax_resid.set_xlabel("Residual (Predicted − True) mg/dL")
ax_resid.set_ylabel("Count")
ax_resid.set_title(
    f"Residuals — Model 2  (bias={reg_results['bias']:+.2f}  ±2 mg/dL: {reg_results['w2']:.1f}%)",
    fontsize=9,
)
ax_resid.legend(fontsize=8)
ax_resid.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "regression_eval.png"), dpi=150)
logging.info("regression_eval.png saved")


# ── plot 4: Bhutani nomogram overlay ─────────────────────────────────────────

fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.fill_between(BHUTANI_HOURS, P95, 30,  alpha=0.15, color="red",
                 label="High Risk (>95th)")
ax4.fill_between(BHUTANI_HOURS, P75, P95, alpha=0.15, color="orange",
                 label="High-Intermediate (75–95th)")
ax4.fill_between(BHUTANI_HOURS, P40, P75, alpha=0.15, color="yellow",
                 label="Low-Intermediate (40–75th)")
ax4.fill_between(BHUTANI_HOURS, 0,   P40, alpha=0.15, color="green",
                 label="Low Risk (<40th)")
ax4.plot(BHUTANI_HOURS, P95, "r-",              lw=1.5)
ax4.plot(BHUTANI_HOURS, P75, "-", color="orange", lw=1.5)
ax4.plot(BHUTANI_HOURS, P40, "y-",              lw=1.5)

pts_pred  = np.clip(bundle_2["model"].predict(test_df[feats_2]), 0, TSB_CLIP_MAX)
pts_hours = test_df["postnatal_age_days"].astype(float).values * 24  # type: ignore
ax4.scatter(pts_hours, pts_pred, s=12, alpha=0.4,
            color="steelblue", edgecolors="none", label="Model 2 Predictions")

ax4.set_xlim(12, 144)
ax4.set_ylim(0, 25)
ax4.set_xlabel("Postnatal Age (hours)", fontsize=11)
ax4.set_ylabel("Estimated TSB (mg/dL)", fontsize=11)
ax4.set_title("Bhutani Nomogram with Model 2 TSB Predictions",
              fontsize=12, fontweight="bold")
ax4.legend(fontsize=9, loc="upper right")
ax4.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "bhutani_nomogram_overlay.png"), dpi=150)
logging.info("bhutani_nomogram_overlay.png saved")

plt.show()