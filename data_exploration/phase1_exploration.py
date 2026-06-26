"""
phase1_exploration.py Phase 1: Dataset Structure Audit

Phase 1 covers three steps:
  1. Structure audit: shape, nulls, augmented split, label balance, feature groups
  2. Target distribution: blood_mg_dl histogram + Q-Q plot + Shapiro-Wilk
  3. Feature skewness: all 45 input features, flag candidates for log-transform

All analysis runs exclusively on original (non-augmented) rows.

Outputs:
  __plots__/explore/
    csv/
      01_structure_audit.csv
      02_target_distribution.csv
      03_feature_skewness.csv
    png/
      02_target_distribution.png
      03_feature_skewness.png

Usage:
  python phase1_exploration.py
"""

import logging
import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

# Import shared plot style from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_academic_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("phase1")

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_PATH = "__data__/neo/out/training_engineered.csv"
OUT_DIR   = "__plots__/explore"
CSV_DIR   = os.path.join(OUT_DIR, "csv")
PNG_DIR   = os.path.join(OUT_DIR, "png")

TARGET    = "blood_mg_dl"
LABEL_COL = "jaundice_label"

# |r| below this is flagged as near-zero in skewness table
WEAK_CORR_THRESHOLD = 0.10

ACCENT = "#0984e3"
RED    = "#d63031"
GREEN  = "#00b894"
GRAY   = "#636e72"

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

apply_academic_style()

df_all = pd.read_csv(DATA_PATH)

# ── Null removal ──────────────────────────────────────────────────────────────
rows_before = len(df_all)
null_counts_before = df_all.isnull().sum()
null_cols_before   = null_counts_before[null_counts_before > 0]

if len(null_cols_before):
    log.info("Null removal: dropping rows with nulls in %d column(s): %s",
             len(null_cols_before), null_cols_before.to_dict())
    df_all = df_all.dropna().reset_index(drop=True)
    rows_dropped = rows_before - len(df_all)
    log.info("  Dropped %d row(s)  (%d → %d)", rows_dropped, rows_before, len(df_all))
    df_all.to_csv(DATA_PATH, index=False)
    log.info("  Saved cleaned dataset → %s", DATA_PATH)
else:
    log.info("Null removal: no nulls found, dataset unchanged.")
# ─────────────────────────────────────────────────────────────────────────────

orig = df_all[df_all["is_augmented"] == False].copy()

ZONE_FEATURES = [c for c in df_all.columns if c.startswith("zone")]
META_FEATURES = ["gestational_age", "postnatal_age_days", "weight"]
ALL_FEATURES  = ZONE_FEATURES + META_FEATURES

log.info("Total rows: %d  |  Original: %d  |  Augmented: %d",
         len(df_all), len(orig), len(df_all) - len(orig))


# Structure audit

log.info("Structure audit")

null_counts = orig.isnull().sum()
null_cols   = null_counts[null_counts > 0]

label_dist = orig[LABEL_COL].value_counts().reset_index()
label_dist.columns = [LABEL_COL, "count"]
label_dist["pct"] = (label_dist["count"] / len(orig) * 100).round(2)

audit_rows = [
    {"metric": "total_rows_all",     "value": str(len(df_all))},
    {"metric": "original_rows",      "value": str(len(orig))},
    {"metric": "augmented_rows",     "value": str(len(df_all) - len(orig))},
    {"metric": "n_patients",         "value": str(orig["patient_id"].nunique())},
    {"metric": "n_columns",          "value": str(len(df_all.columns))},
    {"metric": "n_zone_features",    "value": str(len(ZONE_FEATURES))},
    {"metric": "n_meta_features",    "value": str(len(META_FEATURES))},
    {"metric": "augmentation_ratio", "value": f"{(len(df_all) - len(orig)) / len(orig):.1f}x per patient"},
]

for col, cnt in null_cols.items():
    audit_rows.append({"metric": f"nulls_{col}", "value": str(cnt)})

for _, row in label_dist.iterrows():
    lbl = "normal" if row[LABEL_COL] == 0 else "jaundiced"
    audit_rows.append({"metric": f"label_{lbl}_n",   "value": str(row["count"])})
    audit_rows.append({"metric": f"label_{lbl}_pct", "value": f"{row['pct']}%"})

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(os.path.join(CSV_DIR, "01_structure_audit.csv"), index=False)
log.info("  Saved 01_structure_audit.csv  (%d entries)", len(audit_df))
log.info("  Null columns: %s", null_cols.to_dict() if len(null_cols) else "none")


# Target variable distribution

log.info("Target distribution (blood_mg_dl)")

tsb = orig[TARGET].dropna()

stat_sw, p_sw = stats.shapiro(tsb.sample(min(200, len(tsb)), random_state=42))

q1 = float(np.percentile(tsb, 25))
q3 = float(np.percentile(tsb, 75))

target_stats = {
    "n":               len(tsb),
    "mean":            round(float(tsb.mean()), 4),
    "median":          round(float(tsb.median()), 4),
    "std":             round(float(tsb.std()), 4),
    "min":             round(float(tsb.min()), 4),
    "max":             round(float(tsb.max()), 4),
    "q1":              round(q1, 4),
    "q3":              round(q3, 4),
    "iqr":             round(q3 - q1, 4),
    "skewness":        round(float(tsb.skew()), 4), # type: ignore
    "kurtosis":        round(float(tsb.kurt()), 4), # type: ignore
    "shapiro_stat":    round(float(stat_sw), 6),
    "shapiro_p":       round(float(p_sw), 6),
    "normal":          "No" if p_sw < 0.05 else "Yes",
    "recommended_corr": "Spearman" if p_sw < 0.05 else "Pearson",
}

pd.DataFrame([target_stats]).T.reset_index().rename(
    columns={"index": "statistic", 0: "value"}
).to_csv(os.path.join(CSV_DIR, "02_target_distribution.csv"), index=False)

log.info("  Shapiro-Wilk: W=%.4f  p=%.6f  → %s  → use %s",
         stat_sw, p_sw,
         "Non-normal" if p_sw < 0.05 else "Normal",
         target_stats["recommended_corr"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(tsb, bins=25, density=True, color=ACCENT, edgecolor="white",
        alpha=0.80, label="Histogram (density)")
kde   = gaussian_kde(tsb)
x_kde = np.linspace(tsb.min(), tsb.max(), 300)
ax.plot(x_kde, kde(x_kde), color="#2d3436", linewidth=2.2, label="KDE")
ax.axvline(float(tsb.median()), color=RED,   linestyle="-.", linewidth=2.0,
           label=f"Median: {tsb.median():.2f} mg/dL")
ax.axvline(float(tsb.mean()),   color=GREEN, linestyle="--", linewidth=1.8,
           label=f"Mean: {tsb.mean():.2f} mg/dL")
ax.axvspan(12, float(tsb.max()) + 0.5, alpha=0.08, color=RED,
           label="Phototherapy zone (>12 mg/dL)")
ax.set_xlabel("Total Serum Bilirubin (mg/dL)")
ax.set_ylabel("Density")
ax.set_title("Distribution of blood_mg_dl")
ax.legend(loc="upper right")

ax = axes[1]
(osm, osr), (slope, intercept, _) = stats.probplot(tsb, dist="norm")
ax.scatter(osm, osr, s=14, color=ACCENT, alpha=0.6, edgecolors="none",
           label="Sample quantiles")
ref_line = slope * np.array([osm[0], osm[-1]]) + intercept # type: ignore
ax.plot([osm[0], osm[-1]], ref_line, color=RED, linewidth=1.8, linestyle="--",
        label="Normal reference line")
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles")
ax.set_title(
    f"Q-Q Plot: blood_mg_dl\n"
    f"Shapiro-Wilk: W={stat_sw:.4f}, p={p_sw:.4f}  "
    f"({'Non-normal' if p_sw < 0.05 else 'Normal'})"
)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(PNG_DIR, "02_target_distribution.png"))
plt.close()
log.info("  Saved 02_target_distribution.png")


# Feature skewness across all 45 input features

log.info("Feature skewness")

skew_rows = []
for col in ALL_FEATURES:
    sk = float(orig[col].dropna().skew()) # type: ignore
    skew_rows.append({
        "feature":  col,
        "skewness": round(sk, 4),
        "abs_skew": round(abs(sk), 4),
        "flag":     "log-transform candidate" if abs(sk) > 1.0 else "",
    })

skew_df = pd.DataFrame(skew_rows).sort_values("abs_skew", ascending=False)
skew_df.to_csv(os.path.join(CSV_DIR, "03_feature_skewness.csv"), index=False)

flagged = skew_df[skew_df["abs_skew"] > 1.0]["feature"].tolist()
log.info("  Features with |skew| > 1.0: %s", flagged)

fig, ax = plt.subplots(figsize=(14, 6))
bar_colors = [RED if abs(r["skewness"]) > 1.0 else ACCENT for _, r in skew_df.iterrows()]
ax.barh(
    skew_df["feature"].tolist(),
    skew_df["skewness"].tolist(),
    color=bar_colors, edgecolor="white", height=0.72, alpha=0.88,
)
ax.axvline(0,    color="#2d3436", linewidth=1.0)
ax.axvline( 1.0, color=RED, linewidth=1.2, linestyle="--", label="|skew| = 1.0 threshold")
ax.axvline(-1.0, color=RED, linewidth=1.2, linestyle="--")

legend_patches = [
    mpatches.Patch(color=ACCENT, label="|skew| ≤ 1.0 (approximately symmetric)"),
    mpatches.Patch(color=RED,    label="|skew| > 1.0 (log-transform candidate)"),
]
ax.legend(handles=legend_patches + [ax.lines[1]], loc="lower right")
ax.set_xlabel("Skewness")
ax.set_title("Feature Skewness: All 45 Input Features (original rows only)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(PNG_DIR, "03_feature_skewness.png"))
plt.close()
log.info("  Saved 03_feature_skewness.png")


log.info("Phase 1 complete. Outputs in %s/", OUT_DIR)
log.info("  Shapiro-Wilk p=%.6f → use %s for Phase 2 onward",
         p_sw, target_stats["recommended_corr"])
log.info("  Log-transform candidates: %s", flagged)
log.info("  Null columns to resolve before Phase 2: %s",
         list(null_cols.index) if len(null_cols) else "none")