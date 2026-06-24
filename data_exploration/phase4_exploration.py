"""
phase4_feature_engineering.py  Phase 4: Feature Engineering

Engineered features (all tested against blood_mg_dl on original rows):
  1. Cross-zone means        — mean of Lab_b, Cb, H, B, S across zones 1-3
  2. R/B ratio per zone      — R_mean / B_mean per zone
  3. G-B difference per zone — G_mean - B_mean per zone
  4. Cross-zone gradients    — zone3 - zone1 for Lab_b_mean, Cb_mean, H_mean
  5. Log postnatal age       — log1p(postnatal_age_days)
  6. ITA per zone            — arctan((Lab_L_mean - 50) / Lab_b_mean) × 180/π

Each engineered feature is evaluated by:
  - Spearman r vs blood_mg_dl (original rows only)
  - Comparison to the best individual component's Spearman r
  - Keep/drop recommendation based on improvement threshold (+0.02)

Outputs:
  __plots__/explore/
    csv/
      07_engineered_feature_corr.csv
      08_engineered_vs_components.csv
    png/
      07_engineered_feature_corr.png
      08_engineering_gain.png

Usage:
  python phase4_feature_engineering.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_academic_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("phase4")

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_PATH = "__data__/neo/out/training_fix.csv"
OUT_DIR   = "__plots__/explore"
CSV_DIR   = os.path.join(OUT_DIR, "csv")
PNG_DIR   = os.path.join(OUT_DIR, "png")

TARGET = "blood_mg_dl"

ACCENT = "#0984e3"
RED    = "#d63031"
GREEN  = "#00b894"
GRAY   = "#636e72"
ORANGE = "#e17055"

KEEP_THRESHOLD = 0.02  # minimum Spearman gain over best component to recommend keeping
ZONES = ["zone1", "zone2", "zone3"]

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

apply_academic_style()

df_all = pd.read_csv(DATA_PATH)
orig   = df_all[df_all["is_augmented"] == False].copy()

log.info("Loaded %d original rows", len(orig))


def spearman_r(a, b):
    mask = a.notna() & b.notna()
    r, _ = stats.spearmanr(a[mask], b[mask])
    return float(r) # type: ignore


def safe_div(num, den):
    result = num.copy().astype(float)
    result[den == 0] = np.nan
    result[den != 0] = num[den != 0] / den[den != 0]
    return result


eng_rows   = []
comp_rows  = []


def record(name, series, best_component_r, component_names):
    r = spearman_r(series, orig[TARGET])
    gain = abs(r) - abs(best_component_r)
    keep = gain >= KEEP_THRESHOLD or abs(r) >= abs(best_component_r)
    eng_rows.append({
        "feature":           name,
        "spearman_r":        round(r, 4),
        "abs_spearman":      round(abs(r), 4),
        "best_component_r":  round(best_component_r, 4),
        "gain_over_best":    round(gain, 4),
        "recommend_keep":    keep,
        "components":        ", ".join(component_names),
    })
    return r


# 1. Cross-zone means

log.info("Engineering: cross-zone means")

cross_mean_channels = ["Lab_b_mean", "Cb_mean", "H_mean", "B_mean", "S_mean"]

for ch in cross_mean_channels:
    cols = [f"{z}_{ch}" for z in ZONES if f"{z}_{ch}" in orig.columns]
    if len(cols) < 2:
        continue
    series = orig[cols].mean(axis=1)
    component_rs = [spearman_r(orig[c], orig[TARGET]) for c in cols]
    best_r = max(component_rs, key=abs)
    r = record(f"mean_zones_{ch}", series, best_r, cols)
    log.info("  mean_zones_%s  r=%.4f  (best component r=%.4f)", ch, r, best_r)

    for col, cr in zip(cols, component_rs):
        comp_rows.append({
            "engineered_feature": f"mean_zones_{ch}",
            "component":          col,
            "component_r":        round(cr, 4),
        })


# 2. R/B ratio per zone

log.info("Engineering: R/B ratio per zone")

for zone in ZONES:
    r_col = f"{zone}_R_mean"
    b_col = f"{zone}_B_mean"
    if r_col not in orig.columns or b_col not in orig.columns:
        continue
    series = safe_div(orig[r_col], orig[b_col])
    best_r = max(
        spearman_r(orig[r_col], orig[TARGET]),
        spearman_r(orig[b_col], orig[TARGET]),
        key=abs,
    )
    r = record(f"{zone}_R_div_B", series, best_r, [r_col, b_col])
    log.info("  %s_R_div_B  r=%.4f  (best component r=%.4f)", zone, r, best_r)

    for col in [r_col, b_col]:
        comp_rows.append({
            "engineered_feature": f"{zone}_R_div_B",
            "component":          col,
            "component_r":        round(spearman_r(orig[col], orig[TARGET]), 4),
        })


# 3. G-B difference per zone

log.info("Engineering: G-B difference per zone")

for zone in ZONES:
    g_col = f"{zone}_G_mean"
    b_col = f"{zone}_B_mean"
    if g_col not in orig.columns or b_col not in orig.columns:
        continue
    series = orig[g_col] - orig[b_col]
    best_r = max(
        spearman_r(orig[g_col], orig[TARGET]),
        spearman_r(orig[b_col], orig[TARGET]),
        key=abs,
    )
    r = record(f"{zone}_G_minus_B", series, best_r, [g_col, b_col])
    log.info("  %s_G_minus_B  r=%.4f  (best component r=%.4f)", zone, r, best_r)

    for col in [g_col, b_col]:
        comp_rows.append({
            "engineered_feature": f"{zone}_G_minus_B",
            "component":          col,
            "component_r":        round(spearman_r(orig[col], orig[TARGET]), 4),
        })


# 4. Cross-zone gradients (zone3 - zone1)

log.info("Engineering: cross-zone gradients (zone3 - zone1)")

gradient_channels = ["Lab_b_mean", "Cb_mean", "H_mean"]

for ch in gradient_channels:
    z3_col = f"zone3_{ch}"
    z1_col = f"zone1_{ch}"
    if z3_col not in orig.columns or z1_col not in orig.columns:
        continue
    series = orig[z3_col] - orig[z1_col]
    best_r = max(
        spearman_r(orig[z3_col], orig[TARGET]),
        spearman_r(orig[z1_col], orig[TARGET]),
        key=abs,
    )
    r = record(f"grad_z3z1_{ch}", series, best_r, [z3_col, z1_col])
    log.info("  grad_z3z1_%s  r=%.4f  (best component r=%.4f)", ch, r, best_r)

    for col in [z3_col, z1_col]:
        comp_rows.append({
            "engineered_feature": f"grad_z3z1_{ch}",
            "component":          col,
            "component_r":        round(spearman_r(orig[col], orig[TARGET]), 4),
        })


# 5. Log postnatal age

log.info("Engineering: log(postnatal_age_days)")

if "postnatal_age_days" in orig.columns:
    raw_r  = spearman_r(orig["postnatal_age_days"], orig[TARGET])
    log_series = np.log1p(orig["postnatal_age_days"])
    r = record("log1p_postnatal_age_days", log_series, raw_r, ["postnatal_age_days"])
    log.info("  log1p_postnatal_age_days  r=%.4f  (raw r=%.4f)", r, raw_r)
    comp_rows.append({
        "engineered_feature": "log1p_postnatal_age_days",
        "component":          "postnatal_age_days",
        "component_r":        round(raw_r, 4),
    })


# 6. ITA (Individual Typology Angle) per zone

log.info("Engineering: ITA per zone")

for zone in ZONES:
    l_col = f"{zone}_Lab_L_mean"
    b_col = f"{zone}_Lab_b_mean"
    if l_col not in orig.columns or b_col not in orig.columns:
        continue
    denom = orig[b_col].copy()
    ita = np.where(
        denom != 0,
        np.degrees(np.arctan((orig[l_col] - 50) / denom)),
        np.nan,
    )
    series = pd.Series(ita, index=orig.index)
    best_r = max(
        spearman_r(orig[l_col], orig[TARGET]),
        spearman_r(orig[b_col], orig[TARGET]),
        key=abs,
    )
    r = record(f"{zone}_ITA", series, best_r, [l_col, b_col])
    log.info("  %s_ITA  r=%.4f  (best component r=%.4f)", zone, r, best_r)

    for col in [l_col, b_col]:
        comp_rows.append({
            "engineered_feature": f"{zone}_ITA",
            "component":          col,
            "component_r":        round(spearman_r(orig[col], orig[TARGET]), 4),
        })


# Save CSVs

eng_df  = pd.DataFrame(eng_rows).sort_values("abs_spearman", ascending=False)
comp_df = pd.DataFrame(comp_rows)

eng_df.to_csv(os.path.join(CSV_DIR, "07_engineered_feature_corr.csv"), index=False)
comp_df.to_csv(os.path.join(CSV_DIR, "08_engineered_vs_components.csv"), index=False)

kept    = eng_df[eng_df["recommend_keep"]]["feature"].tolist()
dropped = eng_df[~eng_df["recommend_keep"]]["feature"].tolist()
log.info("Recommended keep (%d): %s", len(kept), kept)
log.info("Recommended drop (%d): %s", len(dropped), dropped)


# Plot 1: engineered feature Spearman bar chart

fig, ax = plt.subplots(figsize=(13, 7))
bar_colors = [GREEN if r else RED for r in eng_df["recommend_keep"]]
ax.barh(eng_df["feature"].tolist(), eng_df["spearman_r"].tolist(),
        color=bar_colors, edgecolor="white", height=0.72, alpha=0.88)
ax.axvline(0,     color="#2d3436", linewidth=1.0)
ax.axvline( 0.50, color=GRAY, linewidth=1.2, linestyle="--", alpha=0.6,
            label="|r| = 0.50")
ax.axvline(-0.50, color=GRAY, linewidth=1.2, linestyle="--", alpha=0.6)

keep_patch = mpatches.Patch(color=GREEN, label="Recommend keep")
drop_patch = mpatches.Patch(color=RED,   label="Recommend drop")
ax.legend(handles=[keep_patch, drop_patch], loc="lower right")
ax.set_xlabel("Spearman r with blood_mg_dl")
ax.set_title("Phase 4: Engineered Feature Spearman Correlations (original rows only)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(PNG_DIR, "07_engineered_feature_corr.png"))
plt.close()
log.info("Saved 07_engineered_feature_corr.png")


# Plot 2: gain over best component

fig, ax = plt.subplots(figsize=(13, 7))
gain_colors = [GREEN if g >= KEEP_THRESHOLD else (ORANGE if g >= 0 else RED)
               for g in eng_df["gain_over_best"]]
ax.barh(eng_df["feature"].tolist(), eng_df["gain_over_best"].tolist(),
        color=gain_colors, edgecolor="white", height=0.72, alpha=0.88)
ax.axvline(0,               color="#2d3436", linewidth=1.0)
ax.axvline(KEEP_THRESHOLD,  color=GREEN, linewidth=1.4, linestyle="--",
           label=f"Keep threshold (+{KEEP_THRESHOLD})")

gain_patches = [
    mpatches.Patch(color=GREEN,  label=f"Gain ≥ {KEEP_THRESHOLD} (clear improvement)"),
    mpatches.Patch(color=ORANGE, label=f"0 ≤ gain < {KEEP_THRESHOLD} (marginal)"),
    mpatches.Patch(color=RED,    label="Gain < 0 (worse than component)"),
]
ax.legend(handles=gain_patches, loc="lower right")
ax.set_xlabel("Δ |Spearman r| vs best component")
ax.set_title("Phase 4: Engineering Gain over Best Individual Component")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(PNG_DIR, "08_engineering_gain.png"))
plt.close()
log.info("Saved 08_engineering_gain.png")


log.info("Phase 4 complete. Outputs in %s/", OUT_DIR)
log.info("Final engineered feature count: %d kept, %d dropped", len(kept), len(dropped))