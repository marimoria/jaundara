"""
predict.py — Neonatal Jaundice Inference

Pipeline:
  1. Model 1  : binary detection gate (jaundiced vs normal)
  2. Model 2  : TSB regression : estimated blood_mg_dl
  3. Bhutani logic: risk zone + action from (tsb_mgdl, age_hours)

Engineered features are computed inline from raw zone values before inference,
matching the feature set produced by train_models.py:build_features().
"""

import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")

MODELS_DIR = "__models__"

# Piecewise-linear approximation of the Bhutani nomogram
# (Pediatrics 2000; 114:297-316). Valid range: 12–144 postnatal hours.
_BHUTANI_TABLE: list[tuple[int, float, float, float]] = [
    (12,  3.5,  5.5,  7.5),
    (24,  5.5,  7.5, 10.0),
    (36,  7.0,  9.5, 12.5),
    (48,  8.5, 11.0, 14.5),
    (60,  9.5, 12.5, 16.0),
    (72, 10.5, 13.5, 17.0),
    (84, 11.0, 14.0, 17.5),
    (96, 11.0, 14.0, 17.5),
    (108, 10.5, 13.5, 17.0),
    (120, 10.0, 13.0, 16.5),
    (132,  9.5, 12.5, 16.0),
    (144,  9.0, 12.0, 15.5),
]

_BHUTANI_HOURS = np.array([r[0] for r in _BHUTANI_TABLE], dtype=float)
_BHUTANI_P40   = np.array([r[1] for r in _BHUTANI_TABLE], dtype=float)
_BHUTANI_P75   = np.array([r[2] for r in _BHUTANI_TABLE], dtype=float)
_BHUTANI_P95   = np.array([r[3] for r in _BHUTANI_TABLE], dtype=float)


def _bhutani_thresholds(age_hours: float) -> tuple[float, float, float]:
    """Interpolated (P40, P75, P95) for a given postnatal age; clamped to [12, 144] h."""
    age_clamped = float(np.clip(age_hours, 12.0, 144.0))
    return (
        float(np.interp(age_clamped, _BHUTANI_HOURS, _BHUTANI_P40)),
        float(np.interp(age_clamped, _BHUTANI_HOURS, _BHUTANI_P75)),
        float(np.interp(age_clamped, _BHUTANI_HOURS, _BHUTANI_P95)),
    )


def bhutani_risk_zone(tsb_mgdl: float, postnatal_age_hours: float) -> dict:
    """
    Classify TSB into a Bhutani risk zone.

    Returns dict with keys: zone, zone_code (0–3), action, thresholds (p40/p75/p95).
    """
    p40, p75, p95 = _bhutani_thresholds(postnatal_age_hours)

    if tsb_mgdl >= p95:
        zone, code = "High Risk", 3
        action = (
            "HIGH RISK — Bilirubin is above the 95th percentile. "
            "Seek immediate medical evaluation today. "
            "Phototherapy may be required."
        )
    elif tsb_mgdl >= p75:
        zone, code = "High-Intermediate Risk", 2
        action = (
            "HIGH-INTERMEDIATE RISK — Bilirubin is above the 75th percentile. "
            "Schedule a checkup with your pediatrician today or tomorrow. "
            "Recheck bilirubin within 24 hours."
        )
    elif tsb_mgdl >= p40:
        zone, code = "Low-Intermediate Risk", 1
        action = (
            "LOW-INTERMEDIATE RISK — Bilirubin is slightly elevated. "
            "Ensure adequate feeding and recheck in 24–48 hours. "
            "Contact your doctor if the baby seems more yellow."
        )
    else:
        zone, code = "Low Risk", 0
        action = (
            "LOW RISK — Bilirubin is within the low-risk zone. "
            "Continue normal care and monitoring. "
            "Recheck if jaundice appears to worsen."
        )

    return {
        "zone": zone, "zone_code": code, "action": action,
        "thresholds": {"p40": round(p40, 2), "p75": round(p75, 2), "p95": round(p95, 2)},
    }


def _load_model(name: str) -> dict:
    with open(f"{MODELS_DIR}/{name}.pkl", "rb") as f:
        return pickle.load(f)


def _build_row(zone_features: dict,
               postnatal_age_days: float,
               gestational_age: Optional[float],
               weight: Optional[float]) -> pd.DataFrame:
    """
    Construct a single-row DataFrame with raw zone features plus all
    engineered features used during training.

    Engineered features added here (must mirror train_models.py:build_features):
      mean_zones_H_mean
      grad_z3z1_{Lab_b_mean, Cb_mean, H_mean}

    Already expected in zone_features (caller responsibility, same as CSV):
      mean_zones_Lab_b_mean, mean_zones_Cb_mean, mean_zones_B_mean,
      mean_zones_S_mean, zone{1-3}_R_div_B, zone{1-3}_G_minus_B
    """
    row: dict = dict(zone_features)

    if gestational_age is not None:
        row["gestational_age"] = gestational_age
    row["postnatal_age_days"] = postnatal_age_days
    if weight is not None:
        row["weight"] = weight

    # Engineered features not expected to be pre-computed by caller
    zones = ["zone1", "zone2", "zone3"]

    h_cols = [f"{z}_H_mean" for z in zones]
    if all(c in row for c in h_cols):
        row["mean_zones_H_mean"] = float(np.mean([row[c] for c in h_cols]))

    for ch in ["Lab_b_mean", "Cb_mean", "H_mean"]:
        z3_col = f"zone3_{ch}"
        z1_col = f"zone1_{ch}"
        if z3_col in row and z1_col in row:
            row[f"grad_z3z1_{ch}"] = row[z3_col] - row[z1_col]

    return pd.DataFrame([row])


def predict_patient(
    zone_features: dict,
    postnatal_age_hours: float,
    gestational_age: Optional[float] = None,
    postnatal_age_days: Optional[float] = None,
    weight: Optional[float] = None,
    detection_threshold: float = 0.5,
) -> dict:
    """
    Full inference pipeline.

    Parameters
    ----------
    zone_features         : raw color zone values (zone1_*, zone2_*, zone3_*)
                            plus any pre-computed engineered features already
                            present in your data (mean_zones_*, *_R_div_B, *_G_minus_B).
                            grad_z3z1_* and mean_zones_H_mean are computed internally.
    postnatal_age_hours   : postnatal age in hours — required for Bhutani lookup
    gestational_age       : weeks (optional)
    postnatal_age_days    : derived from postnatal_age_hours if omitted
    weight                : grams (optional; low Spearman r but retained for compatibility)
    detection_threshold   : P(jaundice) threshold for binary gate (default 0.5)

    Returns
    -------
    dict: jaundice_detected, detection_proba, tsb_estimated, bhutani, postnatal_age_hours
    """
    if postnatal_age_days is None:
        postnatal_age_days = postnatal_age_hours / 24.0

    df_row = _build_row(zone_features, postnatal_age_days, gestational_age, weight)

    det_bundle = _load_model("model_1")
    det_feats  = [f for f in det_bundle["features"] if f in df_row.columns]
    det_proba  = float(det_bundle["model"].predict_proba(df_row[det_feats])[0, 1])
    jaundice_detected = det_proba >= detection_threshold

    reg_bundle = _load_model("model_2")
    reg_feats  = [f for f in reg_bundle["features"] if f in df_row.columns]
    raw_tsb    = float(reg_bundle["model"].predict(df_row[reg_feats])[0])
    tsb_estimated = round(float(np.clip(raw_tsb, 0.0, 40.0)), 2)

    return {
        "jaundice_detected":  jaundice_detected,
        "detection_proba":    round(det_proba, 4),
        "tsb_estimated":      tsb_estimated,
        "bhutani":            bhutani_risk_zone(tsb_estimated, postnatal_age_hours),
        "postnatal_age_hours": postnatal_age_hours,
    }


# ── example ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Raw zone values — caller provides all zone{1-3}_* color columns.
    # mean_zones_* and *_R_div_B and *_G_minus_B can optionally be pre-computed
    # by the caller; if absent they are not re-derived here (only grad_z3z1_*
    # and mean_zones_H_mean are computed internally).
    sample_zones = {
        "zone1_R_mean": 185.0, "zone1_G_mean": 145.0, "zone1_B_mean": 95.0,
        "zone1_R_std":  18.0,  "zone1_Y_mean": 148.0, "zone1_Cr_mean": 162.0,
        "zone1_Cb_mean": 99.0, "zone1_H_mean": 26.0,  "zone1_S_mean": 45.0,
        "zone1_L_mean": 53.0,  "zone1_Lab_L_mean": 61.0, "zone1_Lab_a_mean": 15.0,
        "zone1_Lab_b_mean": 30.0, "zone1_Lab_L_std": 5.5,

        "zone2_R_mean": 190.0, "zone2_G_mean": 148.0, "zone2_B_mean": 90.0,
        "zone2_R_std":  16.0,  "zone2_Y_mean": 151.0, "zone2_Cr_mean": 165.0,
        "zone2_Cb_mean": 97.0, "zone2_H_mean": 25.5,  "zone2_S_mean": 46.0,
        "zone2_L_mean": 54.0,  "zone2_Lab_L_mean": 62.0, "zone2_Lab_a_mean": 16.0,
        "zone2_Lab_b_mean": 32.0, "zone2_Lab_L_std": 5.1,

        "zone3_R_mean": 188.0, "zone3_G_mean": 143.0, "zone3_B_mean": 88.0,
        "zone3_R_std":  17.0,  "zone3_Y_mean": 150.0, "zone3_Cr_mean": 163.0,
        "zone3_Cb_mean": 98.0, "zone3_H_mean": 26.1,  "zone3_S_mean": 44.5,
        "zone3_L_mean": 52.0,  "zone3_Lab_L_mean": 60.5, "zone3_Lab_a_mean": 15.5,
        "zone3_Lab_b_mean": 31.0, "zone3_Lab_L_std": 5.3,

        # pre-compute engineered features that the CSV already contains
        "mean_zones_Lab_b_mean": (30.0 + 32.0 + 31.0) / 3,
        "mean_zones_Cb_mean":    (99.0 + 97.0 + 98.0) / 3,
        "mean_zones_B_mean":     (95.0 + 90.0 + 88.0) / 3,
        "mean_zones_S_mean":     (45.0 + 46.0 + 44.5) / 3,
        "zone1_R_div_B": 185.0 / 95.0,
        "zone2_R_div_B": 190.0 / 90.0,
        "zone3_R_div_B": 188.0 / 88.0,
        "zone1_G_minus_B": 145.0 - 95.0,
        "zone2_G_minus_B": 148.0 - 90.0,
        "zone3_G_minus_B": 143.0 - 88.0,
        # grad_z3z1_* and mean_zones_H_mean computed internally by _build_row
    }

    result = predict_patient(
        zone_features=sample_zones,
        postnatal_age_hours=72,
        gestational_age=37,
        weight=2800,
    )

    b = result["bhutani"]
    logging.info("jaundice_detected   : %s", result["jaundice_detected"])
    logging.info("detection_proba     : %.4f", result["detection_proba"])

    logging.info("tsb_estimated       : %.2f mg/dL", result["tsb_estimated"])

    logging.info("postnatal_age_hours : %s h", result["postnatal_age_hours"])
    logging.info("risk_zone           : %s (code=%d)", b["zone"], b["zone_code"])
    logging.info("thresholds @ %sh     : P40=%.2f  P75=%.2f  P95=%.2f mg/dL",
                 result["postnatal_age_hours"],
                 b["thresholds"]["p40"], b["thresholds"]["p75"], b["thresholds"]["p95"])
    logging.info("action              : %s", b["action"])

    logging.info("--- bhutani lookup examples ---")
    for age_h, tsb in [(48, 8.0), (72, 14.0), (72, 17.5), (96, 11.5)]:
        r = bhutani_risk_zone(tsb, age_h)
        logging.info("  age=%3dh  tsb=%5.1f mg/dL  ->  %s", age_h, tsb, r["zone"])