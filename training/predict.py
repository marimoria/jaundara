# predict_v4.py — Neonatal Jaundice Inference (Plan v4)
#
# Pipeline:
#   Step 1 — Model 1A/1B  : Binary detection gate (jaundiced vs normal)
#   Step 2 — Model 2A/2B  : TSB regression → estimated blood_mg_dl
#   Step 3 — Bhutani logic : risk zone + action from (tsb_mgdl, age_hours)
#
# The risk zone classification is done HERE in Python (and mirrors the
# Flutter app logic), using the Bhutani Nomogram thresholds.

import pickle
import numpy as np
import pandas as pd
from typing import Optional

MODELS_DIR = "__models__"


# ─────────────────────────────────────────────────────────────
# Bhutani Nomogram
# ─────────────────────────────────────────────────────────────
# Piecewise-linear approximation of the published Bhutani nomogram
# (Pediatrics 2000; 114:297-316) for term and near-term neonates.
# Hours → percentile thresholds in mg/dL.
#
# Keys: postnatal age in hours (applicable range: 12–144 h)
# Values: (P40_threshold, P75_threshold, P95_threshold)
#
# Risk zones (from lowest to highest risk):
#   Low Risk          : TSB < P40
#   Low-Intermediate  : P40 ≤ TSB < P75
#   High-Intermediate : P75 ≤ TSB < P95
#   High Risk         : TSB ≥ P95

_BHUTANI_TABLE: list[tuple[int, float, float, float]] = [
    # (hours, P40,  P75,  P95)
    (12,  3.5,  5.5,  7.5),
    (24,  5.5,  7.5, 10.0),
    (36,  7.0,  9.5, 12.5),
    (48,  8.5, 11.0, 14.5),
    (60,  9.5, 12.5, 16.0),
    (72, 10.5, 13.5, 17.0),
    (84, 11.0, 14.0, 17.5),
    (96, 11.0, 14.0, 17.5),
    (108,10.5, 13.5, 17.0),
    (120,10.0, 13.0, 16.5),
    (132, 9.5, 12.5, 16.0),
    (144, 9.0, 12.0, 15.5),
]

_BHUTANI_HOURS = np.array([r[0] for r in _BHUTANI_TABLE], dtype=float)
_BHUTANI_P40   = np.array([r[1] for r in _BHUTANI_TABLE], dtype=float)
_BHUTANI_P75   = np.array([r[2] for r in _BHUTANI_TABLE], dtype=float)
_BHUTANI_P95   = np.array([r[3] for r in _BHUTANI_TABLE], dtype=float)


def _bhutani_thresholds(age_hours: float) -> tuple[float, float, float]:
    """
    Returns (P40, P75, P95) thresholds for a given postnatal age in hours.
    Clamps to [12, 144] hours — the valid range of the nomogram.
    Uses piecewise-linear interpolation between table points.
    """
    age_clamped = float(np.clip(age_hours, 12.0, 144.0))
    p40 = float(np.interp(age_clamped, _BHUTANI_HOURS, _BHUTANI_P40))
    p75 = float(np.interp(age_clamped, _BHUTANI_HOURS, _BHUTANI_P75))
    p95 = float(np.interp(age_clamped, _BHUTANI_HOURS, _BHUTANI_P95))
    return p40, p75, p95


def bhutani_risk_zone(tsb_mgdl: float, postnatal_age_hours: float) -> dict:
    """
    Classify TSB into a Bhutani risk zone given the baby's age.

    Parameters
    ----------
    tsb_mgdl            : estimated or measured total serum bilirubin (mg/dL)
    postnatal_age_hours : baby's age in hours since birth

    Returns
    -------
    dict with keys:
        zone        — "Low Risk" | "Low-Intermediate" | "High-Intermediate" | "High Risk"
        zone_code   — 0 | 1 | 2 | 3  (increasing severity)
        action      — recommended clinical action string
        thresholds  — dict with p40/p75/p95 at this age
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
        "zone":       zone,
        "zone_code":  code,
        "action":     action,
        "thresholds": {"p40": round(p40, 2), "p75": round(p75, 2), "p95": round(p95, 2)},
    }


# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────

def _load_model(name: str) -> dict:
    with open(f"{MODELS_DIR}/{name}.pkl", "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# Main inference function
# ─────────────────────────────────────────────────────────────

def predict_patient(
    zone_features: dict,
    postnatal_age_hours: float,          # REQUIRED for Bhutani nomogram
    gestational_age: Optional[float] = None,
    postnatal_age_days: Optional[float] = None,  # derived from postnatal_age_hours if None
    weight: Optional[float] = None,
    detection_threshold: float = 0.5,
) -> dict:
    """
    Full v4 inference pipeline.

    Parameters
    ----------
    zone_features         : dict of color zone feature values (zone1_*, zone2_*, zone3_*)
    postnatal_age_hours   : baby's age in hours since birth — required for Bhutani lookup
    gestational_age       : weeks (optional — enables Model A variants)
    postnatal_age_days    : days since birth (derived from postnatal_age_hours if omitted)
    weight                : grams (optional)
    detection_threshold   : probability threshold for jaundice detection (default 0.5)

    Returns
    -------
    dict:
      jaundice_detected   : bool
      detection_proba     : float  — P(jaundice) from Model 1
      tsb_estimated       : float   — predicted blood_mg_dl
      bhutani             : dict    — zone, zone_code, action, thresholds
      model_detection     : "1A" | "1B"
      model_regression    : "2A" | "2B"
      postnatal_age_hours : float  (echoed for Flutter)
    """
    # Derive postnatal_age_days if not supplied
    if postnatal_age_days is None:
        postnatal_age_days = postnatal_age_hours / 24.0

    has_meta = all(v is not None for v in [gestational_age, postnatal_age_days, weight])

    # ── Step 1: Detection gate ────────────────────────────────
    det_key = "1A" if has_meta else "1B"
    det_bundle = _load_model(f"model_{det_key}")

    row: dict = {**zone_features}
    if has_meta:
        row["gestational_age"]    = gestational_age
        row["postnatal_age_days"] = postnatal_age_days
        row["weight"]             = weight

    df_row = pd.DataFrame([row])
    det_feats = [f for f in det_bundle["features"] if f in df_row.columns]
    det_proba = float(det_bundle["model"].predict_proba(df_row[det_feats])[0, 1])
    jaundice_detected = det_proba >= detection_threshold

    # ── Step 2: TSB regression (always — normal babies have bilirubin too) ─
    reg_key    = "2A" if has_meta else "2B"
    reg_bundle = _load_model(f"model_{reg_key}")
    reg_feats  = [f for f in reg_bundle["features"] if f in df_row.columns]
    raw_tsb    = float(reg_bundle["model"].predict(df_row[reg_feats])[0])
    tsb_estimated = round(float(np.clip(raw_tsb, 0.0, 40.0)), 2)

    # ── Step 3: Bhutani nomogram (always) ────────────────────
    bhutani_result = bhutani_risk_zone(tsb_estimated, postnatal_age_hours)

    return {
        "jaundice_detected":   jaundice_detected,
        "detection_proba":     round(det_proba, 4),
        "tsb_estimated":       tsb_estimated,
        "bhutani":             bhutani_result,
        "model_detection":     det_key,
        "model_regression":    reg_key,
        "postnatal_age_hours": postnatal_age_hours,
    }


# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_zones = {
        "zone1_R_mean": 185.0, "zone1_G_mean": 145.0, "zone1_B_mean": 95.0,
        "zone1_R_std": 18.0,   "zone1_Y_mean": 148.0, "zone1_Cr_mean": 162.0,
        "zone1_Cb_mean": 99.0, "zone1_H_mean": 26.0,  "zone1_S_mean": 45.0,
        "zone1_L_mean": 53.0,  "zone1_Lab_L_mean": 61.0, "zone1_Lab_a_mean": 15.0,
        "zone1_Lab_b_mean": 30.0, "zone1_Lab_L_std": 5.5,

        "zone2_R_mean": 190.0, "zone2_G_mean": 148.0, "zone2_B_mean": 90.0,
        "zone2_R_std": 16.0,   "zone2_Y_mean": 151.0, "zone2_Cr_mean": 165.0,
        "zone2_Cb_mean": 97.0, "zone2_H_mean": 25.5,  "zone2_S_mean": 46.0,
        "zone2_L_mean": 54.0,  "zone2_Lab_L_mean": 62.0, "zone2_Lab_a_mean": 16.0,
        "zone2_Lab_b_mean": 32.0, "zone2_Lab_L_std": 5.1,

        "zone3_R_mean": 188.0, "zone3_G_mean": 143.0, "zone3_B_mean": 88.0,
        "zone3_R_std": 17.0,   "zone3_Y_mean": 150.0, "zone3_Cr_mean": 163.0,
        "zone3_Cb_mean": 98.0, "zone3_H_mean": 26.1,  "zone3_S_mean": 44.5,
        "zone3_L_mean": 52.0,  "zone3_Lab_L_mean": 60.5, "zone3_Lab_a_mean": 15.5,
        "zone3_Lab_b_mean": 31.0, "zone3_Lab_L_std": 5.3,
    }

    # Example: 3 days old = 72 hours
    result = predict_patient(
        zone_features=sample_zones,
        postnatal_age_hours=72,       # 3 days old
        gestational_age=37,
        weight=2800,
        # postnatal_age_days derived automatically from postnatal_age_hours
    )

    print("\n═══ PREDICTION RESULT (Plan v4) ═══")
    print(f"  Jaundice Detected   : {result['jaundice_detected']}")
    print(f"  Detection P(jaund.) : {result['detection_proba']:.4f}")
    print(f"  Model (Detection)   : {result['model_detection']}")
    print(f"  Estimated TSB       : {result['tsb_estimated']} mg/dL")
    print(f"  Model (Regression)  : {result['model_regression']}")
    print(f"  Postnatal Age       : {result['postnatal_age_hours']} hours")

    if result["bhutani"]:
        b = result["bhutani"]
        print(f"\n  ── Bhutani Nomogram ──")
        print(f"  Risk Zone           : {b['zone']}")
        print(f"  Zone Code           : {b['zone_code']}  (0=Low ... 3=High)")
        print(f"  Thresholds @ {result['postnatal_age_hours']}h  : "
              f"P40={b['thresholds']['p40']}  P75={b['thresholds']['p75']}"
              f"  P95={b['thresholds']['p95']} mg/dL")
        print(f"\n  Action: {b['action']}")

    print()

    # ── Standalone Bhutani helper demo ─────────────────────────────────
    print("─── Bhutani lookup examples ───")
    for age_h, tsb in [(48, 8.0), (72, 14.0), (72, 17.5), (96, 11.5)]:
        r = bhutani_risk_zone(tsb, age_h)
        print(f"  Age={age_h:>3}h  TSB={tsb:>5.1f} mg/dL  →  {r['zone']}")