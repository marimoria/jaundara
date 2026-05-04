"""
predict.py — Neonatal Jaundice Inference
=========================================
Load saved models and run prediction on a new patient.
Run AFTER train_models.py.

Usage:
  python predict.py

Or import predict_patient() into your Flutter backend / REST API.
"""

import pickle
import numpy as np
import pandas as pd

MODELS_DIR    = "models"
SEV_THRESHOLD = 15.0

def load_model(name):
    with open(f"{MODELS_DIR}/{name}.pkl", "rb") as f:
        return pickle.load(f)

def predict_patient(
    zone_features: dict,           # all 42 zone color features as key:value
    gestational_age: float = None, # type: ignore # weeks (optional)
    postnatal_age_days: float = None, # type: ignore
    weight: float = None,          # type: ignore # grams
):
    """
    Returns a dict:
      {
        "jaundiced": True/False,
        "severity": "Mild" | "Severe" | None,
        "action": "Monitor 24h" | "Recheck 12h" | "Seek help today",
        "model_used_detection": "1A" or "1B",
        "model_used_severity":  "2A" or "2B" or None,
        "jaundice_proba": float,
        "severity_proba": float or None,
      }
    """
    has_meta = all(v is not None for v in
                   [gestational_age, postnatal_age_days, weight])

    # Build a 1-row DataFrame
    row = {**zone_features}
    if has_meta:
        row["gestational_age"]    = gestational_age
        row["postnatal_age_days"] = postnatal_age_days
        row["weight"]             = weight

    df_row = pd.DataFrame([row])

    # ── Step 1: Detection ──────────────────────────────────────
    det_key = "1A" if has_meta else "1B"
    det     = load_model(f"model_{det_key}")
    det_feats = [f for f in det["features"] if f in df_row.columns]
    det_proba = det["model"].predict_proba(df_row[det_feats])[0][1]
    jaundiced = det_proba >= 0.5

    if not jaundiced:
        return {
            "jaundiced": False,
            "severity": None,
            "action": "Monitor & recheck in 24 hours",
            "model_used_detection": det_key,
            "model_used_severity": None,
            "jaundice_proba": round(det_proba, 4),
            "severity_proba": None,
        }

    # ── Step 2: Severity ───────────────────────────────────────
    sev_key = "2A" if has_meta else "2B"
    sev     = load_model(f"model_{sev_key}")
    sev_feats = [f for f in sev["features"] if f in df_row.columns]
    sev_proba = sev["model"].predict_proba(df_row[sev_feats])[0][1]
    is_severe = sev_proba >= 0.5

    severity = "Severe" if is_severe else "Mild"
    action   = "Seek help today" if is_severe else "Recheck in 12 hours"

    return {
        "jaundiced": True,
        "severity": severity,
        "action": action,
        "model_used_detection": det_key,
        "model_used_severity": sev_key,
        "jaundice_proba": round(det_proba, 4),
        "severity_proba": round(sev_proba, 4),
    }


# ── Demo: run on a fake patient ────────────────────────────────
if __name__ == "__main__":
    # Replace these values with real extracted features from your image pipeline
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

    result = predict_patient(
        zone_features=sample_zones,
        gestational_age=37,
        postnatal_age_days=3,
        weight=2800,
    )

    print("\n═══ PREDICTION RESULT ═══")
    for k, v in result.items():
        print(f"  {k:<28}: {v}")
