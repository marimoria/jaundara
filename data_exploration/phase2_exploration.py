"""
Normality Test (Kolmogorov-Smirnov)
=====================================================================================
Dataset        : __data__/neo/out/training_engineered.csv (Updated for GitHub portability)
Response Var   : TSB -> 'blood_mg_dl' column
Features Tested: 45 features (Skin color zones RGB/YCrCb/HSL/Lab + basic clinical data)
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# 1. LOAD DATA (Updated with Relative Path for GitHub)
# ---------------------------------------------------------------
# Using relative path so anyone cloning this repository can run it immediately
df = pd.read_csv("__data__/neo/out/training_cleaned.csv")

RESPONSE = "blood_mg_dl"          # TSB (Total Serum Bilirubin)
EXCLUDE = ["patient_id", "is_augmented", "jaundice_label", RESPONSE]
FEATURES = [c for c in df.columns if c not in EXCLUDE]

print(f"Number of features being tested: {len(FEATURES)}")
assert len(FEATURES) == 45, "Feature count is not 45, please check the columns!"

# Handle missing values (postnatal_age_days only, 8 rows) -> drop those rows
# to prevent normality test bias caused by imputation
df_clean = df.dropna(subset=FEATURES + [RESPONSE]).copy()
print(f"Rows used after dropping NA: {df_clean.shape[0]} out of {df.shape[0]}")


# ---------------------------------------------------------------
# 2. KOLMOGOROV-SMIRNOV (KS) NORMALITY TEST FOR FEATURES
# ---------------------------------------------------------------
def ks_normality_test(series):
    x = series.dropna().values
    mu, sigma = np.mean(x), np.std(x, ddof=1)
    if sigma == 0:
        return np.nan, np.nan, mu, sigma
    stat, p = stats.kstest(x, 'norm', args=(mu, sigma))
    return stat, p, mu, sigma

results = []
for col in FEATURES:
    stat, p, mu, sigma = ks_normality_test(df_clean[col])
    skew = stats.skew(df_clean[col].dropna())
    kurt = stats.kurtosis(df_clean[col].dropna())  # excess kurtosis
    results.append({
        "feature": col,
        "n": df_clean[col].dropna().shape[0],
        "mean": mu,
        "std": sigma,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "KS_statistic": stat,
        "KS_pvalue": p,
        "normal_alpha_0.05": "Normal" if p > 0.05 else "Non-Normal"
    })

res_df = pd.DataFrame(results).sort_values("KS_pvalue", ascending=False).reset_index(drop=True)

n_normal = (res_df["normal_alpha_0.05"] == "Normal").sum()
n_total = res_df.shape[0]
print(f"\n=== KOLMOGOROV-SMIRNOV TEST RESULTS (alpha=0.05) ===")
print(f"Features with NORMAL distribution    : {n_normal} / {n_total}")
print(f"Features with NON-NORMAL distribution: {n_total - n_normal} / {n_total}")

# Save feature results using a relative path
res_df.to_csv("__plots__/explore/csv/ks_test_results.csv", index=False)
print("\nFull table saved -> ks_test_results.csv")
print(res_df.to_string(index=False))


# ---------------------------------------------------------------
# 3. NORMALITY TEST FOR RESPONSE VARIABLE (blood_mg_dl)
# ---------------------------------------------------------------
x = df["blood_mg_dl"].dropna().values
stat, p = stats.kstest(x, 'norm', args=(x.mean(), x.std(ddof=1)))  # type: ignore

print("\n=== Normality for blood_mg_dl ===")
print(f"KS Statistic : {stat:.4f}")
print(f"p-value      : {p:.2e}")
print(f"Conclusion   : {'Normal' if p > 0.05 else 'Non-Normal'}")
