# Jaundara: Smartphone-Based Neonatal Jaundice Detection via Multi-Color Feature Analysis and SHAP

Jaundara (Jaundice Indra) is an end-to-end machine learning pipeline for the non-invasive detection of neonatal jaundice (ikterus neonatorum) from smartphone skin images. Designed for independent use by mothers and midwives, the system runs entirely on-device, bridging the gap between clinical accuracy and household accessibility to prevent fatal neurological complications like kernicterus.

## Background

Neonatal jaundice—, haracterized by the yellowing of the skin and sclera due to elevated bilirubin levels, is a critical public health issue. According to the Global Burden of Disease (GBD) 2021, its prevalence increased by 41.5% between 1990 and 2021. In Indonesia, the 2024 Health Profile reported that 26,657 of the 31,393 infant deaths occurred during the neonatal period, with complications related to conditions like severe jaundice playing a significant role.

Existing diagnostic and screening methods present barriers to early, independent detection:

- **Total Serum Bilirubin (TSB) Sampling:** The clinical gold standard, but it is invasive, painful, and requires laboratory infrastructure often unavailable in remote areas.
- **Transcutaneous Bilirubinometry:** Non-invasive alternatives (e.g., BiliCare, JM-105) are highly accurate but prohibitively expensive (USD 3,000–5,000) for primary healthcare facilities.
- **Kramer Visual Scale:** Clinical visual assessment has limited sensitivity and relies heavily on subjective clinical experience.
- **Prior AI Systems:** Existing computer vision approaches often focus on the ocular sclera (impractical for sleeping newborns) or provide only binary classifications without severity gradation, limiting their utility for household screening.

Jaundara addresses these gaps by analyzing skin images from Kramer Zones 1, 2, and 3 (forehead, cheek, and sternum). It combines LightGBM regression modeling with Bhutani Nomogram mapping to provide actionable, severity-graded risk stratification.

## Repository Structure

```text
.
├── __data__/                # Dataset storage (raw images, extracted features, and EDA outputs)
├── __models__/              # LightGBM model weights output directory (.pkl)
├── __plots__/               # Model evaluation and EDA visualization output directory
├── .ruff_cache/             # Linter cache
├── .vscode/                 # Editor configurations
├── color_extraction/        # Core image processing and color feature extraction module
│   ├── __init__.py
│   ├── __main__.py
│   ├── augmentation.py      # Brightness augmentation strictly on the HSL L-channel
│   ├── cli.py
│   ├── color_math.py        # Pure color space conversions (RGB, XYZ, CIELAB, HSL)
│   ├── dataset_pipeline.py
│   ├── debug_visualizer.py
│   ├── feature_extractor.py # Computation of 14 statistical color features per zone
│   ├── image_processor.py   # Center-crop (40%) and pipeline orchestration
│   └── skin_mask.py         # Dual-range HSV algorithm for robust neonatal skin segmentation
├── data_exploration/        # Data exploration and feature engineering pipeline
│   ├── phase1_exploration.py  # Structure audit, target distribution, feature skewness
│   ├── phase2_exploration.py  # Kolmogorov-Smirnov normality tests
│   ├── phase3_exploration.py  # Spearman correlation, intra-zone redundancy, cross-zone Kramer's rule
│   └── phase4_exploration.py  # Feature engineering based on correlation gain thresholds
├── training/                # Modeling, tuning, and evaluation scripts
│   ├── evaluate.py          # Evaluation plot generation (ROC, regression residuals)
│   ├── predict.py           # End-to-end inference simulating the on-device pipeline
│   ├── train_models.py      # LightGBM training (Classification and Regression models)
│   └── tune_regression.py   # Bayesian hyperparameter optimization via Optuna
├── .env                     # Local environment variables (HuggingFace token, etc.)
├── .env.template            # Template for the .env file
├── .gitignore               # Git exclusion configuration
├── hf_manager.py            # Interactive HuggingFace dataset synchronization CLI
├── README.md                # Project documentation
└── requirements.txt         # Python dependency list
```

## Methodology

### 1. Dataset and Preprocessing

- **Dataset:** Built on the NeoJaundice dataset containing 2,235 images from 745 neonates, utilizing three anatomical zones (forehead, cheek, sternum) alongside clinical metadata (gestational age, postnatal age, birth weight).
- **Image Processing:** Images undergo a 40% center crop to remove color reference card borders.
- **Dual-Range HSV Segmentation:** A robust binary mask is constructed using a union of two HSV ranges to account for both light and dark neonatal skin tones, refined via morphological close/open operations.
- **Targeted Augmentation:** To simulate varied smartphone camera lighting while preserving clinical integrity, brightness augmentation is applied _only_ to the L (Lightness) channel in the HSL space (factor: 0.8 to 1.2). Crucial diagnostic channels (Hue, Cr, CIELAB b\*) remain untouched.

### 2. Feature Engineering & Exploration Pipeline

The project utilizes a rigorous 4-phase exploratory data analysis (EDA) and engineering pipeline:

- **Phase 1 & 2:** Analyzes the `blood_mg_dl` target distribution and performs Kolmogorov-Smirnov normality testing, confirming non-normal distributions and establishing Spearman's rank as the primary correlation metric.
- **Phase 3:** Validates Kramer's Rule computationally by demonstrating that correlations with TSB increase from Zone 1 to Zone 3 (e.g., `zone3_Lab_b_mean` shows stronger correlation than `zone1`).
- **Phase 4:** Engineers 18 new features (e.g., G-B difference, R/B ratio, cross-zone gradients, and Individual Typology Angle). Features are only retained if they demonstrate a clear Spearman correlation gain (≥ 0.02) over their base components.

### 3. LightGBM Modeling and SHAP Feature Selection

- **Algorithm:** LightGBM was selected for its exceptional tabular data performance, histogram-based speed, and compact model size (~4.83 MB), making it ideal for on-device Flutter integration.
- **SHAP Selection:** Out of 42 combined raw and engineered features, SHAP (SHapley Additive exPlanations) values filter the dataset down to the most highly predictive subsets—27 features for detection and 31 for regression. Top contributors include `postnatal_age_days`, `zone3_Lab_b_mean`, `zone3_H_mean`, and `zone3_G_minus_B`.
- **Hyperparameter Tuning:** Model 2 (Regression) undergoes 100 iterations of Bayesian optimization via Optuna to minimize MAE and RMSE.

### 4. Bhutani Nomogram Mapping

The pipeline doesn't stop at raw TSB prediction. Model 2's continuous TSB output is evaluated alongside postnatal age (in hours) against the clinical standard **Bhutani Nomogram**. This translates complex biochemical estimations into four clear clinical risk zones (Low, Low Risk, High-Intermediate, High Risk) with actionable recommendations for parents.

## Evaluation Results

Models were rigorously evaluated on a 70/15/15 train/validation/test split, partitioned strictly at the patient level to prevent data leakage.

| Model       | Task                                        | Features Used      | Test Set Performance                                                                  |
| ----------- | ------------------------------------------- | ------------------ | ------------------------------------------------------------------------------------- |
| **Model 1** | Binary Classification (Jaundice vs. Normal) | 27 (SHAP Selected) | **Accuracy:** 84.82% <br><br> **F1-Score:** 82.47% <br><br> **AUC-ROC:** 93.32%       |
| **Model 2** | TSB Regression                              | 31 (SHAP Selected) | **RMSE:** 2.517 mg/dL <br><br> **R²:** 0.7775 <br><br> **Accuracy (±2 mg/dL):** 54.5% |

## Usage

### 1. Dataset Synchronization (`hf_manager.py`)

Keep local and remote datasets synchronized using the interactive HuggingFace CLI tool. Make sure to define `HF_TOKEN` in your `.env` file first.

```bash
python hf_manager.py
```

This launches an interactive menu with the following capabilities:

1. **List local files:** Inspect sizes and paths of the currently active dataset directory.
2. **Upload local files:** Selectively (or entirely) batch upload files/folders to the HuggingFace Hub.
3. **Download ALL files:** Pull a full repository snapshot locally.
4. **Download specific file(s):** Fetch targeted files without downloading the entire repo.
5. **Update / re-upload:** Sync modified local data back to the Hub.
6. **Change Repository and Directory:** Switch targets on the fly.

### 2. Feature Extraction

Extract the validated color features from a raw image directory to build the final tabular dataset.

```bash
# Batch extraction for training dataset construction
python -m color_extraction training \
    --image_dir __data__/neo/images \
    --clinical_csv __data__/neo/neo.csv \
    --output __data__/neo/out/training_engineered.csv

# Single-image debug extraction
python -m color_extraction debug \
    --image __data__/neo/images/0003-1.jpg \
    --debug_dir __data__/neo/out/debug \
    --augment
```

### 3. Data Exploration & Feature Engineering

Run the phased EDA pipeline to evaluate normality, map correlations, and engineer features based on the extracted tabular dataset.

```bash
python data_exploration/phase1_exploration.py
python data_exploration/phase2_exploration.py
python data_exploration/phase3_exploration.py
python data_exploration/phase4_exploration.py
```

### 4. Model Training and Hyperparameter Tuning

```bash
# Train LightGBM models with SHAP integration and detailed logging
python training/train_models.py --log

# Bayesian hyperparameter search via Optuna for TSB Regression
python training/tune_regression.py
```

### 5. Evaluation & Inference

```bash
# Generate evaluation metrics, ROC curves, and residual plots
python training/evaluate.py

# Simulate the full on-device pipeline (Classification -> Regression -> Bhutani Nomogram)
python training/predict.py
```

---

_Submitted to SATRIA DATA 2026_
