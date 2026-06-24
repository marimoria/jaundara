# Jaundara: Smartphone-Based Neonatal Jaundice Detection via Skin Color Analysis

Jaundara (Jaundice Indra) is an end-to-end machine learning pipeline for non-invasive detection of neonatal jaundice (hyperbilirubinemia) from skin images. The system is designed for independent use by caregivers and midwives via smartphone, with the aim of preventing fatal complications such as kernicterus in newborns.

---

## Background

Neonatal jaundice is a condition characterized by skin yellowing due to elevated serum bilirubin, contributing significantly to neonatal mortality in Indonesia (26,657 deaths in 2024). When detected late, bilirubin can penetrate the blood-brain barrier and cause permanent neurological damage (kernicterus).

Existing diagnostic and screening methods present barriers to early independent detection:

1. **Transcutaneous Serum Bilirubin (TSB) Sampling.** The diagnostic gold standard, but invasive, painful, and requires laboratory infrastructure.
2. **Transcutaneous Bilirubinometry.** A non-invasive commercial alternative, but prohibitively expensive (USD 3,000 to USD 5,000 per unit) for primary healthcare facilities.
3. **Kramer Visual Scale.** Clinical visual assessment with limited sensitivity and specificity (approximately 70%), owing to its reliance on subjective clinical experience.
4. **Sclera-Based AI Solutions.** Prior computer vision research targeting the ocular sclera is impractical for home use, as neonatal eyes are frequently closed and difficult to photograph with precision.
5. **Prior AI Systems (NJN Dataset).** Existing models provide only binary classification (normal/jaundice) without severity gradation and are designed for NICU specialists rather than household screening.

Jaundara addresses these gaps through skin image analysis (Kramer Zones 1 and 2), combined with TSB regression modeling and Bhutani Nomogram mapping for actionable risk stratification.

---

## Repository Structure

```
.
├── __data__/                # Dataset storage (raw images and extracted CSV files)
├── __models__/              # LightGBM model weights output directory (.pkl)
├── __plots__/               # Model evaluation visualization output directory
├── color_extraction/        # Core image processing and color feature extraction module
│   ├── __main__.py          # CLI entry point for color_extraction
│   ├── augmentation.py      # Brightness augmentation on the HSL L channel
│   ├── cli.py               # Command-line interface (training and debug modes)
│   ├── color_math.py        # Pure color space conversions (RGB to XYZ, CIELAB, HSL)
│   ├── dataset_pipeline.py  # Tabular dataset construction orchestrator
│   ├── debug_visualizer.py  # Multi-panel diagnostic image generator
│   ├── feature_extractor.py # Computation of 14 statistical color features
│   ├── image_processor.py   # Center-crop and mask application
│   ├── __init__.py          # Public module exports
│   └── skin_mask.py         # Dual-range HSV algorithm for neonatal skin segmentation
├── training/                # Modeling, tuning, and evaluation scripts
│   ├── evaluate.py          # Evaluation plot generation (ROC, regression residuals, Nomogram)
│   ├── predict.py           # End-to-end inference pipeline
│   ├── train_models.py      # LightGBM training script (Models 1A/1B and 2A/2B)
│   └── tune_regression.py   # Bayesian hyperparameter optimization via Optuna
├── .env                     # Local environment variables (HuggingFace token, etc.)
├── .env.template            # Template for the .env file
├── .gitignore               # Git exclusion configuration
├── hf_file_manager.py       # Dataset synchronization with HuggingFace Hub
├── README.md                # Project documentation
└── requirements.txt         # Python dependency list
```

---

## Methodology

### 1. Dataset and Preprocessing

- **Dataset.** The NeoJaundice dataset comprising 2,235 images from 745 neonates, covering three anatomical zones (forehead, cheek, sternum) with clinical metadata (gestational age, postnatal age, birth weight).
- **Center Crop.** Automatic 40-50% center cropping to remove the color reference card borders present in the dataset.
- **Dual-Range HSV Skin Segmentation.** Binary mask construction using the union of two HSV value ranges (covering light and dark neonatal skin tones), refined with morphological operations using a 7x7 elliptical kernel.
- **Domain-Specific Augmentation.** Camera lighting variation is simulated by applying brightness augmentation exclusively to the `L` channel in HSL color space (scale factors in [0.8, 1.2]), preserving diagnostically relevant channels (Hue, Cr, b\*).

### 2. Feature Engineering

From validated skin pixels, 14 statistical features are extracted per anatomical zone, yielding 42 total color features across four color spaces:

- **RGB.** Baseline color representation (`R_mean`, `R_std`).
- **YCbCr.** The `Cr` channel directly encodes spectral shifts associated with bilirubin.
- **HSL.** The `H` (Hue) channel is used for its invariance to illumination intensity changes.
- **CIELAB.** The `b*` axis (blue-yellow) is the most clinically validated and device-independent single indicator of jaundice severity.

### 3. LightGBM Modeling and SHAP Feature Selection

- **Algorithm.** LightGBM was selected for its performance on tabular data, short training time, and compact model export size, which is suitable for on-device smartphone deployment.
- **Model Variants.** To accommodate real-world scenarios where caregivers may not have access to birth records, models are trained in variant **A** (color features with clinical metadata) and variant **B** (color features only).
- **Prediction Tasks.** Model 1 performs binary classification. Model 2 is optimized via **Optuna** (100 Bayesian search trials) for Total Serum Bilirubin (TSB) regression.
- **SHAP-Based Feature Selection.** Features with marginal SHAP contribution below 1% are removed, retaining only high-weight features such as `postnatal_age_days`, `zone3_H_mean`, and `zone3_Lab_b_mean`.

### 4. Bhutani Nomogram Mapping

TSB regression output from Model 2 is not the final clinical result. The predicted TSB value is combined with postnatal age (in hours) and mapped onto the **Bhutani Nomogram** to produce a clinical risk zone category (Low, Low-Intermediate, High-Intermediate, High) with corresponding actionable recommendations for the caregiver.

---

## Evaluation Results

Models were evaluated using a stratified 70/15/15 train/validation/test split with strict patient-level partitioning to prevent data leakage.

| Model    | Task                  | Input                      | Test Set Performance                                  |
| -------- | --------------------- | -------------------------- | ----------------------------------------------------- |
| Model 1A | Binary Classification | Color (3 zones) + Metadata | Accuracy: 84.82% / AUC-ROC: 91.96%                    |
| Model 1B | Binary Classification | Color only                 | Accuracy: 82.14% / AUC-ROC: 87.25%                    |
| Model 2A | TSB Regression        | Color (3 zones) + Metadata | MAE: 2.382 mg/dL / R2: 0.6507 / within 2 mg/dL: 55.4% |
| Model 2B | TSB Regression        | Color only                 | MAE: 3.040 mg/dL / R2: 0.4992                         |

---

## Usage

### 1. Feature Extraction and Dataset Construction

Run the `color_extraction` module to process a raw image directory and match it against a clinical CSV to build the training dataset.

```bash
# Batch extraction for training dataset construction
python -m color_extraction training \
    --image_dir __data__/neo/images \
    --clinical_csv __data__/neo/neo.csv \
    --output __data__/neo/out/training_fix.csv

# Single-image debug extraction
python -m color_extraction debug \
    --image __data__/neo/images/0003-1.jpg \
    --debug_dir out/debug \
    --augment
```

### 2. Model Training and Hyperparameter Tuning

```bash
# Train Models 1A, 1B, 2A, and 2B with detailed logging
python training/train_models.py --log

# Bayesian hyperparameter search via Optuna (minimizes MAE)
python training/tune_regression.py
```

### 3. Evaluation

Loads trained models and writes statistical test metrics and visualizations to `__plots__/`.

```bash
python training/evaluate.py
```

### 4. End-to-End Inference

Simulates the full detection pipeline, returning binary classification, estimated TSB value, and Bhutani Nomogram risk category with recommended action.

```bash
python training/predict.py
```

### 5. HuggingFace Dataset Synchronization

```bash
python hf_file_manager.py
```

---

_Submitted to GAMMAFEST 2026 (ID: ESC26032)_
