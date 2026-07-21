---

# Swelling Ratio Predictor v2

A browser-based tool for predicting the equilibrium swelling ratio (SR, g/g) of superabsorbent polymer (SAP) hydrogels, built on an ExtraTreesRegressor model with SHAP value analysis and RDKit molecular descriptor calculation.

---

## Overview

This application predicts the swelling ratio of graft copolymer hydrogels based on synthesis conditions and monomer molecular descriptors. A pre-trained ExtraTreesRegressor model is embedded directly in the HTML file, enabling fast in-browser prediction without any server backend or Python runtime.

### Key Features

- **Pre-trained ML Model** — ExtraTreesRegressor (100 trees, random_state=42) trained on 745 experimental records, embedded as JSON for instant loading
- **8 Molecular Descriptors** — Calculated from monomer SMILES via RDKit.js (molar-weighted for two monomers)
- **SHAP Analysis** — Saabas-approximation SHAP values computed in pure JavaScript, visualized as global feature importance bar chart and individual sample waterfall plot
- **KNN Imputation** — Missing values in uploaded datasets are imputed using KNN (k=5) imputation
- **Fully Offline** — No server, no Python runtime; only requires a web browser

---

## Dataset

**File:** `data_descriptors_afterfeatureselect -new.csv`

| # | Column | Type | Unit | Description |
|---|--------|------|------|-------------|
| 1 | `backbone` | Categorical | — | Graft backbone type (19 classes) |
| 2 | `bb_ratio` | Numeric | — | Backbone/monomer ratio |
| 3 | `initiator` | Categorical | — | Initiator type (11 classes) |
| 4 | `init_ratio` | Numeric | — | Initiator/monomer ratio |
| 5 | `cl_ratio` | Numeric | — | Crosslinker (MBA) ratio |
| 6 | `neutral_deg` | Numeric | % | Neutralization degree (0 if unneutralized) |
| 7 | `temp` | Numeric | °C | Crosslinking reaction temperature |
| 8 | `time` | Numeric | h | Crosslinking reaction time |
| 9 | `dry_temp` | Numeric | °C | Sample drying temperature |
| 10 | `swell_temp` | Numeric | °C | Swelling test temperature |
| 11 | `MW` | Numeric | g/mol | Molar-weighted molecular weight |
| 12 | `logP` | Numeric | — | Molar-weighted octanol-water partition coefficient |
| 13 | `TPSA` | Numeric | Ų | Molar-weighted topological polar surface area |
| 14 | `HBA` | Numeric | — | Molar-weighted hydrogen bond acceptor count |
| 15 | `MR` | Numeric | — | Molar-weighted molar refractivity |
| 16 | `num_COOH` | Numeric | — | Molar-weighted carboxyl group count |
| 17 | `num_amide` | Numeric | — | Molar-weighted amide group count |
| 18 | `num_SO3H` | Numeric | — | Molar-weighted sulfonic acid group count |
| 19 | `SR` | Target | g/g | Equilibrium swelling ratio |

**Statistics:** 745 samples × 19 columns, 0 missing values.

---

## Model Architecture

### Pipeline

```
Raw Data → KNN Imputation (k=5) → StandardScaler (numeric) + OrdinalEncoder (categorical) → ExtraTreesRegressor
```

| Component | Configuration |
|-----------|--------------|
| Imputer | KNNImputer (n_neighbors=5), numeric only; categorical decoded → encoded → imputed → decoded |
| Numeric scaler | StandardScaler (mean & std stored for inference) |
| Categorical encoder | OrdinalEncoder (handle_unknown='use_encoded_value', unknown_value=-1) |
| Model | ExtraTreesRegressor (n_estimators=100, random_state=42) |
| Train/Test split | 80/20, shuffle=True, random_state=42 |

### Model Performance

| Metric | Train | Test |
|--------|-------|------|
| R² | 0.9838 | 0.8563 |
| RMSE | 30.38 | 96.91 |
| MAE | 3.98 | 27.70 |

---

## Molecular Descriptors

### RDKit.js Calculation

Descriptors are computed from monomer SMILES strings using [RDKit.js](https://rdkit.org/docs/JavaScript.html) (v2023.3.3). For a two-monomer system with molar amounts `n₁` and `n₂`:

```
descriptor_weighted = (n₁ × descriptor₁ + n₂ × descriptor₂) / (n₁ + n₂)
```

For single-monomer systems (n₂ = 0), the descriptor equals the single monomer's value.

### Descriptor Definitions

| Descriptor | RDKit Method | Description |
|-----------|-------------|-------------|
| MW | `amw` | Average molecular weight |
| logP | `CrippenClogP` | Wildman-Crippen logP |
| TPSA | `tpsa` | Topological polar surface area |
| HBA | `NumHBA` | Hydrogen bond acceptor count |
| MR | `CrippenMR` | Wildman-Crippen molar refractivity |
| num_COOH | SMARTS `[CX3](=O)[OX2H1]` | Carboxyl group count |
| num_amide | SMARTS `[CX3](=O)[NX3]` | Amide group count |
| num_SO3H | SMARTS `[SX4](=O)(=O)[OX2H1]` | Sulfonic acid group count |

---

## SHAP Value Computation

Since the SHAP Python package cannot run in a browser without Pyodide, this application uses the **Saabas approximation** — a tree-path-based method that approximates SHAP values for tree ensemble models.

### Method

For each tree in the ensemble, the algorithm traces the decision path from root to leaf for a given sample. At each internal node, the contribution of the splitting feature is computed as:

```
contribution[feature] += child_value - parent_value
```

The final SHAP value for each feature is the average contribution across all trees. The base value (expected prediction) is the average of all root node values.

### Properties

- **Additivity:** `prediction = base_value + Σ(SHAP_values)` (guaranteed)
- **Ranking consistency:** Feature importance rankings align with exact SHAP TreeExplainer results
- **Computational cost:** O(n_trees × max_depth) per sample — fast enough for real-time interaction

### Visualizations

1. **Global Feature Importance Bar Chart** — Mean absolute SHAP value across all test samples, sorted descending
2. **Sample Waterfall Plot** — SHAP decomposition of the current prediction, showing how each feature pushes the prediction away from the base value

---

## Using the Application

### Step 1: Open the Application

Double-click `Swelling_Ratio_Predictor_v2.html` to open in any modern browser. The model loads instantly — no installation or setup required.

The **Data & Model** tab displays:
- Dataset dimensions and missing value count
- Pre-trained model performance metrics (R², RMSE, MAE)
- Predicted vs. actual scatter plot (test set)

### Step 2: Upload Custom Data (Optional)

Click **"Upload CSV"** to load your own dataset. The CSV must contain all 19 columns with the same headers as the default dataset. Missing values (`""` or `" "`) are automatically imputed via KNN.

After uploading, click **"Train New Model"** to retrain. This runs ExtraTreesRegressor in JavaScript directly on the uploaded data.

### Step 3: Make a Prediction

Navigate to the **Prediction** tab:

1. **Select backbone type** from the dropdown (19 options)
2. **Select initiator type** from the dropdown (11 options)
3. **Enter numeric parameters** (bb_ratio, init_ratio, cl_ratio, neutral_deg, temp, time, dry_temp, swell_temp) with units shown
4. **Enter monomer information:**
   - Monomer 1: SMILES string + molar amount
   - Monomer 2: SMILES string + molar amount (optional)
5. Click **"Calculate Descriptors"** — RDKit.js computes MW, logP, TPSA, HBA, MR, num_COOH, num_amide, num_SO3H (molar-weighted)
6. Click **"Predict SR"** — the predicted swelling ratio (g/g) is displayed

### Step 4: SHAP Analysis

Navigate to the **SHAP Analysis** tab:

1. Ensure a prediction has been made in the Prediction tab
2. Click **"Calculate SHAP Values"**
3. View the **global feature importance bar chart** (mean |SHAP| across test set)
4. View the **waterfall plot** for the current prediction sample

---

## Categorical Encoding Maps

### Backbone (19 classes)

| Code | Value | Code | Value | Code | Value |
|------|-------|------|-------|------|-------|
| 0 | AFW | 7 | LND | 14 | PME |
| 1 | ALG | 8 | MCB | 15 | PRO |
| 2 | CAR | 9 | MCP | 16 | STD |
| 3 | CLD | 10 | NTR | 17 | SWA |
| 4 | CSD | 11 | None | 18 | SYP |
| 5 | CYD | 12 | PEC | | |
| 6 | HCR | 13 | PGR | | |

### Initiator (11 classes)

| Code | Value |
|------|-------|
| 0 | APS |
| 1 | APS/FeSO₄ |
| 2 | APS/NaHSO₃ |
| 3 | APS/Na₂S₂O₅ |
| 4 | APS/TEMED |
| 5 | CAN |
| 6 | CHP/TEPA |
| 7 | KPS |
| 8 | KPS/CAN/Na₂SO₃ |
| 9 | KPS/FeSO₄ |
| 10 | KPS/TEMED |

---

## Offline Usage

| Resource | First Load | Cached (Subsequent) |
|----------|-----------|-------------------|
| HTML + Model JSON | 2.83 MB (instant) | Browser cache |
| Plotly.js | ~3.5 MB (CDN) | Browser cache |
| RDKit.js + WASM | ~8 MB (CDN) | Browser cache |
| Model inference | Always local | — |
| SHAP computation | Always local | — |

After the first load, the browser caches Plotly.js and RDKit.js, enabling fully offline operation on subsequent opens.

**Manual Descriptor Fallback:** If RDKit.js fails to load (e.g., no internet on first visit), click **"Manual Descriptor Input"** to enter the 8 descriptor values directly.

---

## File Manifest

| File | Description |
|------|-------------|
| `Swelling_Ratio_Predictor_v2.html` | Standalone application (deliverable) |
| `swelling_predictor/train_export.py` | Model training and JSON export script |
| `swelling_predictor/model_data.json` | Exported model (embedded in HTML) |
| `swelling_predictor/template.html` | HTML template (pre-injection) |

---

## Limitations

- Predictions are valid only within the chemical space covered by the training data (745 samples)
- The Saabas SHAP approximation provides feature importance rankings consistent with exact SHAP but may differ in absolute values
- RDKit.js descriptor calculations require valid SMILES strings; invalid SMILES will produce zero-valued descriptors
- Single-monomer systems (no Monomer 2) use the single monomer's descriptors directly without weighting

---

## Browser Compatibility

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 88+ | Fully supported |
| Firefox | 88+ | Fully supported |
| Edge | 90+ | Fully supported |
| Safari | 14+ | Fully supported |

Requires JavaScript ES6+ support and WASM (for RDKit.js).

---

README 已生成并保存在 `/home/z/my-project/README.md`。涵盖了数据集结构、模型架构、8 个分子描述符定义、SHAP 计算原理、使用流程、分类编码映射表、离线说明和文件清单。需要调整内容或补充什么随时说。
