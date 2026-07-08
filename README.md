
***
# 🧪 Swelling Ratio Predictor
> A browser-based machine learning web application designed to predict polymer **equilibrium swelling ratios (SR)**. Powered by Pyodide, it runs a complete Python ML pipeline directly in the browser without any backend server, ensuring data privacy and zero deployment cost.
![Made with Python](https://img.shields.io/badge/Made_with-Pyodide-3776AB?logo=python&logoColor=white)
![RDKit](https://img.shields.io/badge/Chemistry-RDKit.js-0066CC)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E)
![License](https://img.shields.io/badge/License-MIT-green)
## 🌟 Key Features
- **100% Client-Side Execution:** Runs entirely in the browser as a single HTML file. No data is ever uploaded to a server.
- **Offline Friendly:** After the initial load (which downloads Pyodide and dependencies from CDN), the application is cached by the browser and can be used offline.
- **End-to-End ML Pipeline:** Automates the entire process from data preprocessing (KNN imputation, scaling, encoding) to model training and prediction.
- **Cheminformatics Integration:** Utilizes RDKit.js to calculate real-time mole-weighted molecular descriptors (logP, TPSA, MR) directly from SMILES strings.
- **Explainable AI (XAI):** Integrates TreeSHAP to provide both global feature importance and individual prediction attribution.
---
## 🛠️ Tech Stack
| Component | Purpose | Source |
| :--- | :--- | :--- |
| **Pyodide (v0.26.2)** | In-browser Python runtime | CDN |
| **RDKit.js (v2023.3.3)** | SMILES → Molecular descriptors (logP/TPSA/MR) | CDN |
| **Plotly.js** | SHAP visualizations & data plotting | CDN |
| **scikit-learn** | ExtraTreesRegressor + KNNImputer + OrdinalEncoder + StandardScaler | Pyodide |
| **shap (optional)** | TreeExplainer SHAP values; auto-fallback to Saabas approximation if install fails | micropip |
---
## 📊 Application Modules (3 Tabs)
### Tab 1 — Data & Model
- **Data Import:** Upload a local CSV file or load the built-in default dataset (745 rows × 14 columns, embedded in the HTML).
- **Data Statistics:** Displays row count, column count, missing value count, and a 5-row preview.
- **Model Training:** Executes the pipeline: `KNN Imputation (if needed) → StandardScaler + OrdinalEncoder → ExtraTreesRegressor (random_state=42)`.
- **Evaluation:** Displays R², RMSE, MAE metrics, and a Predicted vs. True value scatter plot.
### Tab 2 — Prediction
- **Categorical Inputs:** Dropdowns for Backbone Type (19 classes) and Initiator Type (11 classes).
- **Numeric Inputs:** Input fields for `bb_ratio`, `init_ratio`, `cl_ratio`, `neutral_deg` (%), `temp` (°C), `time` (h), `dry_temp` (°C), `swell_temp` (°C) with units.
- **Monomer Info:** SMILES strings and molar amounts for up to 2 monomers.
- **Descriptor Calculation:** "Calculate Descriptors" button triggers RDKit.js to compute mole-weighted `weighted_logP`, `weighted_TPSA`, and `weighted_MR`.
- **Prediction Output:** "Predict SR" outputs the predicted equilibrium swelling ratio (g/g).
### Tab 3 — SHAP Analysis
- **Global Feature Importance:** Bar chart showing Mean |SHAP| values, sorted by importance.
- **Individual SHAP Values:** Waterfall plot detailing how each feature contributes to the specific prediction made in Tab 2.
---
## 💡 Key Design Details
1. **Smart KNN Imputation:**
   - Only triggered if missing values exist.
   - Categorical variables are encoded via `OrdinalEncoder` → imputed by KNN → decoded back to strings.
   - The string `"None"` is treated as a valid categorical class and is preserved during imputation.
2. **SHAP Fallback Strategy:**
   - Attempts to install the `shap` package via `micropip` for exact TreeExplainer values.
   - If installation fails, it automatically falls back to the Saabas method (a fast tree-path-based approximation) to ensure functionality.
3. **File Compatibility:**
   - Automatically handles BOM-embedded CSV files (`csv_content.lstrip('\ufeff')`) to prevent parsing errors.
4. **UI/UX Design:**
   - Features a soft gradient background from light yellow (`#FFF9E6`) to light red (`#FFEBEE`), with white cards and light red borders.
---
## 🚀 Usage Workflow
1. **Launch App:** Download the `swelling_ratio_predictor.html` file and open it in a modern browser (Chrome/Edge recommended).
2. **Wait for Loading:** The first launch takes 30-60 seconds to fetch Pyodide and Python packages from the CDN.
3. **Load Data & Train:** Go to the "Data & Model" tab → click **"Use Default Dataset"** → click **"Train Model"**.
4. **Make Predictions:**
   - Switch to the "Prediction" tab.
   - Fill in the synthesis parameters and monomer SMILES.
   - Click **"Calculate Descriptors"** to compute molecular features.
   - Click **"Predict SR"** to get the result.
5. **Model Explanation:**
   - Switch to the "SHAP Analysis" tab.
   - Click **"Calculate SHAP Values"** to view global feature importance and local prediction attributions.
---
## 📦 Local Development
This project is a pure front-end single-file application. To modify or extend its functionality:
1. Download/clone the repository files to your local machine.
2. Open `swelling_ratio_predictor.html` using any code editor (e.g., VS Code).
3. Modify the embedded HTML/CSS/JavaScript code as needed.
4. Refresh the file in your browser to see the changes. No build steps or Node.js environment are required.
> **Note:** To use this application in a completely offline environment, you must download the Pyodide, RDKit.js, and Plotly.js CDN resources locally and update the `<script>` and `<link>` tags in the HTML file accordingly.
## 📄 License
This project is licensed under the MIT License.

