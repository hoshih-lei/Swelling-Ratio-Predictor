#!/usr/bin/env python3
"""
Train ExtraTreesRegressor on the new dataset and export model to JSON for JS inference.
"""
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CSV_PATH = "/home/z/my-project/upload/6a5ecb5dc2f546f92fdcc7eb_data_descriptors_afterfeatureselect -new.csv"
OUTPUT_PATH = "/home/z/my-project/swelling_predictor/model_data.json"

# ── 1. Load data ────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, header=0, keep_default_na=False, na_values=["", " "])
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Missing values: {df.isna().sum().sum()}")

# ── 2. Define features ───────────────────────────────────────
numeric_features = [c for c in df.select_dtypes('number').columns if c != 'SR']
categorical_features = ['backbone', 'initiator']
target = 'SR'

feature_names = numeric_features + categorical_features
X = df[feature_names].copy()
y = df[target].copy()

# Cast categorical to string for encoder
for c in categorical_features:
    X[c] = X[c].astype(str)

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
print(f"Total features: {len(feature_names)}")

# ── 3. KNN Imputation (if needed) ───────────────────────────
if X.isna().sum().sum() > 0:
    print("Applying KNN imputation...")
    # Encode categorical first for KNN
    temp_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_cat_encoded = temp_encoder.fit_transform(X[categorical_features])
    X_combined = X.copy()
    for i, c in enumerate(categorical_features):
        X_combined[c] = X_cat_encoded[:, i]
    
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X_combined)
    
    # Decode categorical back
    X_cat_decoded = temp_encoder.inverse_transform(X_imputed[:, len(numeric_features):])
    X = pd.DataFrame()
    for i, c in enumerate(numeric_features):
        X[c] = X_imputed[:, i]
    for i, c in enumerate(categorical_features):
        X[c] = X_cat_decoded[:, i]
    print(f"Missing after imputation: {X.isna().sum().sum()}")

# ── 4. Train/test split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
)

# ── 5. Build pipeline ──────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ), categorical_features),
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', ExtraTreesRegressor(random_state=RANDOM_STATE)),
])

pipeline.fit(X_train, y_train)
print("ExtraTreesRegressor trained.")

# ── 6. Compute metrics ─────────────────────────────────────
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

metrics = {
    "r2_train": float(r2_score(y_train, y_train_pred)),
    "r2_test": float(r2_score(y_test, y_test_pred)),
    "rmse_train": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
    "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
    "mae_train": float(mean_absolute_error(y_train, y_train_pred)),
    "mae_test": float(mean_absolute_error(y_test, y_test_pred)),
}
print(f"\nMetrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# ── 7. Export model to JSON ─────────────────────────────────
model = pipeline.named_steps['model']
preprocessor_fitted = pipeline.named_steps['preprocessor']

# Export trees
trees = []
for est in model.estimators_:
    tree = est.tree_
    trees.append({
        "cl": tree.children_left.tolist(),
        "cr": tree.children_right.tolist(),
        "f": tree.feature.tolist(),
        "t": tree.threshold.tolist(),
        "v": tree.value.flatten().tolist(),
    })

# Export preprocessor
scaler = preprocessor_fitted.named_transformers_['num']
encoder = preprocessor_fitted.named_transformers_['cat']

# Export test set for SHAP computation
X_test_trans = preprocessor_fitted.transform(X_test)
test_data = X_test_trans.tolist()
test_target = y_test.tolist()

# Export full dataset for predictions
X_full_trans = preprocessor_fitted.transform(X)
full_predictions = pipeline.predict(X).tolist()
full_targets = y.tolist()

model_data = {
    "trees": trees,
    "n_trees": len(trees),
    "preprocessor": {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "scaler_var": scaler.var_.tolist(),
        "encoder_categories": [cat.tolist() for cat in encoder.categories_],
    },
    "feature_names": feature_names,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "n_features": len(feature_names),
    "metrics": metrics,
    "test_data": test_data,
    "test_target": test_target,
    "full_predictions": full_predictions,
    "full_targets": full_targets,
}

# Save
with open(OUTPUT_PATH, 'w') as f:
    json.dump(model_data, f)

import os
size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
n_nodes = sum(len(t["v"]) for t in trees)
print(f"\nModel exported to: {OUTPUT_PATH}")
print(f"File size: {size_mb:.2f} MB")
print(f"Number of trees: {len(trees)}")
print(f"Total nodes: {n_nodes}")
print(f"Features: {feature_names}")
print(f"Encoder categories:")
for i, cats in enumerate(encoder.categories_):
    print(f"  {categorical_features[i]}: {cats.tolist()}")
