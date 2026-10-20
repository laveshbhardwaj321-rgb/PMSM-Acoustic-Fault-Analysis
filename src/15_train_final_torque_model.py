
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent

DATA_FILE = (
    ROOT / "outputs" / "features"
    / "all_recordings_wavelet_features.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "models" / "final_candidate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "torque_nm"
GROUP = "run_id"
TOP_K = 10

RF_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 2,
    "max_features": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}


# Load labelled dataset
df = pd.read_csv(DATA_FILE)

for col in [TARGET, GROUP]:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET, GROUP]).copy()

feature_cols = [
    c for c in df.columns
    if c.startswith("yoke_")
    and pd.api.types.is_numeric_dtype(df[c])
]

if len(feature_cols) < TOP_K:
    raise ValueError(f"Only {len(feature_cols)} yoke features found.")

X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
y = df[TARGET].to_numpy()

# Impute missing feature values
imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X)

# Select top 10 features using the full labelled dataset.
# This is for the final model fit, not an independent evaluation.
selector = RandomForestRegressor(**RF_PARAMS)
selector.fit(X_imp, y)

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": selector.feature_importances_,
}).sort_values("importance", ascending=False)

selected_features = importance_df.head(TOP_K)["feature"].tolist()
selected_indices = [
    feature_cols.index(f) for f in selected_features
]

# Fit the final estimator on the selected features
model = RandomForestRegressor(**RF_PARAMS)
model.fit(X_imp[:, selected_indices], y)

# Save all inference components together
artifact = {
    "imputer": imputer,
    "model": model,
    "all_feature_names": feature_cols,
    "selected_features": selected_features,
    "target": TARGET,
    "training_torque_min_nm": float(np.min(y)),
    "training_torque_max_nm": float(np.max(y)),
    "training_recordings": int(df[GROUP].nunique()),
    "training_windows": int(len(df)),
    "model_type": "RandomForestRegressor",
    "selection_method": "Random Forest feature importance",
    "top_k": TOP_K,
    "warning": (
        "Preliminary candidate. Training data contains rotor-fault "
        "recordings only. Validate on new independent recordings."
    ),
}

model_path = OUTPUT_DIR / "pmsm_yoke_torque_model.joblib"
joblib.dump(artifact, model_path)

importance_df.to_csv(
    OUTPUT_DIR / "final_feature_importance.csv",
    index=False
)

with open(OUTPUT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            k: v for k, v in artifact.items()
            if k not in ["imputer", "model"]
        },
        f,
        indent=2
    )

print("\nFINAL CANDIDATE MODEL SAVED")
print(model_path)

print("\nSelected features:")
for rank, feature in enumerate(selected_features, start=1):
    value = importance_df.loc[
        importance_df["feature"] == feature, "importance"
    ].iloc[0]
    print(f"{rank:2d}. {feature}: importance={value:.5f}")

print(f"\nTraining recordings: {artifact['training_recordings']}")
print(f"Training windows: {artifact['training_windows']}")
print(
    "Training torque range: "
    f"{artifact['training_torque_min_nm']} to "
    f"{artifact['training_torque_max_nm']} N·m"
)
print("\nReminder: training fit is not independent validation.")
