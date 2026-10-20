
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

MODEL_FILE = (
    ROOT / "outputs" / "models" / "final_candidate"
    / "pmsm_yoke_torque_model.joblib"
)

FEATURE_FILE = (
    ROOT / "outputs" / "features"
    / "unknown_torque_wavelet_features.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "models" / "final_candidate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


artifact = joblib.load(MODEL_FILE)
df = pd.read_csv(FEATURE_FILE)

features = artifact["selected_features"]
all_features = artifact["all_feature_names"]
imputer = artifact["imputer"]
model = artifact["model"]

# Identify a recording identifier column.
possible_ids = [
    "recording_id", "recording", "file_name",
    "filename", "source_file", "recording_name"
]

id_col = next((c for c in possible_ids if c in df.columns), None)

if id_col is None:
    raise ValueError(
        "Could not identify the recording ID column. "
        f"Available columns: {df.columns.tolist()}"
    )

missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(
        "Unknown feature file is missing model features: "
        + ", ".join(missing)
    )

# Keep only rows with usable selected features.
X = df[features].replace([np.inf, -np.inf], np.nan)

# Ensure each row has at least one observed selected feature.
usable = X.notna().any(axis=1)
df = df.loc[usable].copy()
X = X.loc[usable]

if df.empty:
    raise ValueError("No usable feature rows found.")

# Imputer and model are those fitted on labelled recordings only.
X_imp = imputer.transform(
    df[all_features].replace([np.inf, -np.inf], np.nan)
)
feature_indices = [all_features.index(f) for f in features]
predictions = model.predict(X_imp[:, feature_indices])

df["predicted_torque_nm"] = predictions

train_min = artifact["training_torque_min_nm"]
train_max = artifact["training_torque_max_nm"]

# A tree model's output is generally bounded by its training targets.
df["outside_training_torque_range"] = (
    (df["predicted_torque_nm"] < train_min)
    | (df["predicted_torque_nm"] > train_max)
)

df.to_csv(
    OUTPUT_DIR / "unknown_window_predictions.csv",
    index=False
)

# Aggregate window predictions for each recording.
summary = (
    df.groupby(id_col)
    .agg(
        windows=("predicted_torque_nm", "size"),
        estimated_torque_mean_nm=("predicted_torque_nm", "mean"),
        estimated_torque_median_nm=("predicted_torque_nm", "median"),
        prediction_std_nm=("predicted_torque_nm", "std"),
        min_window_prediction_nm=("predicted_torque_nm", "min"),
        max_window_prediction_nm=("predicted_torque_nm", "max"),
    )
    .reset_index()
)

summary["training_torque_min_nm"] = train_min
summary["training_torque_max_nm"] = train_max
summary["estimate_status"] = (
    "Experimental estimate; not independently validated"
)

summary.to_csv(
    OUTPUT_DIR / "unknown_recording_torque_estimates.csv",
    index=False
)

print("\nUNKNOWN RECORDING ESTIMATES")
print(summary.round(3).to_string(index=False))

print("\nSaved:")
print(OUTPUT_DIR / "unknown_window_predictions.csv")
print(OUTPUT_DIR / "unknown_recording_torque_estimates.csv")
