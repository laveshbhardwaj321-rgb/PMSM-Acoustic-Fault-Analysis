"""
11_microphone_ablation.py

Compare torque regression using:
  1) End-ring microphone features only
  2) Shaft microphone features only
  3) Yoke microphone features only
  4) All three microphones

Uses Leave-One-Recording-Out validation to keep windows from the same recording
together in a held-out fold. This is a preliminary comparison: only 10 independent
labelled recordings are currently available.

Input:
  C:\\Users\\DELL\\Documents\\PMSM\\outputs\\features\\all_recordings_wavelet_features.csv

Outputs:
  outputs\\models\\microphone_ablation\\microphone_ablation_metrics.csv
  outputs\\models\\microphone_ablation\\microphone_ablation_predictions.csv
  outputs\\models\\microphone_ablation\\recording_level_errors.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(r"C:\Users\DELL\Documents\PMSM")
INPUT_CSV = PROJECT_DIR / "outputs" / "features" / "all_recordings_wavelet_features.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "models" / "microphone_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = [
    "file_name", "relative_path", "run_id", "window_id", "section_id",
    "start_time_s", "end_time_s", "torque_nm", "speed_rpm", "fault"
]
MICROPHONES = {
    "End-ring only": "end_ring_",
    "Shaft only": "shaft_",
    "Yoke only": "yoke_",
    "All microphones": None,
}

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}\n"
            "Copy all_recordings_wavelet_features.csv into outputs\\features."
        )

    df = pd.read_csv(INPUT_CSV)
    missing = set(META_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

    all_features = [c for c in df.columns if c not in META_COLS]
    if len(all_features) != 105:
        raise ValueError(f"Expected 105 acoustic features; found {len(all_features)}.")

    # Confirm that each channel has the expected 35 features.
    for prefix in ("end_ring_", "shaft_", "yoke_"):
        count = sum(c.startswith(prefix) for c in all_features)
        if count != 35:
            raise ValueError(f"Expected 35 features for {prefix}; found {count}.")

    y = pd.to_numeric(df["torque_nm"], errors="raise")
    groups = df["run_id"].astype(str)
    if groups.nunique() < 3:
        raise ValueError("Need at least 3 independent recordings.")

    # Use the same model and validation splits for all microphone configurations.
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.7,
            random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=3, max_features=0.7,
            random_state=42, n_jobs=-1
        ),
    }

    logo = LeaveOneGroupOut()
    predictions = []
    metrics = []

    for mic_name, prefix in MICROPHONES.items():
        cols = all_features if prefix is None else [c for c in all_features if c.startswith(prefix)]
        X = df[cols].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        for model_name, estimator in models.items():
            pred = np.full(len(df), np.nan, dtype=float)

            for train_idx, test_idx in logo.split(X, y, groups):
                pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", clone(estimator))
                ])
                pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
                pred[test_idx] = pipeline.predict(X.iloc[test_idx])

            metrics.append({
                "microphone_configuration": mic_name,
                "model": model_name,
                "feature_count": len(cols),
                "validation": "Leave-One-Recording-Out",
                "recordings": int(groups.nunique()),
                "windows": int(len(df)),
                "MAE_Nm": mean_absolute_error(y, pred),
                "RMSE_Nm": mean_squared_error(y, pred) ** 0.5,
                "R2": r2_score(y, pred),
            })

            p = df[META_COLS].copy()
            p["microphone_configuration"] = mic_name
            p["model"] = model_name
            p["actual_torque_nm"] = y
            p["predicted_torque_nm"] = pred
            p["absolute_error_nm"] = np.abs(y.to_numpy() - pred)
            predictions.append(p)

    metrics_df = pd.DataFrame(metrics).sort_values(["MAE_Nm", "RMSE_Nm"])
    pred_df = pd.concat(predictions, ignore_index=True)
    recording_errors = (
        pred_df.groupby(
            ["microphone_configuration", "model", "run_id", "file_name",
             "torque_nm", "speed_rpm"], as_index=False
        )
        .agg(
            windows=("absolute_error_nm", "size"),
            mean_prediction_nm=("predicted_torque_nm", "mean"),
            MAE_Nm=("absolute_error_nm", "mean"),
            RMSE_Nm=("absolute_error_nm", lambda s: float(np.sqrt(np.mean(np.square(s)))))
        )
        .rename(columns={"torque_nm": "actual_torque_nm"})
    )

    metrics_df.to_csv(OUT_DIR / "microphone_ablation_metrics.csv", index=False)
    pred_df.to_csv(OUT_DIR / "microphone_ablation_predictions.csv", index=False)
    recording_errors.to_csv(OUT_DIR / "recording_level_errors.csv", index=False)

    print("\n=== MICROPHONE ABLATION: GROUPED CROSS-VALIDATION ===")
    print(metrics_df.round(4).to_string(index=False))
    print("\n=== RECORDING-LEVEL ERRORS FOR THE BEST-MAE CONFIGURATION ===")
    best = metrics_df.iloc[0]
    best_rows = recording_errors[
        (recording_errors["microphone_configuration"] == best["microphone_configuration"])
        & (recording_errors["model"] == best["model"])
    ]
    print(best_rows.drop(columns=["microphone_configuration", "model"]).round(3).to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")
    print(
        "\nInterpretation note: these are exploratory comparisons using 10 independent "
        "recordings. The best configuration may change with more labelled runs; do not "
        "treat the best score as proof of generalization to unseen torque/speed ranges."
    )

if __name__ == "__main__":
    main()
