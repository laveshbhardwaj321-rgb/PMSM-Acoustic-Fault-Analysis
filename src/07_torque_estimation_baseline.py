"""Preliminary acoustic-only PMSM torque estimation using grouped CV.

Input: outputs/features/all_recordings_wavelet_features.csv
Outputs: outputs/models/torque_cv_metrics.csv,
         torque_cv_predictions.csv, torque_cv_by_recording.csv,
         random_forest_feature_importance.csv
Dependencies: pandas numpy scikit-learn
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

PROJECT_DIR = Path(r"C:\Users\DELL\Documents\PMSM")
INPUT_CSV = PROJECT_DIR / "outputs" / "features" / "all_recordings_wavelet_features.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "models"
META_COLS = ["file_name", "relative_path", "run_id", "window_id", "section_id", "start_time_s", "end_time_s", "torque_nm", "speed_rpm", "fault"]

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Cannot find input CSV: {INPUT_CSV}\nCopy your combined CSV there or edit INPUT_CSV in this script.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    missing = set(META_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Required metadata columns missing: {sorted(missing)}")
    feature_cols = [c for c in df.columns if c not in META_COLS]
    if len(feature_cols) != 105:
        raise ValueError(f"Expected 105 acoustic features, found {len(feature_cols)}")
    if df.run_id.nunique() < 3:
        raise ValueError("At least 3 independent recordings are needed for grouped CV")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if X.isna().sum().sum():
        print("Warning: missing/non-finite feature values found; imputing column medians.")
        X = X.fillna(X.median())
    y = pd.to_numeric(df.torque_nm, errors="raise")
    groups = df.run_id
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=250, min_samples_leaf=3, max_features=0.7, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=250, min_samples_leaf=3, max_features=0.7, random_state=42, n_jobs=-1),
        "MedianBaseline": DummyRegressor(strategy="median"),
    }
    logo = LeaveOneGroupOut()
    predictions, metrics = [], []
    for name, model in models.items():
        pred = np.full(len(df), np.nan)
        for tr, te in logo.split(X, y, groups):
            model.fit(X.iloc[tr], y.iloc[tr])
            pred[te] = model.predict(X.iloc[te])
        metrics.append({"model": name, "validation": "Leave-One-Recording-Out", "recordings": int(df.run_id.nunique()), "windows": len(df), "MAE_Nm": mean_absolute_error(y, pred), "RMSE_Nm": mean_squared_error(y, pred) ** 0.5, "R2": r2_score(y, pred)})
        out = df[META_COLS].copy()
        out["model"] = name
        out["actual_torque_nm"] = y
        out["predicted_torque_nm"] = pred
        out["absolute_error_nm"] = np.abs(y - pred)
        predictions.append(out)
    metrics_df = pd.DataFrame(metrics)
    pred_df = pd.concat(predictions, ignore_index=True)
    by_run = pred_df.groupby(["model", "run_id", "file_name", "torque_nm", "speed_rpm"], as_index=False).agg(windows=("absolute_error_nm", "size"), mean_prediction_nm=("predicted_torque_nm", "mean"), MAE_Nm=("absolute_error_nm", "mean")).rename(columns={"torque_nm": "actual_torque_nm"})
    metrics_df.to_csv(OUT_DIR / "torque_cv_metrics.csv", index=False)
    pred_df.to_csv(OUT_DIR / "torque_cv_predictions.csv", index=False)
    by_run.to_csv(OUT_DIR / "torque_cv_by_recording.csv", index=False)
    # Importance is descriptive only; fit on all data, not an unbiased test score.
    rf = RandomForestRegressor(n_estimators=250, min_samples_leaf=3, max_features=0.7, random_state=42, n_jobs=-1).fit(X, y)
    imp = pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    imp.to_csv(OUT_DIR / "random_forest_feature_importance.csv", index=False)
    print("\n=== GROUPED CROSS-VALIDATION ===")
    print(metrics_df.round(4).to_string(index=False))
    print("\n=== RANDOM FOREST BY RECORDING ===")
    print(by_run[by_run.model == "RandomForest"].drop(columns="model").round(3).to_string(index=False))
    print("\nTop 15 exploratory feature importances:")
    print(imp.head(15).to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")
    print("CAUTION: only 10 independent recordings; torque and speed are not fully crossed. These are preliminary metrics, not final validated accuracy. This dataset cannot classify healthy vs faulty conditions because all rows are labelled rotor_fault.")

if __name__ == "__main__":
    main()
