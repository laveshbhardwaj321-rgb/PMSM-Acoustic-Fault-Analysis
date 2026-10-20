"""
10_compare_torque_models.py
Compare acoustic-only and acoustic+RPM torque regression using the same
Leave-One-Recording-Out splits. This is a preliminary small-data evaluation.

Expected input:
C:\\Users\\DELL\\Documents\\PMSM\\outputs\\features\\all_recordings_wavelet_features.csv

Outputs in outputs\\models\\comparison:
- model_comparison_metrics.csv
- held_out_predictions.csv
- recording_level_errors.csv
- rf_feature_importance_acoustic_only.csv
- predicted_vs_actual_random_forest.png
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_DIR = Path(r"C:\Users\DELL\Documents\PMSM")
INPUT_CSV = PROJECT_DIR / "outputs" / "features" / "all_recordings_wavelet_features.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "models" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = [
    "file_name", "relative_path", "run_id", "window_id", "section_id",
    "start_time_s", "end_time_s", "torque_nm", "speed_rpm", "fault"
]

def make_models():
    # Median baseline is deliberately feature-independent.
    return {
        "Median baseline": ("acoustic", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="median"))
        ])),
        "Ridge acoustic": ("acoustic", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0))
        ])),
        "Random Forest acoustic": ("acoustic", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300, min_samples_leaf=3, max_features=0.7,
                random_state=42, n_jobs=-1
            ))
        ])),
        "Extra Trees acoustic": ("acoustic", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=300, min_samples_leaf=3, max_features=0.7,
                random_state=42, n_jobs=-1
            ))
        ])),
        "Random Forest acoustic + RPM": ("with_speed", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=300, min_samples_leaf=3, max_features=0.7,
                random_state=42, n_jobs=-1
            ))
        ])),
        "Extra Trees acoustic + RPM": ("with_speed", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=300, min_samples_leaf=3, max_features=0.7,
                random_state=42, n_jobs=-1
            ))
        ])),
    }

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}\n"
            "Copy all_recordings_wavelet_features.csv to outputs\\features."
        )

    df = pd.read_csv(INPUT_CSV)
    missing = set(META_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    if len(feature_cols) != 105:
        raise ValueError(f"Expected 105 acoustic features; found {len(feature_cols)}.")

    y = pd.to_numeric(df["torque_nm"], errors="raise")
    groups = df["run_id"].astype(str)
    X_acoustic = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_acoustic = X_acoustic.replace([np.inf, -np.inf], np.nan)
    X_speed = X_acoustic.copy()
    X_speed["speed_rpm"] = pd.to_numeric(df["speed_rpm"], errors="raise")

    if groups.nunique() < 3:
        raise ValueError("Need at least 3 independent recordings for grouped validation.")

    logo = LeaveOneGroupOut()
    prediction_rows = []
    metric_rows = []
    model_specs = make_models()

    for name, (input_kind, estimator) in model_specs.items():
        pred = np.full(len(df), np.nan, dtype=float)
        for train_idx, test_idx in logo.split(X_acoustic, y, groups):
            X = X_speed if input_kind == "with_speed" else X_acoustic
            model = clone(estimator)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred[test_idx] = model.predict(X.iloc[test_idx])

        metric_rows.append({
            "model": name,
            "input": "105 acoustic features + RPM" if input_kind == "with_speed"
                     else ("baseline; no features" if name == "Median baseline"
                           else "105 acoustic features"),
            "validation": "Leave-One-Recording-Out",
            "recordings": int(groups.nunique()),
            "windows": int(len(df)),
            "MAE_Nm": mean_absolute_error(y, pred),
            "RMSE_Nm": mean_squared_error(y, pred) ** 0.5,
            "R2": r2_score(y, pred),
        })

        temp = df[META_COLS].copy()
        temp["model"] = name
        temp["actual_torque_nm"] = y
        temp["predicted_torque_nm"] = pred
        temp["absolute_error_nm"] = np.abs(y.to_numpy() - pred)
        prediction_rows.append(temp)

    metrics = pd.DataFrame(metric_rows).sort_values("MAE_Nm")
    predictions = pd.concat(prediction_rows, ignore_index=True)
    recording_errors = (
        predictions.groupby(["model", "run_id", "file_name", "torque_nm", "speed_rpm"], as_index=False)
        .agg(
            windows=("absolute_error_nm", "size"),
            mean_prediction_nm=("predicted_torque_nm", "mean"),
            MAE_Nm=("absolute_error_nm", "mean"),
            RMSE_Nm=("absolute_error_nm", lambda x: float(np.sqrt(np.mean(np.square(x)))))
        )
        .rename(columns={"torque_nm": "actual_torque_nm"})
    )

    metrics.to_csv(OUT_DIR / "model_comparison_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / "held_out_predictions.csv", index=False)
    recording_errors.to_csv(OUT_DIR / "recording_level_errors.csv", index=False)

    # Plot Random Forest acoustic-only held-out window predictions.
    rf_name = "Random Forest acoustic"
    rf = predictions[predictions["model"] == rf_name]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(rf["actual_torque_nm"], rf["predicted_torque_nm"], alpha=0.35)
    lo = min(rf["actual_torque_nm"].min(), rf["predicted_torque_nm"].min())
    hi = max(rf["actual_torque_nm"].max(), rf["predicted_torque_nm"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Actual torque (N·m)")
    ax.set_ylabel("Predicted torque (N·m)")
    ax.set_title("Random Forest: held-out recording predictions")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "predicted_vs_actual_random_forest.png", dpi=160)
    plt.close(fig)

    # Fit acoustic-only RF on all data only for exploratory feature importance.
    # This importance is descriptive, not a validation metric.
    rf_model = clone(model_specs[rf_name][1])
    rf_model.fit(X_acoustic, y)
    fitted_rf = rf_model.named_steps["model"]
    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": fitted_rf.feature_importances_
    }).sort_values("importance", ascending=False)
    importances.to_csv(OUT_DIR / "rf_feature_importance_acoustic_only.csv", index=False)

    print("\n=== MODEL COMPARISON: GROUPED CROSS-VALIDATION ===")
    print(metrics.round(4).to_string(index=False))
    print("\n=== RANDOM FOREST ACOUSTIC-ONLY ERRORS BY RECORDING ===")
    print(
        recording_errors[recording_errors["model"] == rf_name]
        .drop(columns="model").round(3).to_string(index=False)
    )
    print("\nTop 10 exploratory acoustic-only RF feature importances:")
    print(importances.head(10).to_string(index=False))
    print(f"\nSaved outputs to: {OUT_DIR}")
    print(
        "\nINTERPRETATION NOTE: only 10 independent labelled recordings are available, "
        "and torque/speed coverage is sparse and confounded. Use these results as "
        "preliminary research results, not as proof of generalization to arbitrary loads/speeds."
    )

if __name__ == "__main__":
    main()
