
from pathlib import Path
import numpy as np
import pandas as pd
import pywt
from scipy.stats import kurtosis, skew

# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path("C:/Users/DELL/Documents/PMSM")
DATA_DIR = PROJECT_DIR / "data"
FEATURES_DIR = PROJECT_DIR / "outputs" / "features"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = DATA_DIR / "window_metadata.csv"
SIGNALS_FILE = DATA_DIR / "segmented_signals.npz"

# ============================================================
# 2. LOAD SEGMENTED DATA
# ============================================================

metadata = pd.read_csv(METADATA_FILE)

with np.load(SIGNALS_FILE) as saved:
    end_ring = saved["end_ring"]
    shaft = saved["shaft"]
    yoke = saved["yoke"]

signals = {
    "end_ring": end_ring,
    "shaft": shaft,
    "yoke": yoke
}

# Validate array sizes and metadata alignment
n_windows = len(metadata)

for mic_name, arr in signals.items():
    if arr.ndim != 2:
        raise ValueError(f"{mic_name} must be a 2D array.")
    if arr.shape[0] != n_windows:
        raise ValueError(
            f"{mic_name} window count does not match metadata."
        )

print("Windows loaded:", n_windows)
print("Samples per window:", end_ring.shape[1])
print("Microphones:", list(signals))

# ============================================================
# 3. FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_wavelet_features(signal, wavelet="db4", level=4):
    signal = np.asarray(signal, dtype=float)

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or infinite values.")

    coeffs = pywt.wavedec(
        signal,
        wavelet,
        level=level,
        mode="symmetric"
    )

    # PyWavelets order: A4, D4, D3, D2, D1
    band_names = [
        f"A{level}"
    ] + [
        f"D{i}" for i in range(level, 0, -1)
    ]

    energies = np.array(
        [np.sum(c ** 2) for c in coeffs],
        dtype=float
    )
    total_energy = energies.sum() + 1e-12

    features = {}

    for band_name, coeff, energy in zip(
        band_names, coeffs, energies
    ):
        coeff = np.asarray(coeff, dtype=float)
        power = coeff ** 2

        probabilities = power / (power.sum() + 1e-12)
        probabilities = probabilities[probabilities > 0]

        features[f"{band_name}_rms"] = np.sqrt(
            np.mean(coeff ** 2)
        )
        features[f"{band_name}_std"] = np.std(coeff)
        features[f"{band_name}_energy"] = energy
        features[f"{band_name}_relative_energy"] = (
            energy / total_energy
        )
        features[f"{band_name}_kurtosis"] = (
            kurtosis(coeff, fisher=True, bias=False)
            if len(coeff) > 3 else 0.0
        )
        features[f"{band_name}_skewness"] = (
            skew(coeff, bias=False)
            if len(coeff) > 2 else 0.0
        )
        features[f"{band_name}_entropy"] = -np.sum(
            probabilities * np.log2(probabilities)
        )

    return features

# ============================================================
# 4. EXTRACT FEATURES FROM ALL WINDOWS AND MICROPHONES
# ============================================================

feature_rows = []

for window_index in range(n_windows):
    row = metadata.iloc[window_index].to_dict()

    for mic_name, arr in signals.items():
        features = extract_wavelet_features(
            arr[window_index],
            wavelet="db4",
            level=4
        )

        # Prefix names to distinguish microphone positions.
        for feature_name, value in features.items():
            row[f"{mic_name}_{feature_name}"] = value

    feature_rows.append(row)

    if (window_index + 1) % 25 == 0 or (
        window_index + 1 == n_windows
    ):
        print(
            f"Processed {window_index + 1}/{n_windows} windows"
        )

features_df = pd.DataFrame(feature_rows)

# ============================================================
# 5. VALIDATE AND SAVE
# ============================================================

metadata_columns = [
    "file_name",
    "run_id",
    "section_id",
    "window_id",
    "start_index",
    "end_index_exclusive",
    "start_time_s",
    "end_time_s",
    "torque_nm",
    "speed_rpm",
    "fault"
]

feature_columns = [
    col for col in features_df.columns
    if col not in metadata_columns
]

if len(feature_columns) != 105:
    raise ValueError(
        f"Expected 105 features, got {len(feature_columns)}."
    )

if not np.isfinite(
    features_df[feature_columns].to_numpy(dtype=float)
).all():
    raise ValueError(
        "Non-finite feature values found. Inspect the signals."
    )

output_file = (
    FEATURES_DIR / "sound_1p5Nm_3000rpm_wavelet_features.csv"
)
features_df.to_csv(output_file, index=False)

print("\n===== FEATURE EXTRACTION COMPLETE =====")
print("Windows:", len(features_df))
print("Features per window:", len(feature_columns))
print("Total feature columns:", len(features_df.columns))
print("Saved to:", output_file)
print("\nPreview:")
print(features_df[metadata_columns + feature_columns[:6]].head())
