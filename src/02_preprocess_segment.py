
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path("C:/Users/DELL/Documents/PMSM")

# Original LVM recording
LVM_FILE = Path("C:/Users/DELL/Downloads/sound1.5nm(3000).lvm")

# Output folders
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
FEATURES_DIR = OUTPUT_DIR / "features"
MODELS_DIR = OUTPUT_DIR / "models"

# Create folders automatically
for folder in [DATA_DIR, OUTPUT_DIR, PLOTS_DIR,
               FEATURES_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Recording information
FS = 25600
WINDOW_SECONDS = 0.5
WINDOW_SIZE = int(FS * WINDOW_SECONDS)  # 12,800 samples

TORQUE_NM = 1.5
SPEED_RPM = 3000
FAULT_LABEL = "fault"  # Change if this recording is a fault test
RUN_ID = LVM_FILE.stem

# Detect gaps much larger than the nominal sample interval
GAP_THRESHOLD_SECONDS = 0.01

# Confirmed microphone order
CHANNEL_NAMES = {
    "end_ring": "v0",
    "shaft": "v1",
    "yoke": "v2",
}


# ============================================================
# 2. LOAD LVM DATA
# ============================================================

def load_lvm(path):
    if not path.exists():
        raise FileNotFoundError(
            f"LVM file not found: {path}\n"
            "Check the LVM_FILE path."
        )

    columns = ["t0", "v0", "t1", "v1", "t2", "v2"]

    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=23,
        header=None,
        usecols=range(6),
        names=columns,
        low_memory=False
    )

    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=columns).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No numeric samples found. Check the LVM header."
        )

    # Verify that all microphone channels share the same time base
    if not (
        np.allclose(df["t0"], df["t1"], atol=1e-6)
        and np.allclose(df["t0"], df["t2"], atol=1e-6)
    ):
        raise ValueError(
            "Microphone timestamps do not match."
        )

    time_s = df["t0"].to_numpy()

    channels = {
        "end_ring": df["v0"].to_numpy(dtype=float),
        "shaft": df["v1"].to_numpy(dtype=float),
        "yoke": df["v2"].to_numpy(dtype=float)
    }

    return time_s, channels


# ============================================================
# 3. DETECT ACQUISITION GAPS
# ============================================================

def find_sections(time_s):
    dt = np.diff(time_s)

    if np.any(dt <= 0):
        raise ValueError(
            "Timestamps are not strictly increasing."
        )

    gap_indices = np.where(
        dt > GAP_THRESHOLD_SECONDS
    )[0]

    section_starts = np.r_[0, gap_indices + 1]
    section_ends = np.r_[gap_indices + 1, len(time_s)]

    sections = []

    for section_id, (start, end) in enumerate(
        zip(section_starts, section_ends)
    ):
        sections.append({
            "section_id": section_id,
            "start_index": int(start),
            "end_index": int(end),
            "sample_count": int(end - start),
            "start_time_s": float(time_s[start]),
            "end_time_s": float(time_s[end - 1])
        })

    return dt, gap_indices, sections


# ============================================================
# 4. SEGMENT THE THREE MICROPHONE SIGNALS
# ============================================================

def segment_recording(time_s, channels, sections):
    windows = []

    for section in sections:
        start = section["start_index"]
        end = section["end_index"]
        section_length = end - start

        # Only full windows; never cross acquisition gaps
        number_of_windows = section_length // WINDOW_SIZE

        for local_id in range(number_of_windows):
            a = start + local_id * WINDOW_SIZE
            b = a + WINDOW_SIZE

            row = {
                "file_name": LVM_FILE.name,
                "run_id": RUN_ID,
                "section_id": section["section_id"],
                "window_id": len(windows),
                "start_index": a,
                "end_index_exclusive": b,
                "start_time_s": float(time_s[a]),
                "end_time_s": float(time_s[b - 1]),
                "torque_nm": TORQUE_NM,
                "speed_rpm": SPEED_RPM,
                "fault": FAULT_LABEL
            }

            for mic_name, signal in channels.items():
                row[f"{mic_name}_signal"] = signal[a:b].copy()

            windows.append(row)

    return windows


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def main():
    print("Loading LVM file...")
    time_s, channels = load_lvm(LVM_FILE)

    dt, gap_indices, sections = find_sections(time_s)

    print("\n===== RECORDING CHECK =====")
    print("File:", LVM_FILE.name)
    print("Number of samples:", len(time_s))
    print("Nominal sample rate:", FS, "Hz")
    print("Median timestamp interval:", np.median(dt), "seconds")
    print("Rate estimated from timestamps:",
          round(1 / np.median(dt), 3), "Hz")
    print("Largest timestamp interval:", np.max(dt), "seconds")
    print("Detected gaps:", len(gap_indices))
    print("Continuous sections:", len(sections))

    print("\n===== FIRST 12 SECTIONS =====")
    section_df = pd.DataFrame(sections)

    print(section_df[
        ["section_id", "sample_count",
         "start_time_s", "end_time_s"]
    ].head(12).to_string(index=False))

    # Save the acquisition section report
    section_path = DATA_DIR / "section_report.csv"
    section_df.to_csv(section_path, index=False)

    print("\nSegmenting all three microphones...")
    windows = segment_recording(time_s, channels, sections)

    print("Window duration:", WINDOW_SECONDS, "seconds")
    print("Samples per window:", WINDOW_SIZE)
    print("Valid windows:", len(windows))

    if not windows:
        raise ValueError(
            "No complete windows found. Inspect section lengths."
        )

    # Save metadata separately from the raw signal arrays
    metadata_rows = []

    for row in windows:
        metadata_rows.append({
            key: value
            for key, value in row.items()
            if not key.endswith("_signal")
        })

    metadata_df = pd.DataFrame(metadata_rows)

    metadata_path = DATA_DIR / "window_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    # Save aligned microphone windows
    signals_path = DATA_DIR / "segmented_signals.npz"

    np.savez_compressed(
        signals_path,
        end_ring=np.stack([
            row["end_ring_signal"] for row in windows
        ]),
        shaft=np.stack([
            row["shaft_signal"] for row in windows
        ]),
        yoke=np.stack([
            row["yoke_signal"] for row in windows
        ]),
        fs=np.array(FS),
        window_seconds=np.array(WINDOW_SECONDS)
    )

    print("\n===== FILES SAVED =====")
    print("Section report:", section_path)
    print("Window metadata:", metadata_path)
    print("Segmented signals:", signals_path)

    print("\n===== FIRST FIVE WINDOWS =====")
    print(metadata_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
