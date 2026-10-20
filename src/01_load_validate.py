from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE = Path("C:/Users/DELL/Downloads/sound1.5nm(3000).lvm")

def load_lvm(path):
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

    # Confirm that all three microphones share the same time base.
    if not (
        np.allclose(df["t0"], df["t1"], atol=1e-6)
        and np.allclose(df["t0"], df["t2"], atol=1e-6)
    ):
        raise ValueError("Microphone timestamps do not match.")

    time_s = df["t0"].to_numpy()
    channels = {
        "end_ring": df["v0"].to_numpy(),
        "shaft": df["v1"].to_numpy(),
        "yoke": df["v2"].to_numpy()
    }

    return time_s, channels

time_s, channels = load_lvm(FILE)

dt = np.diff(time_s)
positive_dt = dt[dt > 0]
fs = 1 / np.median(positive_dt)

print("Number of samples:", len(time_s))
print("Estimated sample rate:", fs, "Hz")
print("Timestamp duration:", time_s[-1] - time_s[0], "seconds")
print("Channel names:", list(channels))

for name, signal in channels.items():
    print(
        name,
        "minimum =", np.min(signal),
        "maximum =", np.max(signal),
        "RMS =", np.sqrt(np.mean(signal**2))
    )

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

for ax, (name, signal) in zip(axes, channels.items()):
    ax.plot(time_s, signal, linewidth=0.4)
    ax.set_title(name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal (V)")
    ax.grid(True)

plt.tight_layout()
plt.savefig("c:/Users/DELL/Documents/PMSM/outputs/plots/raw_signals.png", dpi=200)
plt.show()


plot_dir = Path("c:/Users/DELL/Documents/PMSM/outputs/plots")
plot_dir.mkdir(parents=True, exist_ok=True)

def plot_zoomed_signals(time_s, channels, start_s, duration_s=0.1):
    mask = (
        (time_s >= start_s)
        & (time_s < start_s + duration_s)
    )

    if not mask.any():
        print("No samples found in the requested interval.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for ax, (name, signal) in zip(axes, channels.items()):
        ax.plot(time_s[mask], signal[mask], linewidth=0.7)
        ax.set_title(name)
        ax.set_ylabel("Signal (V)")
        ax.grid(True)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(plot_dir / f"zoom_{start_s:.1f}s.png", dpi=200)
    plt.show()
    
# ===== TIMESTAMP CONTINUITY CHECK =====

import numpy as np

dt = np.diff(time_s)
positive_dt = dt[dt > 0]

nominal_dt = np.median(positive_dt)
fs_estimated = 1 / nominal_dt

gap_threshold = 1.5 * nominal_dt
gap_indices = np.where(dt > gap_threshold)[0]

print("\n===== TIMESTAMP CHECK =====")
print(f"Number of samples: {len(time_s)}")
print(f"Median sample interval: {nominal_dt:.10f} seconds")
print(f"Estimated sample rate: {fs_estimated:.3f} Hz")
print(f"Largest time interval: {np.max(dt):.10f} seconds")
print(f"Number of unusually large intervals: {len(gap_indices)}")
print(f"Number of non-positive intervals: {np.sum(dt <= 0)}")

for i in gap_indices[:10]:
    print(
        f"Gap at sample {i}: "
        f"{time_s[i]:.6f} s -> {time_s[i + 1]:.6f} s "
        f"(interval = {dt[i]:.8f} s)"
    )    