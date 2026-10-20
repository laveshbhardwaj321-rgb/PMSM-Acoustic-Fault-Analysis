# PMSM-Acoustic-Fault-Analysis
# Fault Analysis and Torque Estimation of PMSM Using Acoustic Signals

## Overview

This project investigates acoustic signal processing and machine learning for the analysis of a **Permanent Magnet Synchronous Motor (PMSM)** used in electric vehicle (EV) applications. It explores whether acoustic signals captured at different motor locations can be used to extract meaningful features for preliminary torque estimation.

The project uses multi-microphone acoustic recordings, wavelet-based feature extraction, and regression models to analyze the relationship between acoustic characteristics and motor torque.

## Objectives

- Analyze acoustic signals collected from different locations on a PMSM.
- Preprocess and segment recorded signals into fixed-duration windows.
- Apply Discrete Wavelet Transform (DWT) for multiresolution signal analysis.
- Extract statistical features from wavelet coefficients.
- Compare machine learning regression models for torque estimation.
- Evaluate the contribution of individual microphone positions.
- Investigate the limitations of acoustic-only torque estimation.

## System Overview

Three microphone positions are considered:

- **End ring**
- **Shaft**
- **Yoke**

The recorded acoustic signals are processed to extract features that serve as inputs to machine learning models.

## Methodology

### 1. Signal Preprocessing

Acoustic recordings stored in LabVIEW Measurement (LVM) files are loaded and checked for timestamp discontinuities. Continuous signal sections are segmented into non-overlapping windows of 0.5 seconds.

- Nominal sampling frequency: 25.6 kHz
- Window duration: 0.5 seconds
- Samples per nominal window: 12,800

### 2. Wavelet Decomposition

A level-4 Discrete Wavelet Transform using the **Daubechies-4 (db4)** wavelet is applied to each signal window.

The decomposition produces five coefficient bands:

- A4 — approximation coefficients
- D4 — detail coefficients
- D3 — detail coefficients
- D2 — detail coefficients
- D1 — detail coefficients

### 3. Feature Extraction

Seven statistical descriptors are calculated for each wavelet band:

- Root Mean Square (RMS)
- Standard deviation
- Signal energy
- Relative energy
- Kurtosis
- Skewness
- Entropy

For each microphone, this produces 35 features per window, resulting in **105 acoustic features per window** across all three microphone positions.

### 4. Machine Learning

Regression models are evaluated to estimate motor torque from the extracted acoustic features. The evaluated approaches include:

- Random Forest Regression
- Extra Trees Regression
- Ridge Regression
- Linear Regression
- Partial Least Squares (PLS) Regression

**Leave-One-Recording-Out cross-validation** is used to evaluate generalization to recordings excluded from model training. All windows from a held-out recording are kept together in the test fold to reduce data leakage between correlated windows.

### 5. Microphone Ablation and Feature Selection

Separate experiments compare the end-ring, shaft, and yoke microphone configurations against the combination of all three microphones.

The yoke-only configuration and a top-10-feature selection approach are investigated as candidates for a simpler acoustic sensing system.

## Preliminary Results

The current labelled dataset contains 10 independent rotor-fault recordings, with 2,210 segmented windows and torque labels spanning 1.5–18.2 N·m.

| Experiment | Preliminary result |
|---|---|
| Acoustic feature extraction | 105 features per window |
| Model comparison | Extra Trees achieved MAE of approximately 1.72 N·m in the 105-feature comparison |
| Microphone ablation | Yoke-only Random Forest achieved MAE of approximately 1.61 N·m |
| Yoke feature selection | Top-10 features achieved window-level MAE of approximately 1.58 N·m |
| Unknown-torque recordings | Preliminary estimates generated for four recordings |

These results are preliminary and correspond to different model configurations and evaluation experiments. Their metrics should not be interpreted as a single directly comparable model ranking.

## Limitations

- The labelled dataset contains only 10 independent recordings.
- All currently labelled recordings represent rotor-fault conditions.
- Torque and speed coverage is uneven across the available recordings.
- The 18.2 N·m recording is substantially underpredicted when held out during validation because the training data do not contain a comparable high-torque target.
- Torque estimates for additional recordings have not been verified against reference torque measurements.
- The current results do not establish validated healthy-versus-fault classification performance.

Consequently, the model should be considered a **preliminary acoustic torque-estimation approach**, not a validated replacement for a torque sensor.


## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- PyWavelets
- Scikit-learn
- Signal Processing
- Machine Learning
- Acoustic Data Analysis



## Usage

1. Place the permitted acoustic LVM recordings in the configured data directory.
2. Run the signal validation and preprocessing scripts.
3. Generate wavelet features using the feature-extraction script.
4. Build the combined labelled dataset.
5. Run the model evaluation and microphone-ablation scripts.
6. Train the candidate model and generate predictions for additional recordings.

Update any local file paths in the scripts before running them. Results depend on the input recordings, preprocessing configuration, and model settings.

## Future Work

- Collect additional recordings with verified torque measurements.
- Improve torque and speed coverage using repeated measurements at matched operating conditions.
- Validate the final model on independent recording sessions.
- Include healthy motor recordings and verified fault classes.
- Investigate robustness to noise, sensor placement, and operating conditions.
- Explore CNN-based approaches after collecting a sufficiently large labelled dataset.
- Investigate real-time acoustic monitoring for EV motor condition analysis.

## Disclaimer

This repository presents an ongoing experimental study. Reported torque estimates are preliminary, and predictions for recordings without reference torque measurements are unverified. The results should not be interpreted as validated diagnostic or safety-critical performance.

## Author

**Project developed as part of an academic study on PMSM acoustic signal analysis for EV applications.**
