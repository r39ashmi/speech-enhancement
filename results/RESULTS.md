# Loss-function evaluation results

Seeds per loss: {'R0': np.int64(3), 'R1': np.int64(3), 'R2': np.int64(3), 'R3': np.int64(3), 'R4': np.int64(3), 'R5': np.int64(3), 'R6': np.int64(3), 'X-C0.5': np.int64(3), 'X-C0.7': np.int64(3), 'X-LIN': np.int64(3)}. Values are mean ± std **across seeds** (each seed's value is the mean over 60 test utterances).

Unprocessed input: -5dB dry: PESQ 1.23, STOI 0.735; 5dB dry: PESQ 1.34, STOI 0.852; 15dB dry: PESQ 2.10, STOI 0.948; 5dB reverb: PESQ 1.30, STOI 0.826


## PESQ (wideband, higher better)

| Loss | Description | -5dB dry | 5dB dry | 15dB dry | 5dB reverb |
|---|---|---|---|---|---|
| X-LIN | linear magnitude MSE | 1.21 ± 0.08 | 1.29 ± 0.12 | 1.67 ± 0.36 | 1.26 ± 0.09 |
| X-C0.7 | compressed mag c=0.7 | 1.24 ± 0.13 | 1.34 ± 0.17 | 1.75 ± 0.50 | 1.30 ± 0.11 |
| X-C0.5 | compressed mag c=0.5 | 1.29 ± 0.08 | 1.41 ± 0.11 | 1.93 ± 0.35 | 1.36 ± 0.06 |
| R0 | compressed mag c=0.3 | 1.23 ± 0.08 | 1.32 ± 0.12 | 1.67 ± 0.36 | 1.30 ± 0.09 |
| R1 | R0 + compressed complex | 1.39 ± 0.04 | 1.53 ± 0.01 | 2.22 ± 0.01 | 1.42 ± 0.01 |
| R2 | R1 + multi-res (5/10/20/40ms) | 1.45 ± 0.02 | 1.59 ± 0.01 | 2.31 ± 0.01 | 1.47 ± 0.00 |
| R3 | R2 + over-suppression pen. | **1.45 ± 0.01** | **1.60 ± 0.02** | 2.35 ± 0.01 | 1.48 ± 0.01 |
| R4 | R3 + loudness (A-wt) weighting | 1.44 ± 0.03 | 1.60 ± 0.03 | **2.37 ± 0.00** | **1.49 ± 0.00** |
| R5 | R3 + SI-SNR | 1.34 ± 0.19 | 1.47 ± 0.20 | 2.06 ± 0.54 | 1.39 ± 0.15 |
| R6 | R3 + loudness + SI-SNR | 1.31 ± 0.15 | 1.52 ± 0.09 | 2.04 ± 0.31 | 1.43 ± 0.07 |

## STOI (higher better)

| Loss | Description | -5dB dry | 5dB dry | 15dB dry | 5dB reverb |
|---|---|---|---|---|---|
| X-LIN | linear magnitude MSE | 0.715 ± 0.040 | 0.812 ± 0.047 | 0.881 ± 0.056 | 0.784 ± 0.049 |
| X-C0.7 | compressed mag c=0.7 | 0.721 ± 0.034 | 0.817 ± 0.045 | 0.885 ± 0.053 | 0.790 ± 0.044 |
| X-C0.5 | compressed mag c=0.5 | 0.733 ± 0.024 | 0.831 ± 0.031 | 0.902 ± 0.038 | 0.804 ± 0.030 |
| R0 | compressed mag c=0.3 | 0.715 ± 0.029 | 0.810 ± 0.039 | 0.881 ± 0.046 | 0.783 ± 0.038 |
| R1 | R0 + compressed complex | 0.723 ± 0.005 | 0.831 ± 0.005 | 0.919 ± 0.004 | 0.804 ± 0.006 |
| R2 | R1 + multi-res (5/10/20/40ms) | 0.742 ± 0.001 | 0.846 ± 0.001 | 0.927 ± 0.001 | 0.821 ± 0.001 |
| R3 | R2 + over-suppression pen. | **0.755 ± 0.006** | **0.855 ± 0.002** | **0.936 ± 0.001** | **0.831 ± 0.003** |
| R4 | R3 + loudness (A-wt) weighting | 0.753 ± 0.004 | 0.853 ± 0.002 | 0.934 ± 0.002 | 0.828 ± 0.003 |
| R5 | R3 + SI-SNR | 0.731 ± 0.052 | 0.835 ± 0.048 | 0.916 ± 0.047 | 0.811 ± 0.048 |
| R6 | R3 + loudness + SI-SNR | 0.740 ± 0.019 | 0.845 ± 0.015 | 0.925 ± 0.016 | 0.819 ± 0.018 |

## SI-SDR improvement dB (higher better)

| Loss | Description | -5dB dry | 5dB dry | 15dB dry | 5dB reverb |
|---|---|---|---|---|---|
| X-LIN | linear magnitude MSE | -4.95 ± 5.30 | -11.71 ± 6.06 | -22.79 ± 6.45 | -11.09 ± 5.99 |
| X-C0.7 | compressed mag c=0.7 | -18.48 ± 2.20 | -25.41 ± 0.86 | -37.41 ± 1.32 | -25.26 ± 1.35 |
| X-C0.5 | compressed mag c=0.5 | -14.80 ± 5.04 | -22.54 ± 5.21 | -33.88 ± 5.56 | -21.70 ± 5.07 |
| R0 | compressed mag c=0.3 | -14.22 ± 5.45 | -20.93 ± 4.89 | -32.77 ± 4.85 | -20.05 ± 5.20 |
| R1 | R0 + compressed complex | 6.16 ± 0.72 | 2.88 ± 0.58 | -4.63 ± 0.14 | 2.30 ± 0.40 |
| R2 | R1 + multi-res (5/10/20/40ms) | **6.80 ± 0.26** | 3.78 ± 0.16 | -3.58 ± 0.20 | 3.03 ± 0.05 |
| R3 | R2 + over-suppression pen. | 6.77 ± 0.20 | **4.18 ± 0.08** | **-2.57 ± 0.36** | **3.19 ± 0.13** |
| R4 | R3 + loudness (A-wt) weighting | 6.60 ± 0.31 | 3.87 ± 0.12 | -2.95 ± 0.31 | 3.09 ± 0.26 |
| R5 | R3 + SI-SNR | 5.09 ± 3.30 | 2.66 ± 3.71 | -4.07 ± 7.14 | 1.91 ± 2.87 |
| R6 | R3 + loudness + SI-SNR | 5.14 ± 2.40 | 1.82 ± 2.94 | -6.93 ± 5.82 | 1.36 ± 2.08 |

## Low-energy log-spectral distortion dB (LOWER better)

| Loss | Description | -5dB dry | 5dB dry | 15dB dry | 5dB reverb |
|---|---|---|---|---|---|
| X-LIN | linear magnitude MSE | 13.56 ± 0.11 | 11.66 ± 0.06 | 9.22 ± 0.86 | 10.91 ± 0.24 |
| X-C0.7 | compressed mag c=0.7 | 12.67 ± 0.17 | 10.74 ± 0.29 | 8.66 ± 0.81 | 10.36 ± 0.29 |
| X-C0.5 | compressed mag c=0.5 | 11.99 ± 0.08 | 10.27 ± 0.29 | 8.18 ± 0.74 | **10.05 ± 0.35** |
| R0 | compressed mag c=0.3 | 11.49 ± 0.19 | 10.09 ± 0.25 | 8.52 ± 0.78 | 10.16 ± 0.27 |
| R1 | R0 + compressed complex | 13.54 ± 0.88 | 13.44 ± 1.10 | 11.54 ± 0.65 | 14.63 ± 1.15 |
| R2 | R1 + multi-res (5/10/20/40ms) | 11.59 ± 0.06 | 10.89 ± 0.22 | 9.42 ± 0.32 | 11.64 ± 0.06 |
| R3 | R2 + over-suppression pen. | **10.64 ± 0.04** | **9.63 ± 0.23** | 7.83 ± 0.13 | 10.12 ± 0.15 |
| R4 | R3 + loudness (A-wt) weighting | 10.69 ± 0.16 | 9.63 ± 0.18 | **7.79 ± 0.11** | 10.14 ± 0.03 |
| R5 | R3 + SI-SNR | 11.43 ± 0.45 | 10.16 ± 0.39 | 8.44 ± 0.47 | 10.18 ± 0.11 |
| R6 | R3 + loudness + SI-SNR | 11.20 ± 0.39 | 10.07 ± 0.37 | 8.52 ± 0.52 | 10.14 ± 0.09 |

## Difference vs. R0 (compressed magnitude), in units of seed-to-seed std

A difference is only treated as real if |Δ| > 2× the larger of the two seed stds (rough screen; n=3 seeds).

| Loss | -5dB dry ΔPESQ | 5dB dry ΔPESQ | 15dB dry ΔPESQ | 5dB reverb ΔPESQ | -5dB dry ΔSI-SDRi | 5dB dry ΔSI-SDRi | 15dB dry ΔSI-SDRi | 5dB reverb ΔSI-SDRi |
|---|---|---|---|---|---|---|---|---|
| X-LIN | -0.02 | -0.03 | -0.01 | -0.03 | +9.27 | +9.22 | +9.97 | +8.97 |
| X-C0.7 | +0.01 | +0.02 | +0.07 | -0.00 | -4.26 | -4.48 | -4.65 | -5.21 |
| X-C0.5 | +0.06 | +0.09 | +0.25 | +0.06 | -0.58 | -1.61 | -1.11 | -1.65 |
| R1 | +0.17 * | +0.21 | +0.54 | +0.12 | +20.38 * | +23.81 * | +28.14 * | +22.35 * |
| R2 | +0.22 * | +0.27 * | +0.64 | +0.17 | +21.03 * | +24.71 * | +29.18 * | +23.08 * |
| R3 | +0.22 * | +0.28 * | +0.68 | +0.18 | +20.99 * | +25.12 * | +30.20 * | +23.24 * |
| R4 | +0.22 * | +0.28 * | +0.70 | +0.19 * | +20.82 * | +24.81 * | +29.81 * | +23.14 * |
| R5 | +0.12 | +0.15 | +0.39 | +0.09 | +19.32 * | +23.60 * | +28.69 * | +21.97 * |
| R6 | +0.09 | +0.20 | +0.37 | +0.13 | +19.36 * | +22.76 * | +25.83 * | +21.42 * |

`*` = exceeds 2× seed std.