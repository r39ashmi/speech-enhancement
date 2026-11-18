# Which training loss works best for a small speech-enhancement model?

Evaluation of 10 loss functions on a low-complexity causal speech-enhancement model.
Based on the loss-function ladder (Approach 6, §4.9) of *SE_Research_Report_Rashmi_Kethireddy.docx*.

---

## 1. Motive of the experiments

**Problem.** A small masking-based speech-enhancement model (about 80K parameters, built for real-time use on constrained devices) works acceptably at positive SNR but degrades at low or negative SNR, under reverberation, and by suppressing weak speech sounds.

**Why study the loss?** Of the six improvement levers in the report, the loss function is the only one that costs nothing at run time. Inference speed, model size and latency are identical whatever loss trained the model. If a better loss improves quality, it is a free upgrade.

**Why a ladder?** The report argues that each loss term should be justified by a failure it fixes, not assumed to help. So terms are added one at a time to a fixed model:

| Step | Added term | Failure it targets |
|---|---|---|
| X-LIN | none (plain linear-magnitude error) | baseline: loud sounds dominate, weak consonants cost almost nothing to erase |
| X-C0.7 / X-C0.5 / R0 | magnitude compression (exponent 0.7 / 0.5 / 0.3) | shrinks the loud-vs-quiet gap |
| R1 | compressed complex (phase-aware) error | magnitude-only ignores phase, which matters most at low SNR |
| R2 | multi-resolution spectral error (5/10/20/40 ms) | smearing in time, esp. reverberation |
| R3 | over-suppression penalty | asymmetric cost: removing speech is worse than leaving noise |
| R4 | loudness (A-weighting) frequency weighting | focus on frequencies the ear is most sensitive to |
| R5 | SI-SNR term on the waveform | direct signal-fidelity objective |
| R6 | loudness weighting + SI-SNR | combination |

R0–R6 follow the report's Table 5. The X-* rows come from the report's Table 4 but were not run in the report. The report also ran each loss only once, which left run-to-run noise unmeasured. This study uses 3 seeds per loss.

The study asks four questions:
1. Does each added term improve quality at low SNR, with reverb, and for weak speech?
2. Which terms help and which don't?
3. Are the differences larger than random run-to-run variation?
4. Does optimising for waveform fidelity (SI-SNR) help or hurt?

---

## 2. Data used

All data is public and was downloaded for this study (about 1 GB extracted).

### Clean speech: LibriSpeech dev-clean
| Property | Value |
|---|---|
| Source | openslr.org/12 (`dev-clean.tar.gz`, 338 MB) |
| Content | Read English audiobook speech, clean recordings |
| Size | 2,703 utterances, 40 speakers (about 5.4 h) |
| Format | 16 kHz, mono, FLAC |
| Used for training | 750 utterances (25 each) from 30 speakers |
| Used for testing | 97 utterances (up to 12 each, at least 4 s long, cropped to 4 s) from the 10 other speakers |

Training and test speakers do not overlap, so test speakers are unseen. The fixed test set uses the first 60 of the 97 test utterances, so it covers roughly half of the held-out speakers.

### Noise: ESC-50
| Property | Value |
|---|---|
| Source | github.com/karoldvl/ESC-50 (646 MB) |
| Content | 2,000 five-second environmental sound clips in 50 classes (animals, weather, household, urban, human non-speech sounds) |
| Format | 44.1 kHz originally, resampled to 16 kHz mono |
| After cleaning | 1,999 clips (one near-silent clip dropped) |
| Used for training | folds 1–4 (1,599 clips) |
| Used for testing | fold 5 (400 clips), so test noise types are unseen clips |

ESC-50 noise is non-stationary (dog barks, door knocks, rain), which suits the report's focus on non-stationary noise. Many clips contain stretches of near-silence, so the effective noise is bursty.

### How the training and test audio are built
- **Mixing:** speech plus noise at a random SNR from -5 to 15 dB during training. Input level is normalised to a fixed RMS.
- **Reverberation (50% of training examples):** synthetic exponentially decaying impulse responses, RT60 0.2–0.8 s, direct-to-reverberant ratio 0–10 dB.
- **Target:** clean speech, or for reverberant input the early-reflection speech (first 50 ms of the impulse response). This is the report's partial-dereverberation target (Approach 4).
- **Fixed test set:** 4 conditions × 60 utterances with a fixed random seed: -5 dB dry, 5 dB dry, 15 dB dry, and 5 dB with reverb. Note that the same 60 clean utterances are reused in every condition, with different noise draws.

### Why these datasets
- Both are free, direct downloads, and the report's own reference setup used them.
- Speaker-disjoint and noise-fold-disjoint splits give a real test of generalisation to unseen speakers and noise.
- They are small enough for laptop-CPU training.

### Limits of the data
- **Not VoiceBank+DEMAND or DNS Challenge**, the report's primary benchmarks. Absolute numbers cannot be compared with published tables.
- **Synthetic reverberation**, not measured room responses.
- **Small test set**: 60 utterances per condition, roughly half the held-out speakers.
- **Read speech only**: no spontaneous or multilingual speech.

---

## 3. Model and training setup

| Item | Value |
|---|---|
| Model | causal conv-recurrent network predicting a complex ratio mask |
| Parameters | 77,159 |
| Structure | 3 causal conv blocks (16 channels) → 1 GRU (48 units) → 3 transposed convs with skips → bounded complex mask |
| Features | STFT with 20 ms window, 10 ms hop (161 bins), compressed complex spectrum as input |
| Training | 1,200 steps, batch 12 × 3 s, Adam with one-cycle schedule |
| Seeds | 3 per loss (30 runs in total) |
| Compute | laptop CPU |

Only the loss changes between runs. Architecture, data, schedule and seeds are identical.

**Metrics** (each seed's value is the mean over the 60 test utterances):
- **PESQ (wideband):** perceptual quality estimate, higher is better.
- **STOI:** intelligibility estimate, higher is better.
- **SI-SDR improvement (dB):** signal fidelity gained over the noisy input, higher is better.
- **Low-energy LSD (dB):** log-spectral error on the quietest 30% of speech-active frames, lower is better. It measures over-suppression of weak speech.

---

## 4. Results

Mean over 3 seeds ± seed std. Full tables are in `RESULTS.md`.

| Loss | PESQ -5 dB | PESQ 5 dB | PESQ 5 dB reverb | SI-SDRi 5 dB (dB) | Low-energy LSD 5 dB (dB, lower is better) |
|---|---|---|---|---|---|
| Unprocessed input | 1.23 | 1.34 | 1.30 | 0 | n/a |
| X-LIN | 1.21 ± 0.08 | 1.29 ± 0.12 | 1.26 ± 0.09 | -11.7 ± 6.1 | 11.66 |
| R0 (c=0.3) | 1.23 ± 0.08 | 1.32 ± 0.12 | 1.30 ± 0.09 | -20.9 ± 4.9 | 10.09 |
| R1 | 1.39 ± 0.04 | 1.53 ± 0.01 | 1.42 ± 0.01 | +2.9 ± 0.6 | 13.44 |
| R2 | 1.45 ± 0.02 | 1.59 ± 0.01 | 1.47 ± 0.00 | +3.8 ± 0.2 | 10.89 |
| **R3** | **1.45 ± 0.01** | **1.60 ± 0.02** | 1.48 ± 0.01 | **+4.2 ± 0.1** | **9.63** |
| R4 | 1.44 ± 0.03 | 1.60 ± 0.03 | **1.49 ± 0.00** | +3.9 ± 0.1 | 9.63 |
| R5 | 1.34 ± 0.19 | 1.47 ± 0.20 | 1.39 ± 0.15 | +2.7 ± 3.7 | 10.16 |
| R6 | 1.31 ± 0.15 | 1.52 ± 0.09 | 1.43 ± 0.07 | +1.8 ± 2.9 | 10.07 |

(A full chart is in `loss_comparison.png`.)

---

## 5. What we learned

1. **Phase-aware error is the single biggest gain (R0 → R1).** Magnitude-only losses (X-LIN, X-C*, R0) give the model no phase supervision. These models score about the same as unprocessed input on PESQ and make SI-SDR much worse (-12 to -37 dB). Adding the compressed complex term lifted PESQ by 0.1–0.2 and moved SI-SDR improvement from about -14 dB to +6 dB at -5 dB SNR.

2. **Multi-resolution (R2) and the over-suppression penalty (R3) each add a smaller gain.** Together they raised PESQ by another 0.06–0.07 and STOI by about 0.03 over R1. The seed std for R1–R4 is small (about 0.01–0.04 PESQ), so these steps look real, although with only 3 seeds this is a rough judgement.

3. **Complex error alone hurts weak speech, and the later terms repair it.** R1 has worse low-energy log-spectral distortion than R0 (13.4 dB against 10.1 dB at 5 dB SNR), meaning it suppresses weak speech more. The multi-resolution term and especially the over-suppression penalty bring it back below R0 (R3: 9.6 dB). This supports the report's argument that over-suppression is a real failure that needs its own term.

4. **Loudness weighting (R4) made no clear difference.** R3 and R4 are within seed noise on every metric. With an A-weighting proxy, there is no evidence it helps.

5. **Adding SI-SNR (R5, R6) made results worse and much less stable.** Mean PESQ fell below R3, and the seed std grew to 3–7 dB on SI-SDR improvement, which means some seeds worked and others didn't. The 0.01 weight was not tuned, so this may reflect the weight rather than SI-SNR itself.

6. **The compression exponent is not decisive at this scale.** c=0.5 had the best mean PESQ among the magnitude-only losses, but the differences between c=0.3, 0.5 and 0.7 are within seed noise. The report cites c=0.3 as better than 0.7, and this study can't confirm that.

7. **Best overall: R3.** Compressed magnitude, compressed complex error, multi-resolution error and the over-suppression penalty gave the best or tied-best result on most metrics, with low variance.

8. **Run-to-run variance matters.** The magnitude-only losses and the SI-SNR variants have large seed variation. The report's single-seed design could not have told these cases apart, which supports running multiple seeds.

---

## 6. Limitations and caveats

- **Undertrained:** 1,200 steps on a laptop CPU. Absolute quality is modest. At 15 dB SNR the best PESQ (2.37) is only slightly above unprocessed (2.10), and SI-SDR improvement there is still negative. The rankings are relative, not deployable numbers.
- **Only 3 seeds**, so std estimates are rough. A formal significance test was not run.
- **Small, partly reused test set:** 60 utterances per condition, from about half the held-out speakers, with the same clean utterances in every condition.
- **Synthetic reverb and a non-standard dataset:** conclusions may not transfer to real rooms or to VoiceBank+DEMAND and DNS data.
- **Choices I made without guidance from the report:** loss weights (0.3/0.7 for magnitude and complex, 1:1 for the extra terms, 0.01 for SI-SNR), the A-weighting proxy, and the model's exact layers. Different choices could shift the results.
- **PESQ-based losses not tested.** The §4.8 PESQ-loss arms (A2–A4) need a differentiable PESQ implementation that isn't available here.
- **No listening test.** All conclusions rest on objective metrics, which the report itself warns can diverge from listener preference.

---

## 7. Suggested next steps

1. Train longer (5–10× the steps) to see whether the ranking holds when the models are better trained.
2. Tune the SI-SNR weight (for example 0.001–0.05) before concluding it hurts.
3. Rerun on VoiceBank+DEMAND to allow comparison with published results.
4. Add more seeds and a paired significance test.
5. Add the PESQ-loss arms from §4.8.

---

## Files

- `code/selib.py`: data, model, losses and metrics
- `code/run_one.py`, `code/run_all.sh`: training and evaluation
- `code/analyze.py`: builds the tables and chart
- `results/RESULTS.md`: full result tables
- `results/loss_comparison.png`: chart
- `results/all_runs.csv`: per-run numbers
