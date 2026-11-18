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

A second experiment covers the report's §4.8, **PESQ as a training loss**. The report cites work finding that PESQ-based losses mainly raise PESQ itself while lowering fidelity, and it leaves open whether this holds for a tiny model. Three arms were added on top of R3 (the best ladder loss, called A1 in the report; A0 is R1):

| Arm | Loss | PESQ term |
|---|---|---|
| A2 | R3 + differentiable PESQ (`torch-pesq`) | weight 0.1 (low) |
| A3 | R3 + differentiable PESQ | weight 1.0 (high) |
| A4 | R3 + MetricGAN-style discriminator that learns real PESQ | weight 0.05 |

The study asks four questions:
1. Does each added term improve quality at low SNR, with reverb, and for weak speech?
2. Which terms help and which don't?
3. Are the differences larger than random run-to-run variation?
4. Does optimising for waveform fidelity (SI-SNR) help or hurt?
5. Does training directly on a PESQ term raise real PESQ, and at what cost to fidelity and weak speech?

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
| Seeds | 3 per loss (39 runs in total: 30 for the ladder, 9 for the PESQ arms) |
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

### 4b. PESQ as a training loss (§4.8)

Evaluation uses the standard `pesq` package, not the differentiable surrogate used in training, so the PESQ gains below are real PESQ gains. Mean ± seed std at 5 dB dry (other conditions show the same pattern):

| Arm | PESQ | STOI | SI-SDRi (dB) | Low-energy LSD (dB, lower is better) |
|---|---|---|---|---|
| A1 = R3 (no PESQ term) | 1.60 ± 0.02 | 0.855 ± 0.002 | +4.18 ± 0.08 | 9.63 ± 0.23 |
| A2 (PESQ, low weight) | 2.32 ± 0.05 | 0.853 ± 0.001 | -6.05 ± 0.68 | 11.28 ± 0.08 |
| A3 (PESQ, high weight) | **2.43 ± 0.07** | 0.837 ± 0.007 | -11.72 ± 0.24 | 16.79 ± 0.74 |
| A4 (MetricGAN, 0.05) | 1.58 ± 0.01 | 0.851 ± 0.001 | +4.18 ± 0.14 | 9.43 ± 0.21 |

PESQ at -5 dB: R3 1.45, A2 1.92, A3 1.97, A4 1.45. At 15 dB: R3 2.35, A2 3.15, A3 3.30, A4 2.31.

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

9. **A direct PESQ term inflates PESQ and pays for it in fidelity (A2, A3).** PESQ rose by 0.5–1.0 over R3 in every condition, far beyond any other change in the study. But SI-SDR improvement fell from +4 dB to -6 dB (low weight) and -12 dB (high weight) at 5 dB SNR, and low-energy distortion worsened from 9.6 dB to 11.3 dB and 16.8 dB. STOI dropped at the high weight. The effect grows with the PESQ weight, which is the dose-response the report predicted. This supports H1 (PESQ rises) and H2 (SI-SDR and low-energy retention fall as the weight rises). The big PESQ gain alongside a large SI-SDR loss means the model learned to produce outputs that PESQ rates highly but that are far from the clean waveform. That is a sign of exploiting the metric, not of better speech.

10. **The MetricGAN discriminator had no measurable effect (A4).** At weight 0.05 it matched R3 on every metric within seed noise. This does not show that MetricGAN losses are harmless. The discriminator was small, trained on only 4 examples per step for 1,200 steps, and weighted weakly, so it may simply not have learned a useful signal.

11. **Decision rule from §4.8.** A PESQ term is adopted only if it wins a listening test, or improves at least two non-PESQ metrics on the mismatched set with an SI-SDR loss under 1 dB. Neither A2 nor A3 comes close (SI-SDR loss of 10–16 dB, and weaker low-energy retention), and A4 changes nothing. So **A1 (R3) stays the final loss**, based on the objective metrics.

---

## 6. Limitations and caveats

- **Undertrained:** 1,200 steps on a laptop CPU. Absolute quality is modest. At 15 dB SNR the best PESQ (2.37) is only slightly above unprocessed (2.10), and SI-SDR improvement there is still negative. The rankings are relative, not deployable numbers.
- **Only 3 seeds**, so std estimates are rough. A formal significance test was not run.
- **Small, partly reused test set:** 60 utterances per condition, from about half the held-out speakers, with the same clean utterances in every condition.
- **Synthetic reverb and a non-standard dataset:** conclusions may not transfer to real rooms or to VoiceBank+DEMAND and DNS data.
- **Choices I made without guidance from the report:** loss weights (0.3/0.7 for magnitude and complex, 1:1 for the extra terms, 0.01 for SI-SNR), the A-weighting proxy, and the model's exact layers. Different choices could shift the results.
- **PESQ-loss arms are only partly tested.** H1 and H2 were tested. H3 (the PESQ gain shrinks or reverses on mismatched data) was not, because there is no EARS-WHAM or similar set here. My test speakers and noise clips are unseen, but they come from the same corpora. H4 (listeners do not prefer the PESQ arms) needs a listening test, which was not run.
- **The PESQ weights (0.1 and 1.0) were my choice.** The report does not give them, and the low-weight arm already shows a large fidelity cost, so a still lower weight might behave differently. The surrogate (`torch-pesq`) also produced NaN gradients on some utterances, which I zeroed.
- **A4 is a light implementation of MetricGAN.** See finding 10.
- **No listening test.** All conclusions rest on objective metrics, which the report itself warns can diverge from listener preference. The fidelity drop in A2 and A3 is a reason to doubt that the higher PESQ reflects better listening quality, but only a listening test can settle it.

---

## 7. Suggested next steps

1. Train longer (5–10× the steps) to see whether the ranking holds when the models are better trained.
2. Tune the SI-SNR weight (for example 0.001–0.05) before concluding it hurts.
3. Rerun on VoiceBank+DEMAND to allow comparison with published results.
4. Add more seeds and a paired significance test.
5. Test the PESQ-loss arms on a mismatched set (for example EARS-WHAM) for H3, and run a listening test for H4.
6. Try smaller PESQ weights (for example 0.01) and a stronger MetricGAN discriminator, to see whether any PESQ term helps without the fidelity loss.

---

## Files

- `code/selib.py`: data, model, losses and metrics
- `code/run_one.py`, `code/run_all.sh`, `code/run_pesq.sh`: training and evaluation (ladder sweep and PESQ-loss arms)
- `code/analyze.py`: builds the tables and chart
- `results/RESULTS.md`: full result tables
- `results/loss_comparison.png`: chart
- `results/all_runs.csv`: per-run numbers
