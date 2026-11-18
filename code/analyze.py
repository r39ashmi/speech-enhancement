import json, glob, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/results"
ORDER = ["X-LIN", "X-C0.7", "X-C0.5", "R0", "R1", "R2", "R3", "R4", "R5", "R6"]
DESC = {"X-LIN": "linear magnitude MSE", "X-C0.7": "compressed mag c=0.7", "X-C0.5": "compressed mag c=0.5",
        "R0": "compressed mag c=0.3", "R1": "R0 + compressed complex", "R2": "R1 + multi-res (5/10/20/40ms)",
        "R3": "R2 + over-suppression pen.", "R4": "R3 + loudness (A-wt) weighting", "R5": "R3 + SI-SNR",
        "R6": "R3 + loudness + SI-SNR"}
MET = ["PESQ", "PESQ_in", "STOI", "STOI_in", "SI-SDRi", "LE-LSD", "LE-LSD_in"]
rows = []
for f in sorted(glob.glob(R + "/runs/*.json")):
    d = json.load(open(f))
    for cond, v in d["res"].items():
        m = np.nanmean(np.array(v), 0)
        rows.append(dict(loss=d["loss"], seed=d["seed"], cond=cond, params=d["params"], **dict(zip(MET, m))))
df = pd.DataFrame(rows); df.to_csv(R + "/all_runs.csv", index=False)
conds = list(df.cond.unique()); nseed = df.groupby("loss").seed.nunique()
g = df.groupby(["loss", "cond"])[MET].agg(["mean", "std"])

def cell(l, c, m, p=3):
    mu, sd = g.loc[(l, c), (m, "mean")], g.loc[(l, c), (m, "std")]
    return f"{mu:.{p}f} ± {sd:.{p}f}"

md = ["# Loss-function evaluation results\n",
      f"Seeds per loss: {dict(nseed)}. Values are mean ± std **across seeds** (each seed's value is the mean over 60 test utterances).\n"]
inp = df.groupby("cond")[["PESQ_in", "STOI_in", "LE-LSD_in"]].mean()
md.append("Unprocessed input: " + "; ".join(f"{c}: PESQ {inp.loc[c,'PESQ_in']:.2f}, STOI {inp.loc[c,'STOI_in']:.3f}" for c in conds) + "\n")
for m, label, hi in [("PESQ", "PESQ (wideband, higher better)", 1), ("STOI", "STOI (higher better)", 1),
                     ("SI-SDRi", "SI-SDR improvement dB (higher better)", 1), ("LE-LSD", "Low-energy log-spectral distortion dB (LOWER better)", 0)]:
    md.append(f"\n## {label}\n\n| Loss | Description | " + " | ".join(conds) + " |\n|---|---|" + "---|" * len(conds))
    for l in ORDER:
        if l not in nseed.index: continue
        best = {c: (g.xs(c, level="cond")[(m, "mean")].idxmax() if hi else g.xs(c, level="cond")[(m, "mean")].idxmin()) for c in conds}
        md.append(f"| {l} | {DESC[l]} | " + " | ".join(("**" + cell(l, c, m, 2 if m != 'STOI' else 3) + "**") if best[c] == l else cell(l, c, m, 2 if m != 'STOI' else 3) for c in conds) + " |")

# significance vs. reference R0 (paired over seeds is weak with n=3 -> report diff vs seed std)
md.append("\n## Difference vs. R0 (compressed magnitude), in units of seed-to-seed std\n")
md.append("A difference is only treated as real if |Δ| > 2× the larger of the two seed stds (rough screen; n=3 seeds).\n")
md.append("| Loss | " + " | ".join(f"{c} ΔPESQ" for c in conds) + " | " + " | ".join(f"{c} ΔSI-SDRi" for c in conds) + " |\n|---|" + "---|" * (2 * len(conds)))
for l in ORDER:
    if l == "R0" or l not in nseed.index: continue
    cells = []
    for m in ["PESQ", "SI-SDRi"]:
        for c in conds:
            d = g.loc[(l, c), (m, "mean")] - g.loc[("R0", c), (m, "mean")]
            s = max(g.loc[(l, c), (m, "std")], g.loc[("R0", c), (m, "std")])
            cells.append(f"{d:+.2f}{' *' if abs(d) > 2 * s else ''}")
    md.append(f"| {l} | " + " | ".join(cells) + " |")
md.append("\n`*` = exceeds 2× seed std.")
open(R + "/RESULTS.md", "w").write("\n".join(md)); print("\n".join(md))

fig, ax = plt.subplots(1, 4, figsize=(17, 4))
for a, m, t in zip(ax, ["PESQ", "STOI", "SI-SDRi", "LE-LSD"], ["PESQ ↑", "STOI ↑", "SI-SDR improvement (dB) ↑", "Low-energy LSD (dB) ↓"]):
    w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        ls = [l for l in ORDER if l in nseed.index]
        a.bar(np.arange(len(ls)) + i * w, [g.loc[(l, c), (m, "mean")] for l in ls], w, yerr=[g.loc[(l, c), (m, "std")] for l in ls], label=c, capsize=2)
    a.set_xticks(np.arange(len(ls)) + 0.4 - w / 2); a.set_xticklabels(ls, rotation=60); a.set_title(t)
    if m in ("PESQ", "STOI"): a.set_ylim(df[m].min() * 0.9, None)
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(R + "/loss_comparison.png", dpi=130)
