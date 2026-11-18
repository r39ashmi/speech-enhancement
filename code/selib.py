"""Speech-enhancement loss-function evaluation: data, model, losses, metrics."""
import glob, os, math
import numpy as np, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import resample_poly, fftconvolve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000
WIN, HOP, NF = 320, 160, 161          # 20 ms window, 10 ms hop
SEG = 3 * SR
CACHE = os.path.join(ROOT, "results", "cache")
LS = os.path.join(ROOT, "data/librispeech/LibriSpeech/dev-clean")
ESC = glob.glob(os.path.join(ROOT, "data/esc50/ESC-50-*"))[0]


# ------------------------------------------------------------------ data
def build_cache():
    os.makedirs(CACHE, exist_ok=True)
    if os.path.exists(f"{CACHE}/clean.npz"):
        return
    spk = sorted(os.listdir(LS))
    rng = np.random.RandomState(0)
    rng.shuffle(spk)
    split = {"train": spk[:30], "test": spk[30:]}   # speaker-disjoint
    out = {}
    for name, lst in split.items():
        utts = []
        for s in lst:
            fl = sorted(glob.glob(f"{LS}/{s}/*/*.flac"))
            rng.shuffle(fl)
            n = 25 if name == "train" else 12
            for f in fl[:n]:
                x, sr = sf.read(f, dtype="float32")
                if name == "test":
                    if len(x) < 4 * SR: continue
                    x = x[: 4 * SR]
                utts.append(x)
        out[name] = utts
        print(name, len(utts), "utterances,", len(lst), "speakers")
    np.savez(f"{CACHE}/clean.npz",
             train=np.array(out["train"], dtype=object), test=np.array(out["test"], dtype=object),
             allow_pickle=True)
    import csv
    meta = list(csv.DictReader(open(f"{ESC}/meta/esc50.csv")))
    noise = {"train": [], "test": []}
    for m in meta:
        x, sr = sf.read(f"{ESC}/audio/{m['filename']}", dtype="float32")
        if x.ndim > 1: x = x.mean(1)
        x = resample_poly(x, 160, 441).astype(np.float32)   # 44.1k -> 16k
        if np.sqrt(np.mean(x ** 2)) < 1e-3: continue          # skip (near-)silent clips
        noise["train" if int(m["fold"]) <= 4 else "test"].append(x)
    print("noise clips", {k: len(v) for k, v in noise.items()})
    np.savez(f"{CACHE}/noise.npz", train=np.array(noise["train"], dtype=object),
             test=np.array(noise["test"], dtype=object), allow_pickle=True)


def load_cache():
    c = np.load(f"{CACHE}/clean.npz", allow_pickle=True)
    n = np.load(f"{CACHE}/noise.npz", allow_pickle=True)
    return list(c["train"]), list(c["test"]), list(n["train"]), list(n["test"])


def synth_rir(rng):
    """Synthetic exponentially decaying RIR; direct path + diffuse tail."""
    rt60 = rng.uniform(0.2, 0.8); drr = rng.uniform(0, 10)
    n = int(0.8 * SR); t = np.arange(n) / SR
    tail = rng.randn(n) * np.exp(-6.9 * t / rt60)
    tail[:int(0.003 * SR)] = 0                     # keep direct path clean
    tail *= math.sqrt(10 ** (-drr / 10) / (np.sum(tail ** 2) + 1e-12))
    h = tail.copy(); h[0] = 1.0
    return h.astype(np.float32), rt60


def make_example(clean, noise, snr, reverb, rng, length):
    """Return noisy, target (early-reflection speech: first 50 ms of RIR), clean-dry."""
    if len(clean) > length:
        o = rng.randint(0, len(clean) - length + 1); clean = clean[o:o + length]
    elif len(clean) < length:
        clean = np.pad(clean, (0, length - len(clean)))
    if reverb:
        h, rt = synth_rir(rng)
        rev = fftconvolve(clean, h)[:length]
        tgt = fftconvolve(clean, h[:int(0.05 * SR)])[:length]
    else:
        rev = tgt = clean
    if len(noise) < length: noise = np.tile(noise, length // len(noise) + 1)
    o = rng.randint(0, len(noise) - length + 1); nz = noise[o:o + length]
    ps = np.mean(rev ** 2) + 1e-10; pn = np.mean(nz ** 2) + 1e-10
    nz = nz * math.sqrt(ps / (pn * 10 ** (snr / 10)))
    noisy = rev + nz
    g = 0.05 / (np.sqrt(np.mean(noisy ** 2)) + 1e-9)   # fixed input level
    return (noisy * g).astype(np.float32), (tgt * g).astype(np.float32)


def train_batch(clean, noise, rng, bs):
    X, Y = [], []
    for _ in range(bs):
        x, y = make_example(clean[rng.randint(len(clean))], noise[rng.randint(len(noise))],
                            rng.uniform(-5, 15), rng.rand() < 0.5, rng, SEG)
        X.append(x); Y.append(y)
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y))


TEST_CONDS = {"-5dB dry": (-5, False), "5dB dry": (5, False), "15dB dry": (15, False),
              "5dB reverb": (5, True)}


def test_set(clean, noise, n=60):
    rng = np.random.RandomState(123)
    out = {}
    for k, (snr, rv) in TEST_CONDS.items():
        out[k] = [make_example(clean[i % len(clean)], noise[rng.randint(len(noise))], snr, rv, rng, 4 * SR)
                  for i in range(n)]
    return out


# ------------------------------------------------------------------ STFT
_WIN = {}
def stft(x):
    w = _WIN.setdefault("w", torch.sin(math.pi * (torch.arange(WIN) + 0.5) / WIN))
    x = F.pad(x, (WIN - HOP, WIN - HOP))     # causal framing
    return torch.stft(x, WIN, HOP, WIN, w, center=False, return_complex=True)   # (B,F,T)

def istft(S, length):
    w = _WIN.setdefault("w", torch.sin(math.pi * (torch.arange(WIN) + 0.5) / WIN))
    y = torch.istft(S, WIN, HOP, WIN, w, center=False)
    return y[:, WIN - HOP: WIN - HOP + length]


# ------------------------------------------------------------------ model
class CCB(nn.Module):  # causal-in-time conv block, strided in frequency
    def __init__(s, ci, co, k=(2, 3), sf=2):
        super().__init__(); s.k = k
        s.c = nn.Conv2d(ci, co, k, stride=(1, sf), padding=(0, k[1] // 2)); s.n = nn.BatchNorm2d(co); s.a = nn.PReLU()
    def forward(s, x): return s.a(s.n(s.c(F.pad(x, (0, 0, s.k[0] - 1, 0)))))

class CRN(nn.Module):
    """Causal conv-recurrent mask estimator (~80K params). Predicts a complex ratio mask."""
    def __init__(s, c=16, h=48):
        super().__init__()
        s.e1 = CCB(2, c, (1, 5)); s.e2 = CCB(c, c); s.e3 = CCB(c, c)       # 161 -> 81 -> 41 -> 21
        s.gru = nn.GRU(c * 21, h, batch_first=True); s.fc = nn.Linear(h, c * 21)
        s.d3 = nn.ConvTranspose2d(c, c, (1, 3), (1, 2), (0, 1)); s.d2 = nn.ConvTranspose2d(c, c, (1, 3), (1, 2), (0, 1))
        s.d1 = nn.ConvTranspose2d(c, 2, (1, 5), (1, 2), (0, 2)); s.c = c
        s.n3 = nn.PReLU(); s.n2 = nn.PReLU()
    def forward(s, Y):                       # Y (B,F,T) complex
        mag = Y.abs().clamp_min(1e-8); comp = Y / mag * mag ** 0.3
        x = torch.stack([comp.real, comp.imag], 1).permute(0, 1, 3, 2)      # B,2,T,F
        a1 = s.e1(x); a2 = s.e2(a1); a3 = s.e3(a2)
        B, C, T, Fq = a3.shape
        z, _ = s.gru(a3.permute(0, 2, 1, 3).reshape(B, T, C * Fq))
        z = s.fc(z).reshape(B, T, C, Fq).permute(0, 2, 1, 3) + a3
        z = s.n3(s.d3(z, output_size=a2.shape) + a2)
        z = s.n2(s.d2(z, output_size=a1.shape) + a1)
        m = 2 * torch.tanh(s.d1(z, output_size=(B, 2, T, NF)))              # B,2,T,F
        M = torch.complex(m[:, 0], m[:, 1]).permute(0, 2, 1)                # B,F,T
        return M * Y


def count_params(m): return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ losses
def _aw_weights():
    f = torch.linspace(0, SR / 2, NF).clamp_min(20.0)
    ra = (12194 ** 2 * f ** 4) / ((f ** 2 + 20.6 ** 2) * torch.sqrt((f ** 2 + 107.7 ** 2) * (f ** 2 + 737.9 ** 2)) * (f ** 2 + 12194 ** 2))
    w = ra ** 2 / ra.max() ** 2                                              # A-weighting power (proxy for equal-loudness)
    w = w.clamp_min(0.05)
    return (w / w.mean()).view(1, NF, 1)
AW = _aw_weights()

def cmag(S, c): return S.abs().clamp_min(1e-8) ** c
def ccomp(S, c): a = S.abs().clamp_min(1e-8); return S / a * a ** c


def mr_loss(est, ref, c=0.3):
    tot = 0
    for ms in (5, 10, 20, 40):
        n = int(ms * SR / 1000); h = n // 2; w = torch.hann_window(n)
        E = torch.stft(est, n, h, n, w, return_complex=True); R = torch.stft(ref, n, h, n, w, return_complex=True)
        tot = tot + F.mse_loss(cmag(E, c), cmag(R, c))
    return tot / 4


def si_snr(est, ref, eps=1e-8):
    est = est - est.mean(1, keepdim=True); ref = ref - ref.mean(1, keepdim=True)
    t = (est * ref).sum(1, keepdim=True) * ref / (ref.pow(2).sum(1, keepdim=True) + eps)
    return 10 * torch.log10(t.pow(2).sum(1) / ((est - t).pow(2).sum(1) + eps) + eps)

# name -> dict of term switches. R0..R6 follow Table 5 of the report; X-* are extra rungs from Table 4.
LOSSES = {
    "X-LIN":  dict(c=1.0),
    "X-C0.7": dict(c=0.7),
    "X-C0.5": dict(c=0.5),
    "R0":     dict(c=0.3),
    "R1":     dict(c=0.3, cplx=1),
    "R2":     dict(c=0.3, cplx=1, mr=1),
    "R3":     dict(c=0.3, cplx=1, mr=1, ovs=1),
    "R4":     dict(c=0.3, cplx=1, mr=1, ovs=1, loud=1),
    "R5":     dict(c=0.3, cplx=1, mr=1, ovs=1, sisnr=1),
    "R6":     dict(c=0.3, cplx=1, mr=1, ovs=1, loud=1, sisnr=1),
    # Section 4.8 PESQ-loss arms (A0 = R1, A1 = R3 already run)
    "A2":     dict(c=0.3, cplx=1, mr=1, ovs=1, pesq=0.1),    # R3 + differentiable PESQ, low weight
    "A3":     dict(c=0.3, cplx=1, mr=1, ovs=1, pesq=1.0),    # R3 + differentiable PESQ, high weight
    "A4":     dict(c=0.3, cplx=1, mr=1, ovs=1, gan=0.05),    # R3 + MetricGAN-style PESQ discriminator
}

_PL = {}
def pesq_term(x_est, x_ref):
    if "p" not in _PL:
        from torch_pesq import PesqLoss
        _PL["p"] = PesqLoss(1.0, sample_rate=SR)
    if x_est.requires_grad:   # library gives NaN grads on rare utterances: zero them, and drop non-finite items
        x_est.register_hook(lambda g: torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0))
    l = _PL["p"](x_ref, x_est).float()
    ok = torch.isfinite(l)
    return torch.where(ok, l, torch.zeros_like(l)).sum() / ok.sum().clamp_min(1)


class Disc(nn.Module):
    """MetricGAN-style discriminator: predicts normalised PESQ of (enhanced, reference) magnitude pair."""
    def __init__(s, c=16):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(2, c, 5, 2, 2), nn.LeakyReLU(0.2), nn.Conv2d(c, c, 5, 2, 2), nn.LeakyReLU(0.2),
                              nn.Conv2d(c, c, 5, 2, 2), nn.LeakyReLU(0.2), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                              nn.Linear(c, 1), nn.Sigmoid())
    def forward(s, S_est, S_ref):
        x = torch.stack([cmag(S_est, 0.3), cmag(S_ref, 0.3)], 1)
        return s.net(x).squeeze(1)

def compute_loss(cfg, S_est, S_ref, x_est, x_ref):
    c = cfg["c"]; w = AW if cfg.get("loud") else 1.0
    me, mr_ = cmag(S_est, c), cmag(S_ref, c)
    if cfg.get("cplx"):
        L = 0.3 * (w * (me - mr_) ** 2).mean() + 0.7 * (w * (ccomp(S_est, c) - ccomp(S_ref, c)).abs() ** 2).mean()
    else:
        L = (w * (me - mr_) ** 2).mean()
    if cfg.get("mr"):   L = L + mr_loss(x_est, x_ref, c)
    if cfg.get("ovs"):  L = L + (w * F.relu(mr_ - me) ** 2).mean()          # penalise only under-estimation
    if cfg.get("sisnr"): L = L - 0.01 * si_snr(x_est, x_ref).mean()
    if cfg.get("pesq"): L = L + cfg["pesq"] * pesq_term(x_est, x_ref)
    return L


# ------------------------------------------------------------------ metrics
def lowenergy_lsd(est, ref):
    """Log-spectral distortion (dB, lower=better) on the quietest 30 % of speech-active frames of the target."""
    E = stft(torch.from_numpy(est)[None])[0].abs() ** 2; R = stft(torch.from_numpy(ref)[None])[0].abs() ** 2
    fe = R.sum(0); active = fe > fe.max() * 1e-4                              # speech-active (-40 dB re peak)
    idx = torch.where(active)[0]
    if len(idx) < 10: return float("nan")
    q = torch.quantile(fe[idx], 0.3); sel = idx[fe[idx] <= q]
    d = 10 * (torch.log10(E[:, sel] + 1e-10) - torch.log10(R[:, sel] + 1e-10))
    return d.abs().mean().item()

def si_sdr_np(est, ref):
    est = est - est.mean(); ref = ref - ref.mean()
    t = (est @ ref) / (ref @ ref + 1e-10) * ref
    return 10 * np.log10((t @ t) / (((est - t) @ (est - t)) + 1e-10) + 1e-10)

def evaluate(model, tset):
    from pesq import pesq
    from pystoi import stoi
    model.eval(); res = {}
    with torch.no_grad():
        for cond, items in tset.items():
            rows = []
            for noisy, tgt in items:
                y = torch.from_numpy(noisy)[None]
                est = istft(model(stft(y)), len(noisy))[0].numpy()
                try: p = pesq(SR, tgt, est, "wb"); p0 = pesq(SR, tgt, noisy, "wb")
                except Exception: p = p0 = float("nan")
                rows.append([p, p0, stoi(tgt, est, SR), stoi(tgt, noisy, SR),
                             si_sdr_np(est, tgt) - si_sdr_np(noisy, tgt), lowenergy_lsd(est, tgt), lowenergy_lsd(noisy, tgt)])
            res[cond] = np.array(rows)
    return res   # cols: pesq, pesq_in, stoi, stoi_in, sisdri, lelsd, lelsd_in
