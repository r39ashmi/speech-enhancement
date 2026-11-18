"""Train + evaluate one (loss, seed) run.  usage: run_one.py LOSS SEED STEPS"""
import sys, os, time, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import selib as L

name, seed, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = f"{L.ROOT}/results/runs/{name}_s{seed}.json"
if os.path.exists(out): sys.exit()
torch.set_num_threads(2); torch.manual_seed(seed); rng = np.random.RandomState(seed)
cl_tr, cl_te, nz_tr, nz_te = L.load_cache()
tset = L.test_set(cl_te, nz_te)
model = L.CRN(); cfg = L.LOSSES[name]
disc = None
if cfg.get("gan"):
    from pesq import pesq as _pesq
    disc = L.Disc(); optD = torch.optim.Adam(disc.parameters(), 1e-3)
opt = torch.optim.Adam(model.parameters(), 2e-3)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=steps, pct_start=0.1)
t0 = time.time(); hist = []
for it in range(steps):
    model.train()
    x, y = L.train_batch(cl_tr, nz_tr, rng, 12)
    Se = model(L.stft(x)); Sr = L.stft(y)
    xe = L.istft(Se, x.shape[1])
    loss = L.compute_loss(cfg, Se, Sr, xe, y)
    if disc is not None:
        loss = loss + cfg["gan"] * ((disc(Se, Sr) - 1) ** 2).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step(); sched.step()
    if disc is not None:   # discriminator learns true PESQ (normalised to [0,1]) of 4 enhanced examples + clean=1
        k = 4; q = []
        for i in range(k):
            try: q.append((_pesq(L.SR, y[i].numpy(), xe[i].detach().numpy(), "wb") - 1) / 3.5)
            except Exception: q.append(0.0)
        q = torch.tensor(q, dtype=torch.float32).clamp(0, 1)
        dl = ((disc(Se[:k].detach(), Sr[:k]) - q) ** 2).mean() + ((disc(Sr[:k], Sr[:k]) - 1) ** 2).mean()
        optD.zero_grad(); dl.backward(); optD.step()
    if it % 100 == 0: hist.append((it, loss.item())); print(name, seed, it, round(loss.item(), 4), round(time.time() - t0), "s", flush=True)
train_t = time.time() - t0
res = L.evaluate(model, tset)
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"loss": name, "seed": seed, "steps": steps, "params": L.count_params(model), "train_s": train_t,
           "hist": hist, "res": {k: v.tolist() for k, v in res.items()}}, open(out, "w"))
print("done", name, seed, {k: np.nanmean(v[:, 0]).round(3) for k, v in res.items()})
