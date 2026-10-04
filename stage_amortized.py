"""Revision E2 + E3: amortized (real-time) malicious-RIS attack and outdated attacker CSI.

E2. Instead of running an 80-step optimizer per user per coherence interval, the attacker
trains offline, on its OWN users (the training split, disjoint from the test users), a
generator network G that maps its per-user knowledge (noisy direct-channel estimate and
RIS->user cascade) to the M RIS phases in a single forward pass. Online cost is one MLP
forward pass. Trained against the victim (white-box) or against an attacker-trained
surrogate (black-box). Reports test-user damage and single-user latency (CPU and GPU).

E3. Outdated attacker CSI. The attacker's knowledge of the direct channel and of the
RIS->user cascade lags the true channel by tau. Under first-order Gauss-Markov aging the
stale knowledge is x_hat = rho x + sqrt(1-rho^2) e with rho = J0(2 pi f_D tau) (Jakes),
e an independent innovation of matching per-entry power. The phases are crafted from the
stale knowledge (generator input, or the iterative optimizer's channel model) and applied
to the TRUE channel, which the victim measures; the victim's own operation is unchanged,
so only the attacker is affected by staleness. rho = 0 means the attacker knows nothing.

Usage: python3 stage_amortized.py {rsrp|csi}   (one victim per process, run in parallel)
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn
from scipy.special import j0
import beamdata as bd
import ris
import revision_common as rc
from model import DEVICE

M_LIST = [64, 128]
SEEDS = [42, 7, 2024]
ATT_CSI_SNR_DB = 10.0
RHOS = [1.0, 0.99, 0.95, 0.9, 0.8, 0.6, 0.3, 0.0]
N_GEN_TRAIN = 30000          # attacker-side users used to fit the generator


class PhaseGenerator(nn.Module):
    def __init__(self, n, M, hidden=(512, 512)):
        super().__init__()
        layers, d = [], 2 * n + 2 * M
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, M)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)          # unconstrained output used directly as phase angle


def gen_input(H_est, k_u):
    """Attacker's per-user features: unit-norm direct-channel estimate + cascade phases."""
    h = H_est / (torch.linalg.norm(H_est, dim=1, keepdim=True) + 1e-12)
    ang = torch.angle(k_u)
    return torch.cat([h.real, h.imag, torch.cos(ang), torch.sin(ang)], 1).float()


def sub(g, idx):
    return {k: (v[idx] if k in ("H_d", "k_u", "scale") else v) for k, v in g.items()}


def train_generator(target, g_tr, H_est_tr, M, epochs=20, bs=1024, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    G = PhaseGenerator(bd.N_BS_ANT, M).to(DEVICE)
    opt = torch.optim.Adam(G.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    X = gen_input(H_est_tr, g_tr["k_u"])
    Wc = g_tr["W"].conj()
    target.model.eval()
    for p in target.model.parameters():
        p.requires_grad_(False)
    U = X.shape[0]
    for ep in range(epochs):
        G.train(); perm = torch.randperm(U, device=DEVICE); tot = 0.0
        for i in range(0, U, bs):
            idx = perm[i:i + bs]
            theta = G(X[idx])
            loss = rc.soft_objective(target, ris._h_eff(sub(g_tr, idx), theta), Wc).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step()
    for p in target.model.parameters():
        p.requires_grad_(True)
    G.eval()
    return G


def noisy_t(H, snr_db, seed):
    return torch.as_tensor(bd.add_cn_noise(H, snr_db, np.random.default_rng(seed)),
                           dtype=torch.complex64, device=DEVICE)


@torch.no_grad()
def apply_generator(G, g, H_est):
    return G(gen_input(H_est, g["k_u"]))


@torch.no_grad()
def stale_knowledge(g, H_est, rho, seed):
    """Attacker's outdated view: Gauss-Markov-aged direct-channel estimate and cascade."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    def cn(shape):
        return (torch.randn(shape, generator=gen, device=DEVICE)
                + 1j * torch.randn(shape, generator=gen, device=DEVICE)) / np.sqrt(2)
    a = float(np.sqrt(max(0.0, 1 - rho ** 2)))
    sig = torch.linalg.norm(H_est, dim=1, keepdim=True) / np.sqrt(H_est.shape[1])
    H_old = (rho * H_est + a * sig * cn(H_est.shape)).to(torch.complex64)
    g_old = dict(g)
    g_old["k_u"] = (rho * g["k_u"] + a * cn(g["k_u"].shape)).to(torch.complex64)
    g_old["H_d"] = H_old
    return g_old, H_old


def latency(G, x1, device, reps=500):
    """Median single-user forward time; CPU timing uses one thread (single core)."""
    nthr = torch.get_num_threads(); torch.set_num_threads(1)
    G = G.to(device); x1 = x1.to(device)
    with torch.no_grad():
        for _ in range(20):
            G(x1)
        ts = []
        for _ in range(reps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter(); G(x1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    G.to(DEVICE); torch.set_num_threads(nthr)
    return float(np.median(ts) * 1e3)


def main():
    kind = sys.argv[1]
    d = bd.build_dataset()
    H_tr, H_te, W = d["H_tr"], d["H_test"], d["W"]
    sel = np.random.default_rng(0).permutation(len(H_tr))[:N_GEN_TRAIN]
    out = {"kind": kind, "seeds": SEEDS, "rhos": RHOS, "n_gen_train": N_GEN_TRAIN, "results": {}}
    out["results"] = {str(M): [] for M in M_LIST}
    for s in SEEDS:
        victim = rc.train_victim(kind, d, seed=s)
        surrogate = rc.train_victim(kind, d, seed=s + 1000, hidden=(384, 192))
        Hest_tr = noisy_t(H_tr[sel], ATT_CSI_SNR_DB, s + 1)
        Hest_te = noisy_t(H_te, ATT_CSI_SNR_DB, s + 2)
        clean = rc.evaluate_noise_avg(victim, H_te, W)
        for M in M_LIST:
            g_tr = ris.build_geometry(H_tr[sel], M, 1.0, seed=s + 500)
            g_te = ris.build_geometry(H_te, M, 1.0, seed=s)
            r = {"seed": s, "clean": clean}
            r["iterative"] = rc.evaluate_noise_avg(victim, rc.ris_attack(victim, g_te, seed=s), W)
            r["random"] = rc.evaluate_noise_avg(victim, rc.random_ris(g_te, seed=s), W)
            for tag, tgt in (("gen_whitebox", victim), ("gen_blackbox", surrogate)):
                t0 = time.time()
                G = train_generator(tgt, g_tr, Hest_tr, M, seed=s)
                r[f"{tag}_train_s"] = time.time() - t0
                theta = apply_generator(G, g_te, Hest_te)
                r[tag] = rc.evaluate_noise_avg(victim, ris._h_eff(g_te, theta).cpu().numpy(), W)
                if tag == "gen_whitebox":
                    x1 = gen_input(Hest_te[:1], g_te["k_u"][:1])
                    r["latency_ms_cpu"] = latency(G, x1, torch.device("cpu"))
                    if DEVICE.type == "cuda":
                        r["latency_ms_gpu"] = latency(G, x1, torch.device("cuda"))
                    r["gen_params"] = int(sum(p.numel() for p in G.parameters()))
                    # E3: phases from stale knowledge, applied to the true channel
                    r["stale_gen"], r["stale_iter"] = {}, {}
                    for rho in RHOS:
                        g_old, H_old = stale_knowledge(g_te, Hest_te, rho, seed=s + 7)
                        th = apply_generator(G, g_old, H_old)
                        r["stale_gen"][str(rho)] = rc.evaluate_noise_avg(
                            victim, ris._h_eff(g_te, th).cpu().numpy(), W)
                        if rho in (1.0, 0.95, 0.8, 0.6, 0.0):
                            _, th_i = rc.ris_attack(victim, g_old, seed=s, return_theta=True)
                            r["stale_iter"][str(rho)] = rc.evaluate_noise_avg(
                                victim, ris._h_eff(g_te, th_i).cpu().numpy(), W)
            out["results"][str(M)].append(r)
            print(f"[{kind} M={M} s={s}] clean={r['clean']['top1']*100:.1f} rand={r['random']['top1']*100:.1f} "
                  f"iter={r['iterative']['top1']*100:.1f} genWB={r['gen_whitebox']['top1']*100:.1f} "
                  f"genBB={r['gen_blackbox']['top1']*100:.1f} lat_cpu={r['latency_ms_cpu']:.3f}ms "
                  + " ".join(f"rho{k}={v['top1']*100:.1f}" for k, v in r["stale_gen"].items()), flush=True)
            json.dump(out, open(os.path.join(bd.ART, f"stage_amortized_{kind}.json"), "w"), indent=2)

    # Jakes mapping tau -> rho for the reporting table
    c = 3e8
    out["jakes"] = {}
    for fc in (3.5e9, 28e9):
        for vkmh in (5, 60):
            fD = (vkmh / 3.6) * fc / c
            out["jakes"][f"{fc/1e9:g}GHz_{vkmh}kmh"] = {
                "fD_Hz": fD, **{f"rho_tau_{t}ms": float(j0(2 * np.pi * fD * t * 1e-3)) for t in (0.5, 1, 2, 5, 10)}}
    json.dump(out, open(os.path.join(bd.ART, f"stage_amortized_{kind}.json"), "w"), indent=2)
    print(f"Saved stage_amortized_{kind}.json")


if __name__ == "__main__":
    main()
