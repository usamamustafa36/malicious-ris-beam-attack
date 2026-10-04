"""Shared helpers for the journal revision experiments.

Two victims are supported:
  * "rsrp" (primary): 3GPP AI/ML beam-management Case-1 style predictor. The BS sweeps
    L wide beams (Set B, formed by an N/s-element sub-array so the beamwidth matches the
    coarse spacing), the UE reports their RSRPs, and a DNN predicts the best of the
    B narrow DFT beams (Set A). Optionally the BS refines over the DNN's top-k beams.
    This is the setting where learning is actually needed: the model-free baseline
    (narrow beam nearest the best wide beam) is far worse (see stage_partial.py).
  * "csi" (secondary): the original full-CSI MLP of the conference version.

Evaluation draws imperfect-CSI noise from an explicit noise seed, so every number can
be averaged over independent noise realizations. The attacker never sees the
evaluation noise: it optimizes on the noiseless effective channel.
"""
import numpy as np, torch, torch.nn as nn
import beamdata as bd
from model import BeamMLP, train_model, predict_probs, DEVICE
import ris

N = bd.N_BS_ANT
L_WIDE = 16
NOISE_SEEDS = [42, 101, 202, 303, 404]
TRAIN_SEEDS = [42, 7, 2024, 1, 99]


# ---------------------------------------------------------------- victims / features
def wide_codebook(L=L_WIDE, n=N):
    """L wide beams from an (n*L/64)-element sub-array, spatial grid matched to L."""
    ang = np.linspace(-1, 1, L, endpoint=False)
    n_act = n * L // bd.N_BEAMS
    C = np.zeros((L, n), complex)
    C[:, :n_act] = np.exp(1j * np.pi * np.outer(ang, np.arange(n_act))) / np.sqrt(n_act)
    return C.astype(np.complex64)


class RSRPMLP(nn.Module):
    def __init__(self, in_dim, n_beams, hidden=(256, 256), drop=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
            d = h
        layers += [nn.Linear(d, n_beams)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def rsrp_np(Hn, C):
    p = np.abs(Hn.astype(np.complex128) @ C.conj().T) ** 2
    db = 10 * np.log10(p + 1e-30)
    return ((db - db.mean(1, keepdims=True)) / (db.std(1, keepdims=True) + 1e-8)).astype(np.float32)


def rsrp_torch(H, C_t):
    p = (H @ C_t.conj().T).abs() ** 2
    db = 10 * torch.log10(p + 1e-30)
    return ((db - db.mean(1, keepdim=True)) / (db.std(1, keepdim=True) + 1e-8)).float()


class Victim:
    """Bundles a trained model with its numpy/torch feature maps."""

    def __init__(self, kind, model, C=None):
        self.kind, self.model, self.C = kind, model, C
        if kind == "rsrp":
            self.C_t = torch.as_tensor(C, dtype=torch.complex64, device=DEVICE)

    def feat_np(self, Hn):
        return rsrp_np(Hn, self.C) if self.kind == "rsrp" else bd.complex_to_feat(Hn)

    def feat_t(self, H):
        return rsrp_torch(H, self.C_t) if self.kind == "rsrp" else ris._feats_torch(H)


def train_victim(kind, d, seed, hidden=None, epochs=60, snr_db=bd.PILOT_SNR_DB, extra=None, L=L_WIDE):
    """Train a victim on clean training channels plus optional extra channel sets
    (e.g. RIS-perturbed, for adversarial training), each labeled by its own best
    beam; input noise at snr_db. L sets the number of wide beams of the rsrp victim."""
    rng = np.random.default_rng(seed)
    W = d["W"]
    lab = lambda H: np.abs(H.astype(np.complex128) @ W.conj().T).argmax(1)
    if kind == "rsrp":
        C = wide_codebook(L)
        f = lambda H: rsrp_np(bd.add_cn_noise(H, snr_db, rng), C)
        model = RSRPMLP(L, d["n_beams"], hidden=hidden or (256, 256))
    else:
        C = None
        f = lambda H: bd.complex_to_feat(bd.add_cn_noise(H, snr_db, rng))
        model = BeamMLP(2 * N, d["n_beams"], hidden=hidden or (256, 256, 128))
    Htr = [d["H_tr"]] + ([e for e in extra] if extra else [])
    Xtr = np.concatenate([f(H) for H in Htr]); ytr = np.concatenate([lab(H) for H in Htr])
    m = train_model(Xtr, ytr, f(d["H_va"]), lab(d["H_va"]), d["n_beams"],
                    epochs=epochs, seed=seed, model=model, verbose=False)
    return Victim(kind, m, C)


# ---------------------------------------------------------------- attacks
def soft_objective(victim, H, Wc):
    """Achieved/optimal beamforming-gain ratio under the victim's soft decision."""
    p = torch.softmax(victim.model(victim.feat_t(H)), dim=1)
    gains = (H @ Wc.T).abs() ** 2
    opt_gain = gains.max(dim=1, keepdim=True).values.detach()
    return (p * gains).sum(1, keepdim=True) / (opt_gain + 1e-30)


def ris_attack(victim, g, iters=80, lr=0.1, bbit=None, seed=bd.SEED, return_theta=False):
    """Per-user iterative RIS-phase attack (generalizes ris.ris_attack to any victim)."""
    victim.model.eval(); torch.manual_seed(seed)
    U, M = g["H_d"].shape[0], g["M"]
    theta = (torch.rand(U, M, device=DEVICE) * 2 * np.pi).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    Wc = g["W"].conj()
    for _ in range(iters):
        opt.zero_grad()
        soft_objective(victim, ris._h_eff(g, theta), Wc).mean().backward()
        opt.step()
    with torch.no_grad():
        if bbit is not None:
            step = 2 * np.pi / (2 ** bbit)
            theta.data = torch.round(theta.data / step) * step
        heff = ris._h_eff(g, theta).cpu().numpy()
    return (heff, theta.detach()) if return_theta else heff


def random_ris(g, seed=bd.SEED):
    torch.manual_seed(seed)
    th = torch.rand(g["H_d"].shape[0], g["M"], device=DEVICE) * 2 * np.pi
    return ris._h_eff(g, th).detach().cpu().numpy()


# ---------------------------------------------------------------- evaluation
def evaluate(victim, H_eff, W, noise_seed, k=3, snr_db=bd.PILOT_SNR_DB):
    """Victim predicts from a noisy estimate of H_eff (noise_seed); scored on the true
    perturbed channel. Returns top-1, top-k hit, SE ratio of top-1, and SE ratio after
    the BS refines over the DNN's top-k narrow beams using noisy measurements."""
    Hn = bd.add_cn_noise(H_eff, snr_db, np.random.default_rng(noise_seed))
    probs = predict_probs(victim.model, victim.feat_np(Hn))
    topk = np.argsort(-probs, axis=1)[:, :k]
    gains = np.abs(H_eff.astype(np.complex128) @ W.conj().T) ** 2
    y = gains.argmax(1)
    meas = np.abs(Hn.astype(np.complex128) @ W.conj().T) ** 2
    refined = topk[np.arange(len(y)), np.take_along_axis(meas, topk, 1).argmax(1)]
    return {"top1": float(np.mean(topk[:, 0] == y)),
            f"top{k}": float(np.mean((topk == y[:, None]).any(1))),
            "se": bd.se_ratio(H_eff, W, topk[:, 0]),
            f"se_ref{k}": bd.se_ratio(H_eff, W, refined)}


def evaluate_noise_avg(victim, H_eff, W, seeds=NOISE_SEEDS, snr_db=bd.PILOT_SNR_DB):
    rs = [evaluate(victim, H_eff, W, s, snr_db=snr_db) for s in seeds]
    return {key: float(np.mean([r[key] for r in rs])) for key in rs[0]}


def ci95(xs):
    """Mean and half-width of a two-sided 95% Student-t interval."""
    from scipy import stats
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    h = stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs))
    return float(xs.mean()), float(h)
