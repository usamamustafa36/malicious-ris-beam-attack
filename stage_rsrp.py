"""Review-response: victim trained on 3GPP-style coarse-beam RSRP measurements
(an L-beam partial sweep) instead of raw [Re,Im] CSI, testing whether the
malicious-RIS attack still transfers to a victim with a measurement model
closer to what beam-management study items (3GPP AI/ML for beam management)
actually specify, rather than a raw CSI vector.
"""
import os, json, numpy as np, torch, torch.nn as nn
import beamdata as bd
from model import train_model, predict_probs, DEVICE
import ris

L_COARSE = 16
SEED = bd.SEED
rng = np.random.default_rng(SEED)


class RSRPMLP(nn.Module):
    def __init__(self, in_dim, n_beams, hidden=(128, 128), drop=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)]
            d = h
        layers += [nn.Linear(d, n_beams)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _rsrp_np(Hn, Wc):
    p = np.abs(Hn.astype(np.complex128) @ Wc.conj().T) ** 2
    db = 10 * np.log10(p + 1e-30)
    return ((db - db.mean(1, keepdims=True)) / (db.std(1, keepdims=True) + 1e-8)).astype(np.float32)


def _rsrp_torch(Ht, Wc_t):
    p = (Ht @ Wc_t.conj().T).abs() ** 2
    db = 10 * torch.log10(p + 1e-30)
    return ((db - db.mean(1, keepdim=True)) / (db.std(1, keepdim=True) + 1e-8)).float()


def attack_rsrp(model, g, Wc_t, iters=80, lr=0.1, seed=SEED):
    """Malicious-RIS attack against an RSRP-input victim (mirrors ris.ris_attack,
    swapping the raw-CSI feature for the L-beam coarse RSRP feature)."""
    model.eval(); torch.manual_seed(seed)
    U, M = g["H_d"].shape[0], g["M"]
    theta = (torch.rand(U, M, device=DEVICE) * 2 * np.pi).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    Wc = g["W"].conj()
    for _ in range(iters):
        opt.zero_grad()
        H = ris._h_eff(g, theta)
        p = torch.softmax(model(_rsrp_torch(H, Wc_t)), dim=1)
        gains = (H @ Wc.T).abs() ** 2
        opt_gain = gains.max(dim=1, keepdim=True).values.detach()
        achieved = (p * gains).sum(dim=1, keepdim=True)
        loss = (achieved / (opt_gain + 1e-30)).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        heff = ris._h_eff(g, theta).detach().cpu().numpy()
    return heff


def eval_rsrp(model, H_eff, W, Wc):
    Hn = bd.eval_noise(H_eff, bd.PILOT_SNR_DB)   # shared deterministic eval-noise realization
    Xn = _rsrp_np(Hn, Wc)
    pred = predict_probs(model, Xn).argmax(1)
    gains = np.abs(H_eff.astype(np.complex128) @ W.conj().T) ** 2
    y_eff = gains.argmax(1)
    return {"top1": float(np.mean(pred == y_eff)), "se_ratio": bd.se_ratio(H_eff, W, pred)}


def main():
    d = bd.build_dataset()
    H_tr, H_va, H_te, W = d["H_tr"], d["H_va"], d["H_test"], d["W"]
    ang = np.linspace(-1, 1, L_COARSE, endpoint=False)
    Wc = (np.exp(1j * np.pi * np.outer(ang, np.arange(bd.N_BS_ANT))) / np.sqrt(bd.N_BS_ANT)).astype(np.complex64)

    Xtr = _rsrp_np(bd.add_cn_noise(H_tr, bd.PILOT_SNR_DB, rng), Wc)
    ytr = np.abs(H_tr.astype(np.complex128) @ W.conj().T).argmax(1)
    Xva = _rsrp_np(bd.add_cn_noise(H_va, bd.PILOT_SNR_DB, rng), Wc)
    yva = np.abs(H_va.astype(np.complex128) @ W.conj().T).argmax(1)

    rsrp_model = RSRPMLP(L_COARSE, d["n_beams"])
    rsrp_model = train_model(Xtr, ytr, Xva, yva, d["n_beams"], epochs=60, seed=SEED,
                             model=rsrp_model, verbose=False)
    torch.save(rsrp_model.state_dict(), os.path.join(bd.ART, "rsrp_victim.pt"))

    clean = eval_rsrp(rsrp_model, H_te, W, Wc)
    print(f"RSRP victim clean: top1={clean['top1']*100:.1f}%  SE={clean['se_ratio']*100:.1f}%")

    out = {"L_coarse": L_COARSE, "clean": clean, "attack": {}, "random": {}}
    Wc_t = torch.as_tensor(Wc, dtype=torch.complex64, device=DEVICE)
    for M in (64, 128):
        g = ris.build_geometry(H_te, M, 1.0, seed=SEED)
        H_adv = attack_rsrp(rsrp_model, g, Wc_t, iters=80, seed=SEED)
        adv = eval_rsrp(rsrp_model, H_adv, W, Wc)
        torch.manual_seed(bd.SEED)   # seeded random-RIS control
        H_rand = ris._h_eff(g, torch.rand(len(H_te), M, device=DEVICE) * 2 * np.pi).detach().cpu().numpy()
        rnd = eval_rsrp(rsrp_model, H_rand, W, Wc)
        out["attack"][str(M)] = adv; out["random"][str(M)] = rnd
        print(f"  M={M}: malicious top1={adv['top1']*100:5.1f}% SE={adv['se_ratio']*100:4.0f}% |"
              f" random top1={rnd['top1']*100:5.1f}%")

    json.dump(out, open(os.path.join(bd.ART, "stage_rsrp.json"), "w"), indent=2)
    print("Saved stage_rsrp.json")


if __name__ == "__main__":
    main()
