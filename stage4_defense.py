"""Stage 4: defenses vs the malicious RIS.

(A) RIS-augmented adversarial training: retrain the beam predictor on a mix of
    clean + malicious-RIS + random-RIS effective channels (labels = true best beam
    of each channel). Robustness is measured with an ADAPTIVE attack regenerated
    against the defended model (white-box), and with a 3-seed confidence interval.
(B) Lightweight detector: a small binary net that flags RIS-perturbed CSI.
"""
import os, json, numpy as np, torch, torch.nn as nn
import beamdata as bd
from model import BeamMLP, train_model, predict_probs, to_t, DEVICE
import ris

KAPPA, rng = 1.0, np.random.default_rng(bd.SEED)


def best_beam(H, W):
    return np.abs(H.astype(np.complex128) @ W.conj().T).argmax(1)


def noisy_feats(H):
    return bd.complex_to_feat(bd.add_cn_noise(H, bd.PILOT_SNR_DB, rng))


def eval_under_attack(model, H_d, W, M, seed):
    g = ris.build_geometry(H_d, M, KAPPA, seed=seed)
    H_adv = ris.ris_attack(model, g, iters=80, lr=0.1, seed=seed)
    pred = predict_probs(model, bd.eval_feats(H_adv)).argmax(1)   # deterministic eval noise
    y_eff = best_beam(H_adv, W)
    return float(np.mean(pred == y_eff)), bd.se_ratio(H_adv, W, pred)


def build_augmented(victim, H_tr, W):
    """clean + malicious-RIS(M64,M128 vs victim) + random-RIS, with per-channel labels."""
    Xs, ys = [bd.complex_to_feat(bd.add_cn_noise(H_tr, bd.PILOT_SNR_DB, rng))], [best_beam(H_tr, W)]
    for M in (64, 128):
        g = ris.build_geometry(H_tr, M, KAPPA, seed=7)
        H_adv = ris.ris_attack(victim, g, iters=60, lr=0.1, seed=7)
        Xs.append(noisy_feats(H_adv)); ys.append(best_beam(H_adv, W))
        torch.manual_seed(bd.SEED + M)   # seeded random-RIS augmentation (reproducible defended model)
        H_rand = ris._h_eff(g, torch.rand(H_tr.shape[0], M, device=DEVICE) * 2 * np.pi).cpu().numpy()
        Xs.append(noisy_feats(H_rand)); ys.append(best_beam(H_rand, W))
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys)


class Detector(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_detector(Xc, Xa):
    X = np.concatenate([Xc, Xa]).astype(np.float32)
    y = np.concatenate([np.zeros(len(Xc)), np.ones(len(Xa))]).astype(np.float32)
    det = Detector(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(det.parameters(), 1e-3); bce = nn.BCEWithLogitsLoss()
    Xt, yt = to_t(X), to_t(y)
    for _ in range(40):
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), 512):
            idx = perm[i:i + 512]; opt.zero_grad()
            bce(det(Xt[idx]), yt[idx]).backward(); opt.step()
    return det


def auc(det, Xc, Xa):
    with torch.no_grad():
        sc = torch.sigmoid(det(to_t(np.concatenate([Xc, Xa]).astype(np.float32)))).cpu().numpy()
    lbl = np.concatenate([np.zeros(len(Xc)), np.ones(len(Xa))])
    order = np.argsort(sc); ranks = np.empty_like(order); ranks[order] = np.arange(len(sc))
    npos, nneg = lbl.sum(), (1 - lbl).sum()
    return float((ranks[lbl == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg))


def main():
    d = bd.build_dataset()
    H_tr, H_te, W = d["H_tr"], d["H_test"], d["W"]
    victim = BeamMLP(d["Xte"].shape[1], d["n_beams"]).to(DEVICE)
    victim.load_state_dict(torch.load(os.path.join(bd.ART, "victim_model.pt"))); victim.eval()

    print("Building RIS-augmented training set (attacking baseline victim)...")
    Xaug, yaug = build_augmented(victim, H_tr, W)
    print(f"  augmented train size: {len(yaug)} (from {len(H_tr)} clean)")
    defended = train_model(Xaug, yaug, d["Xva"], d["yva"], d["n_beams"], epochs=40, verbose=False)
    torch.save(defended.state_dict(), os.path.join(bd.ART, "defended_model.pt"))

    # clean performance (both models)
    res = {"clean": {}, "attack_Msweep": {"M": [64, 128], "undefended": [], "defended": []}}
    for name, mdl in [("undefended", victim), ("defended", defended)]:
        p = predict_probs(mdl, d["Xte"]).argmax(1)
        res["clean"][name] = {"top1": float(np.mean(p == d["yte"])), "se": bd.se_ratio(H_te, W, p)}
    print(f"\nClean top-1  undef={res['clean']['undefended']['top1']*100:.1f}%  "
          f"def={res['clean']['defended']['top1']*100:.1f}%")

    print("\nAdaptive RIS attack (regenerated vs each model):")
    print(f"{'M':>4} | {'undef top1':>10} {'undef SE':>9} | {'def top1':>9} {'def SE':>7}")
    for M in (64, 128):
        u1, uSE = eval_under_attack(victim, H_te, W, M, seed=bd.SEED)
        d1, dSE = eval_under_attack(defended, H_te, W, M, seed=bd.SEED)
        res["attack_Msweep"]["undefended"].append({"top1": u1, "se": uSE})
        res["attack_Msweep"]["defended"].append({"top1": d1, "se": dSE})
        print(f"{M:4d} | {u1*100:9.1f}% {uSE*100:8.1f}% | {d1*100:8.1f}% {dSE*100:6.1f}%")

    # 3-seed CI on the headline (M=128)
    us = [eval_under_attack(victim, H_te, W, 128, s)[0] for s in (42, 1, 2)]
    ds = [eval_under_attack(defended, H_te, W, 128, s)[0] for s in (42, 1, 2)]
    res["ci_M128"] = {"undef_mean": float(np.mean(us)), "undef_std": float(np.std(us)),
                      "def_mean": float(np.mean(ds)), "def_std": float(np.std(ds))}
    print(f"\nM=128 top-1 over 3 seeds:  undef={np.mean(us)*100:.1f}±{np.std(us)*100:.1f}%  "
          f"def={np.mean(ds)*100:.1f}±{np.std(ds)*100:.1f}%")

    # detector (eval side uses the deterministic eval-noise realization; the detector's
    # TRAINING data below stays stochastic via noisy_feats)
    g = ris.build_geometry(H_te, 128, KAPPA, seed=99)
    Xc = d["Xte"]
    Xa_te = bd.eval_feats(ris.ris_attack(victim, g, iters=80, seed=99))
    gtr = ris.build_geometry(H_tr, 128, KAPPA, seed=5)
    det = train_detector(d["Xtr"], noisy_feats(ris.ris_attack(victim, gtr, iters=60, seed=5)))
    res["detector_auc"] = auc(det, Xc, Xa_te)
    print(f"\nRIS-perturbation detector AUC (test): {res['detector_auc']:.3f}")

    json.dump(res, open(os.path.join(bd.ART, "stage4_defense.json"), "w"), indent=2)
    print("\nSaved stage4_defense.json")


if __name__ == "__main__":
    main()
