"""Revision: detector-aware (adaptive) attack against the PRIMARY partial-measurement victim.

The defender trains a detector to separate clean from malicious-RIS wide-beam RSRPs
(as in stage_rsrp_generality.py). The attacker then minimizes
    J(theta) + lam * sigmoid(detector(features)),
trading damage against evasion, for lam in LAMS. We report victim top-1 / eta_3 and the
detector's AUC and true-positive rate at 5% false alarms on clean users, and its false
alarm rate on a benign random RIS. Finally the defender retrains its detector on the
evasive attacks (one round of the arms race) and we re-measure. 3 trials, M = 64, 128.
"""
import os, json, numpy as np, torch
import beamdata as bd
import ris
import revision_common as rc
from model import DEVICE, to_t
from stage_rsrp_generality import train_detector
from stage4_defense import auc

SEEDS = [42, 7, 2024]
LAMS = [0.0, 0.5, 2.0, 8.0]
M_LIST = [64, 128]


def detector_aware_attack(v, det, g, lam, iters=120, lr=0.1, seed=bd.SEED):
    v.model.eval(); det.eval(); torch.manual_seed(seed)
    theta = (torch.rand(g["H_d"].shape[0], g["M"], device=DEVICE) * 2 * np.pi).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    Wc = g["W"].conj()
    for _ in range(iters):
        opt.zero_grad()
        H = ris._h_eff(g, theta)
        loss = rc.soft_objective(v, H, Wc).mean() + lam * torch.sigmoid(det(v.feat_t(H))).mean()
        loss.backward(); opt.step()
    return ris._h_eff(g, theta).detach().cpu().numpy()


@torch.no_grad()
def scores(det, X):
    return torch.sigmoid(det(to_t(X.astype(np.float32)))).cpu().numpy()


def det_metrics(det, Xc, Xa, Xr):
    sc, sa, sr = scores(det, Xc), scores(det, Xa), scores(det, Xr)
    thr = np.quantile(sc, 0.95)                         # 5% false alarms on clean users
    return {"auc": auc(det, Xc, Xa), "tpr_at_fpr5": float(np.mean(sa > thr)),
            "benign_false_alarm": float(np.mean(sr > thr))}


def main():
    d = bd.build_dataset()
    H_tr, H_te, W = d["H_tr"], d["H_test"], d["W"]
    rows = []
    for s in SEEDS:
        v = rc.train_victim("rsrp", d, seed=s)
        noisy = lambda H, k: v.feat_np(bd.add_cn_noise(H, bd.PILOT_SNR_DB, np.random.default_rng(s + k)))
        gtr = ris.build_geometry(H_tr, 128, 1.0, seed=s + 900)
        det = train_detector(noisy(H_tr, 0), noisy(rc.ris_attack(v, gtr, iters=60, seed=s), 1), s)
        Xc = v.feat_np(bd.eval_noise(H_te))
        r = {"clean": rc.evaluate_noise_avg(v, H_te, W)}
        for M in M_LIST:
            g = ris.build_geometry(H_te, M, 1.0, seed=s)
            Xr = v.feat_np(bd.eval_noise(rc.random_ris(g, seed=s)))
            for lam in LAMS:
                Ha = detector_aware_attack(v, det, g, lam, seed=s)
                r[f"M{M}_lam{lam}"] = {**rc.evaluate_noise_avg(v, Ha, W),
                                        **det_metrics(det, Xc, v.feat_np(bd.eval_noise(Ha)), Xr)}
            # one arms-race round: defender retrains on evasive attacks (lam=2) crafted on training users
            gtr2 = ris.build_geometry(H_tr, M, 1.0, seed=s + 950 + M)
            det2 = train_detector(noisy(H_tr, 2), noisy(detector_aware_attack(v, det, gtr2, 2.0, iters=60, seed=s), 3), s + 1)
            Ha = detector_aware_attack(v, det, g, 2.0, seed=s)            # same evasive attack, new detector
            r[f"M{M}_retrained"] = {**rc.evaluate_noise_avg(v, Ha, W),
                                    **det_metrics(det2, Xc, v.feat_np(bd.eval_noise(Ha)), Xr)}
            Ha2 = detector_aware_attack(v, det2, g, 2.0, seed=s)          # attacker adapts again
            r[f"M{M}_retrained_adapt"] = {**rc.evaluate_noise_avg(v, Ha2, W),
                                          **det_metrics(det2, Xc, v.feat_np(bd.eval_noise(Ha2)), Xr)}
        rows.append(r)
        print(f"[s={s}] clean top1={r['clean']['top1']*100:.1f} | " + " | ".join(
            f"{k}: top1={x['top1']*100:.1f} se3={x['se_ref3']*100:.1f} auc={x['auc']:.3f} tpr5={x['tpr_at_fpr5']*100:.0f} fa={x['benign_false_alarm']*100:.0f}"
            for k, x in r.items() if k != "clean"), flush=True)
    out = {"seeds": SEEDS, "lams": LAMS,
           "summary": {k: {m: rc.ci95([rr[k][m] for rr in rows]) for m in rows[0][k]} for k in rows[0]},
           "trials": rows}
    json.dump(out, open(os.path.join(bd.ART, "stage_detector_rsrp.json"), "w"), indent=2)
    print("Saved stage_detector_rsrp.json")


if __name__ == "__main__":
    main()
