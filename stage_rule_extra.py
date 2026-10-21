"""DNN decision threshold on two more victims. Wide-beam flips do not depend on the victim.

Does not overwrite artifacts/stage_rule.json (seed 42).
"""
import json, os
import numpy as np
import beamdata as bd
import revision_common as rc
import stage_rule as sr
from model import predict_probs

SEEDS = [7, 2024]


def one(seed, d, H, C, a, hnorm, far, clean_wide):
    print("training", seed, flush=True)
    v = rc.train_victim("rsrp", d, seed=seed)
    phases = np.linspace(0, 2 * np.pi, sr.N_PHASE, endpoint=False)
    U = H.shape[0]
    clean_pred = predict_probs(v.model, v.feat_np(H.astype(np.complex64))).argmax(1)
    gamma_th = np.full(U, 40.0)
    pred_changed = {}
    for ratio in sr.RATIOS:
        amp = hnorm * 10 ** (ratio / 20.0)
        best_loss = np.full(U, np.inf)
        best_pred = np.zeros(U, dtype=int)
        for p in phases:
            beta = amp * np.exp(1j * p)
            He = (H + beta[:, None] * a[None, :]).astype(np.complex64)
            probs = predict_probs(v.model, v.feat_np(He))
            pred = probs.argmax(1)
            gains = np.abs(He.astype(np.complex128) @ d["W"].conj().T) ** 2
            opt = gains.max(1)
            achieved = (probs * gains).sum(1) / (opt + 1e-30)
            take = achieved < best_loss
            best_loss[take] = achieved[take]
            best_pred[take] = pred[take]
        flip = best_pred != clean_pred
        gamma_th[flip] = np.minimum(gamma_th[flip], ratio)
        pred_changed[str(ratio)] = float(np.mean(flip))
        print(f"  s={seed} {ratio:+5.0f} pred-changed {pred_changed[str(ratio)]*100:4.1f}", flush=True)
    def summ(x):
        return {"median": float(np.median(x)), "p25": float(np.percentile(x, 25)),
                "p75": float(np.percentile(x, 75))}
    return {"seed": seed, "gamma_theta": summ(gamma_th),
            "gamma_theta_far": summ(gamma_th[far]), "gamma_theta_near": summ(gamma_th[~far]),
            "pred_changed": pred_changed,
            "clean_top1_noiseless": float(np.mean(clean_pred == np.abs(H @ d["W"].conj().T).argmax(1)))}


def main():
    d = bd.build_dataset()
    H = d["H_test"].astype(np.complex128)
    C = rc.wide_codebook(rc.L_WIDE)
    a = np.exp(1j * np.pi * np.arange(H.shape[1]) * sr.U_RIS) / np.sqrt(H.shape[1])
    hnorm = np.linalg.norm(H, axis=1)
    ang = np.linspace(-1, 1, bd.N_BEAMS, endpoint=False)
    u_user = ang[np.argmax(np.abs(H @ d["W"].conj().T) ** 2, axis=1)]
    gap = np.abs(u_user - sr.U_RIS)
    gap = np.minimum(gap, 2 - gap)
    far = gap > (2.0 / rc.L_WIDE)
    # wide-beam geometry is victim-independent; recompute once as a sanity check
    g, r = sr.wide_stats(H, C, a)
    clean_wide = np.abs(g).argmax(1)
    out = [one(s, d, H, C, a, hnorm, far, clean_wide) for s in SEEDS]
    path = os.path.join(bd.ART, "stage_rule_extra.json")
    json.dump(out, open(path, "w"), indent=2)
    print("saved", path)
    for row in out:
        print(row["seed"], row["gamma_theta"])


if __name__ == "__main__":
    main()
