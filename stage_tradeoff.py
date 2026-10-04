"""Revision: why learning is used, and what the attack does to that benefit.

Beam-management overhead/performance trade-off on the primary (partial-measurement)
setting: SE ratio vs number of beam measurements for
  * DNN + top-k refinement   : L wide-beam RSRPs + k narrow-beam measurements
  * classical local sweep    : L wide-beam RSRPs + w narrow beams under the best wide beam
  * exhaustive narrow sweep  : B measurements (64)
clean, under the DNN-aware malicious RIS (M=64, M=128), and a random RIS.
Also an aperture sweep M in {16..256} for the primary victim. 3 trials, noise-averaged.
"""
import os, json, numpy as np
import beamdata as bd
import ris
import revision_common as rc
from model import predict_probs

SEEDS = [42, 7, 2024]
KS = [1, 2, 3, 4, 6, 8]
WS = [1, 2, 4, 6, 8, 12]
M_SWEEP = [16, 32, 64, 128, 256]


def curves(v, H, W):
    """SE ratio for DNN top-k refinement and classical local sweep, noise-averaged."""
    B, L = bd.N_BEAMS, v.C.shape[0]
    s = B // L
    dnn = {k: [] for k in KS}; cl = {w: [] for w in WS}
    gains = np.abs(H.astype(np.complex128) @ W.conj().T) ** 2
    for n in rc.NOISE_SEEDS:
        Hn = bd.add_cn_noise(H, bd.PILOT_SNR_DB, np.random.default_rng(n))
        meas = np.abs(Hn.astype(np.complex128) @ W.conj().T) ** 2
        order = np.argsort(-predict_probs(v.model, v.feat_np(Hn)), 1)
        for k in KS:
            c = order[:, :k]
            pick = c[np.arange(len(c)), np.take_along_axis(meas, c, 1).argmax(1)]
            dnn[k].append(bd.se_ratio(H, W, pick))
        wide = np.abs(Hn.astype(np.complex128) @ v.C.conj().T) ** 2
        centre = wide.argmax(1) * s          # wide beam i points where narrow beam i*s does
        for w in WS:
            c = (centre[:, None] + np.arange(w)[None, :] - w // 2) % B
            pick = c[np.arange(len(c)), np.take_along_axis(meas, c, 1).argmax(1)]
            cl[w].append(bd.se_ratio(H, W, pick))
    return {"dnn": {k: float(np.mean(x)) for k, x in dnn.items()},
            "classical": {w: float(np.mean(x)) for w, x in cl.items()}}


def main():
    d = bd.build_dataset()
    H_te, W = d["H_test"], d["W"]
    out = {"seeds": SEEDS, "KS": KS, "WS": WS, "L": rc.L_WIDE, "tradeoff": [], "M_sweep": M_SWEEP, "sweep": []}
    for s in SEEDS:
        v = rc.train_victim("rsrp", d, seed=s)
        t = {"clean": curves(v, H_te, W)}
        for M in (64, 128):
            g = ris.build_geometry(H_te, M, 1.0, seed=s)
            t[f"attack_{M}"] = curves(v, rc.ris_attack(v, g, seed=s), W)
            t[f"random_{M}"] = curves(v, rc.random_ris(g, seed=s), W)
        out["tradeoff"].append(t)
        sw = {"attack": [], "random": []}
        for M in M_SWEEP:
            g = ris.build_geometry(H_te, M, 1.0, seed=s)
            sw["attack"].append(rc.evaluate_noise_avg(v, rc.ris_attack(v, g, seed=s), W))
            sw["random"].append(rc.evaluate_noise_avg(v, rc.random_ris(g, seed=s), W))
        sw["clean"] = rc.evaluate_noise_avg(v, H_te, W)
        out["sweep"].append(sw)
        print(f"[s={s}] clean dnn k3={t['clean']['dnn'][3]*100:.1f} cl w4={t['clean']['classical'][4]*100:.1f} | "
              f"M64 dnn k3={t['attack_64']['dnn'][3]*100:.1f} cl w4={t['attack_64']['classical'][4]*100:.1f} | "
              "sweep " + " ".join(f"{m}:{a['top1']*100:.1f}" for m, a in zip(M_SWEEP, sw['attack'])), flush=True)
    json.dump(out, open(os.path.join(bd.ART, "stage_tradeoff.json"), "w"), indent=2)
    print("Saved stage_tradeoff.json")


if __name__ == "__main__":
    main()
