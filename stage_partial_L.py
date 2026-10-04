"""Revision (R3.4): partial-measurement victim with L in {8, 16, 32} wide beams.

For each L the victim is retrained, and we report, without RIS, under a random RIS and
under the malicious RIS (M=64, M=128): DNN top-1 / top-3 / SE ratio after top-k
refinement, and the model-free local sweep of the s=64/L narrow beams under the strongest
wide beam (L+s measurements; a centred 3-beam window when s=2), together with the DNN at
the same budget (k equal to the window).
3 trials (victim seed = geometry seed = attack seed), noise-averaged scoring.
"""
import os, json, numpy as np
import beamdata as bd
import ris
import revision_common as rc
from model import predict_probs

SEEDS = [42, 7, 2024]
LS = [8, 16, 32]


def score(v, H, W):
    """Noise-averaged DNN (k=1,3,s) and local-sweep metrics on channels H."""
    B, L = bd.N_BEAMS, v.C.shape[0]
    s = B // L
    w = max(s, 3)                                   # local window (centred 3 beams if s=2)
    ks = sorted({1, 3, w})
    gains = np.abs(H.astype(np.complex128) @ W.conj().T) ** 2
    y = gains.argmax(1)
    acc = {f"dnn_top{k}": [] for k in ks} | {f"dnn_se{k}": [] for k in ks} | {"local_top1": [], "local_se": []}
    for n in rc.NOISE_SEEDS:
        Hn = bd.add_cn_noise(H, bd.PILOT_SNR_DB, np.random.default_rng(n))
        meas = np.abs(Hn.astype(np.complex128) @ W.conj().T) ** 2
        order = np.argsort(-predict_probs(v.model, v.feat_np(Hn)), 1)
        for k in ks:
            c = order[:, :k]
            pick = c[np.arange(len(c)), np.take_along_axis(meas, c, 1).argmax(1)]
            acc[f"dnn_top{k}"].append(float(np.mean((c == y[:, None]).any(1))))
            acc[f"dnn_se{k}"].append(bd.se_ratio(H, W, pick))
        wide = np.abs(Hn.astype(np.complex128) @ v.C.conj().T) ** 2
        cand = (wide.argmax(1)[:, None] * s + np.arange(-(w // 2), w - w // 2)[None, :]) % B
        loc = cand[np.arange(len(y)), np.take_along_axis(meas, cand, 1).argmax(1)]
        acc["local_top1"].append(float(np.mean(loc == y)))
        acc["local_se"].append(bd.se_ratio(H, W, loc))
    return {k: float(np.mean(x)) for k, x in acc.items()}


def main():
    d = bd.build_dataset()
    H_te, W = d["H_test"], d["W"]
    out = {"seeds": SEEDS, "Ls": LS, "trials": {}}
    for L in LS:
        rows = []
        for s in SEEDS:
            v = rc.train_victim("rsrp", d, seed=s, L=L)
            r = {"clean": score(v, H_te, W)}
            for M in (64, 128):
                g = ris.build_geometry(H_te, M, 1.0, seed=s)
                r[f"random_{M}"] = score(v, rc.random_ris(g, seed=s), W)
                r[f"attack_{M}"] = score(v, rc.ris_attack(v, g, seed=s), W)
            rows.append(r)
            print(f"[L={L} s={s}] clean top1={r['clean']['dnn_top1']*100:.1f} se3={r['clean']['dnn_se3']*100:.1f} "
                  f"loc={r['clean']['local_se']*100:.1f} | M64 top1={r['attack_64']['dnn_top1']*100:.1f} "
                  f"se3={r['attack_64']['dnn_se3']*100:.1f} loc={r['attack_64']['local_se']*100:.1f} | "
                  f"M128 top1={r['attack_128']['dnn_top1']*100:.1f} se3={r['attack_128']['dnn_se3']*100:.1f} "
                  f"loc={r['attack_128']['local_se']*100:.1f}", flush=True)
        out["trials"][str(L)] = rows
        out.setdefault("summary", {})[str(L)] = {
            c: {m: rc.ci95([r[c][m] for r in rows]) for m in rows[0][c]} for c in rows[0]}
        json.dump(out, open(os.path.join(bd.ART, "stage_partial_L.json"), "w"), indent=2)
    print("Saved stage_partial_L.json")


if __name__ == "__main__":
    main()
