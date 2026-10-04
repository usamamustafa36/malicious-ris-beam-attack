"""Revision: generality and defense results for the PRIMARY (partial-measurement) victim,
with 3 trials each (victim seed = geometry seed = attack seed) and noise-averaged scoring.

  * discrete RIS phases (1/2/3-bit) at M=64 and M=128
  * operating pilot SNR 0/10/20 dB (victim retrained at each SNR)
  * 28 GHz mmWave scenario (city_0_newyork_28)
  * RIS-adversarial training under an adaptive attack, and a perturbation detector
"""
import os, json, numpy as np, torch
import beamdata as bd
import ris
import revision_common as rc
from stage_ci import classical_rsrp, avg
from stage4_defense import Detector, auc
from model import to_t, DEVICE
import torch.nn as nn

SEEDS = [42, 7, 2024]


def summarize(rows):
    keys = rows[0].keys()
    return {k: {m: rc.ci95([r[k][m] for r in rows]) for m in rows[0][k]} for k in keys}


def train_detector(Xc, Xa, seed):
    torch.manual_seed(seed)
    X = np.concatenate([Xc, Xa]).astype(np.float32)
    y = np.concatenate([np.zeros(len(Xc)), np.ones(len(Xa))]).astype(np.float32)
    det = Detector(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(det.parameters(), 1e-3); bce = nn.BCEWithLogitsLoss()
    Xt, yt = to_t(X), to_t(y)
    for _ in range(30):
        perm = torch.randperm(len(Xt), device=DEVICE)
        for i in range(0, len(Xt), 512):
            idx = perm[i:i + 512]; opt.zero_grad()
            bce(det(Xt[idx]), yt[idx]).backward(); opt.step()
    det.eval()
    return det


def main():
    d = bd.build_dataset()
    H_tr, H_te, W = d["H_tr"], d["H_test"], d["W"]
    out = {"seeds": SEEDS}

    # ---- discrete phases + defense + detector (3.5 GHz)
    rows_b, rows_def = [], []
    for s in SEEDS:
        v = rc.train_victim("rsrp", d, seed=s)
        rb = {}
        for M in (64, 128):
            g = ris.build_geometry(H_te, M, 1.0, seed=s)
            for b in (1, 2, 3, None):
                rb[f"M{M}_{'cont' if b is None else str(b)+'bit'}"] = rc.evaluate_noise_avg(
                    v, rc.ris_attack(v, g, bbit=b, seed=s), W)
        rows_b.append(rb)

        # RIS-adversarial training: perturbed training channels crafted against the victim
        extra = []
        for M in (64, 128):
            gtr = ris.build_geometry(H_tr, M, 1.0, seed=s + 300 + M)
            extra.append(rc.ris_attack(v, gtr, iters=60, seed=s + M))
            extra.append(rc.random_ris(gtr, seed=s + M))
        dfd = rc.train_victim("rsrp", d, seed=s, extra=extra, epochs=40)
        rd = {"undef_clean": rc.evaluate_noise_avg(v, H_te, W),
              "def_clean": rc.evaluate_noise_avg(dfd, H_te, W)}
        for M in (64, 128):
            g = ris.build_geometry(H_te, M, 1.0, seed=s)
            rd[f"undef_M{M}"] = rc.evaluate_noise_avg(v, rc.ris_attack(v, g, seed=s), W)
            rd[f"def_adaptive_M{M}"] = rc.evaluate_noise_avg(dfd, rc.ris_attack(dfd, g, seed=s), W)
            rd[f"def_adaptive250_M{M}"] = rc.evaluate_noise_avg(dfd, rc.ris_attack(dfd, g, iters=250, seed=s), W)
        # detector: clean vs malicious (test), and malicious vs benign random RIS
        gtr = ris.build_geometry(H_tr, 128, 1.0, seed=s + 900)
        det = train_detector(v.feat_np(bd.add_cn_noise(H_tr, 10, np.random.default_rng(s))),
                             v.feat_np(bd.add_cn_noise(rc.ris_attack(v, gtr, iters=60, seed=s), 10,
                                                       np.random.default_rng(s + 1))), s)
        g = ris.build_geometry(H_te, 128, 1.0, seed=s)
        Xc = v.feat_np(bd.eval_noise(H_te)); Xa = v.feat_np(bd.eval_noise(rc.ris_attack(v, g, seed=s)))
        Xr = v.feat_np(bd.eval_noise(rc.random_ris(g, seed=s)))
        rd["detector"] = {"auc_clean_vs_mal": auc(det, Xc, Xa), "auc_benign_vs_mal": auc(det, Xr, Xa)}
        rows_def.append(rd)
        print(f"[s={s}] bbit M128: " + " ".join(f"{k}={x['top1']*100:.1f}" for k, x in rb.items() if 'M128' in k)
              + f" | def clean={rd['def_clean']['top1']*100:.1f} adaptive128={rd['def_adaptive_M128']['top1']*100:.1f}"
              f" seRef={rd['def_adaptive_M128']['se_ref3']*100:.1f} | det={rd['detector']}", flush=True)
    out["bbit"] = summarize(rows_b)
    out["defense"] = {k: v for k, v in summarize([{k: v for k, v in r.items() if k != "detector"} for r in rows_def]).items()}
    out["detector"] = {m: rc.ci95([r["detector"][m] for r in rows_def]) for m in rows_def[0]["detector"]}

    # ---- pilot SNR sweep
    out["snr"] = {}
    for snr in (0.0, 10.0, 20.0):
        rows = []
        for s in SEEDS:
            v = rc.train_victim("rsrp", d, seed=s, snr_db=snr)
            r = {"clean": rc.evaluate_noise_avg(v, H_te, W, snr_db=snr)}
            for M in (64, 128):
                g = ris.build_geometry(H_te, M, 1.0, seed=s)
                r[f"attack_{M}"] = rc.evaluate_noise_avg(v, rc.ris_attack(v, g, seed=s), W, snr_db=snr)
            rows.append(r)
        out["snr"][str(snr)] = summarize(rows)
        print(f"SNR {snr}: " + " ".join(f"{k}={v['top1'][0]*100:.1f}" for k, v in out['snr'][str(snr)].items()), flush=True)

    # ---- 28 GHz mmWave
    dm = bd.build_dataset(scenario="city_0_newyork_28", cache_name="channels_mmw.npy")
    rows = []
    for s in SEEDS:
        v = rc.train_victim("rsrp", dm, seed=s)
        cls = lambda H: avg([classical_rsrp(H, dm["W"], v.C, n) for n in rc.NOISE_SEEDS])
        r = {"clean": rc.evaluate_noise_avg(v, dm["H_test"], dm["W"]), "clean_classical": cls(dm["H_test"])}
        for M in (64, 128):
            g = ris.build_geometry(dm["H_test"], M, 1.0, seed=s)
            Ha = rc.ris_attack(v, g, seed=s)
            r[f"attack_{M}"] = rc.evaluate_noise_avg(v, Ha, dm["W"])
            r[f"attack_{M}_classical"] = cls(Ha)
            r[f"random_{M}"] = rc.evaluate_noise_avg(v, rc.random_ris(g, seed=s), dm["W"])
        rows.append(r)
    out["mmwave"] = summarize(rows)
    print("mmWave: " + " ".join(f"{k}={list(v.values())[0][0]*100:.1f}" for k, v in out["mmwave"].items()), flush=True)

    json.dump(out, open(os.path.join(bd.ART, "stage_rsrp_generality.json"), "w"), indent=2)
    print("Saved stage_rsrp_generality.json")


if __name__ == "__main__":
    main()
