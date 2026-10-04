"""Revision E7: sensitivity to the RIS channel model (replaces the rank-one Rician test).

The conference version's ``Rician'' test still used a single BS-side vector, i.e. a
rank-one reflected path. Here the BS<-RIS channel is a genuine N x M matrix with a LoS
term plus P=8 scattered paths (rank up to 9), at K in {LoS, 10 dB, 0 dB, scattered-only},
and optionally a Rician RIS->user link (K_ru = 5 dB, 4 paths per user). The total
reflected power is held fixed (||G||_F^2 = M), so only the spatial structure changes.
Conditions: DNN-aware attack, model-blind SNR jamming, random RIS; 3 trials.
"""
import os, json, numpy as np
import beamdata as bd
import ris
import revision_common as rc
from stage_baselines_phy import optimize, capacity_ratio

SEEDS = [42, 7, 2024]
CHANNELS = [("LoS", float("inf"), None), ("K=10dB", 10.0, None), ("K=0dB", 0.0, None),
            ("scattered", -float("inf"), None), ("K=0dB+RU", 0.0, 5.0)]


def main():
    d = bd.build_dataset()
    H_te, W = d["H_test"], d["W"]
    out = {"seeds": SEEDS, "channels": [c[0] for c in CHANNELS], "results": {}}
    for kind in ("rsrp", "csi"):
        out["results"][kind] = {}
        rows = {f"{name}_M{M}": [] for name, _, _ in CHANNELS for M in (64, 128)}
        for s in SEEDS:
            v = rc.train_victim(kind, d, seed=s)
            for name, K, Kru in CHANNELS:
                for M in (64, 128):
                    g = ris.build_geometry_multipath(H_te, M, 1.0, K, seed=s, ris_user_K_db=Kru)
                    r = {"rank": g["rank"]}
                    for cond, H in (("dnn_aware", rc.ris_attack(v, g, seed=s)),
                                    ("snr_jam", optimize(g, lambda G: G.max(1).values, seed=s)),
                                    ("random", rc.random_ris(g, seed=s))):
                        r[cond] = {**rc.evaluate_noise_avg(v, H, W), "cap_ratio": capacity_ratio(H, H_te, W)}
                    rows[f"{name}_M{M}"].append(r)
                    print(f"[{kind} s={s} {name} M={M} rank={g['rank']}] " + "  ".join(
                        f"{c}: top1={r[c]['top1']*100:.1f} seRef={r[c]['se_ref3']*100:.1f} cap={r[c]['cap_ratio']*100:.0f}"
                        for c in ("dnn_aware", "snr_jam", "random")), flush=True)
        for key, rs in rows.items():
            out["results"][kind][key] = {c: {m: rc.ci95([r[c][m] for r in rs]) for m in rs[0][c]}
                                         for c in ("dnn_aware", "snr_jam", "random")}
            out["results"][kind][key]["rank"] = rs[0]["rank"]
        json.dump(out, open(os.path.join(bd.ART, "stage_multipath.json"), "w"), indent=2)
    print("Saved stage_multipath.json")


if __name__ == "__main__":
    main()
