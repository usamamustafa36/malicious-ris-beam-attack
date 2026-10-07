"""Revision: attack under a realistic distribution of RIS-to-direct power ratios.

The main results fix the ratio at 0 or +6 dB for every user. Here each DeepMIMO test user
instead receives a ratio drawn from the per-user physical ratios of the ray-traced
scenarios (rt_ratio_samples.npz, pooled over scenes; users the surface cannot reach get no
RIS path). The geometric cascade of Eq. (2) is scaled per user accordingly
(kappa_u = 10^(r_u/20) at M = 64). Reported overall and by ratio bin, for 1.37 m (32x32)
and 2.74 m (64x64) surfaces. 3 trials, noise-averaged scoring.
"""
import os, json, numpy as np
import beamdata as bd
import ris
import revision_common as rc
from stage_ci import classical_rsrp, classical_csi, avg

SEEDS = [42, 7, 2024]
SIDES = [32, 64]
BINS = {"lt-10": (-np.inf, -10), "-10to0": (-10, 0), "ge0": (0, np.inf)}


def pooled(side):
    z = np.load(os.path.join(bd.ART, "rt_ratio_samples.npz"))
    keys = [k for k in z.files if k.endswith(f"_{side}")]
    return np.concatenate([z[k] for k in keys]), keys


def metrics(v, kind, H, W, m):
    if m.sum() < 30:
        return None
    cls = (lambda h, n: classical_rsrp(h, W, v.C, n)) if kind == "rsrp" else (lambda h, n: classical_csi(h, W, n))
    return {**rc.evaluate_noise_avg(v, H[m], W), **avg([cls(H[m], n) for n in rc.NOISE_SEEDS]), "n": int(m.sum())}


def main():
    d = bd.build_dataset()
    H_te, W = d["H_test"], d["W"]
    out = {"seeds": SEEDS, "results": {}, "sources": {}}
    for kind in ("rsrp", "csi"):
        rows = []
        for s in SEEDS:
            v = rc.train_victim(kind, d, seed=s)
            r = {"clean_all": metrics(v, kind, H_te, W, np.ones(len(H_te), bool))}
            for side in SIDES:
                pool, keys = pooled(side)
                out["sources"][str(side)] = keys
                ratio = np.random.default_rng(s + side).choice(pool, size=len(H_te))
                kappa = np.where(np.isfinite(ratio), 10 ** (np.clip(ratio, -80, 40) / 20), 0.0)
                g = ris.build_geometry(H_te, 64, kappa, seed=s)
                masks = {"all": np.ones(len(H_te), bool)} | {b: (ratio >= lo) & (ratio < hi) for b, (lo, hi) in BINS.items()}
                Ha, Hr = rc.ris_attack(v, g, seed=s), rc.random_ris(g, seed=s)
                for mk, m in masks.items():
                    r[f"{side}_attack_{mk}"] = metrics(v, kind, Ha, W, m)
                    r[f"{side}_random_{mk}"] = metrics(v, kind, Hr, W, m)
                    r[f"{side}_clean_{mk}"] = metrics(v, kind, H_te, W, m)
                r[f"{side}_frac"] = {b: float(m.mean()) for b, m in masks.items()}
                a, c0 = r[f"{side}_attack_all"], r[f"{side}_clean_all"]
                print(f"[{kind} s={s} {side}x{side}] all: clean top1={c0['top1']*100:.1f} se3={c0['se_ref3']*100:.1f} "
                      f"-> attack top1={a['top1']*100:.1f} se3={a['se_ref3']*100:.1f} | ge0 ({masks['ge0'].mean()*100:.0f}%): "
                      f"{r[f'{side}_clean_ge0']['top1']*100:.1f} -> {r[f'{side}_attack_ge0']['top1']*100:.1f}", flush=True)
            rows.append(r)
        keys = [k for k in rows[0] if not k.endswith("_frac") and all(rr.get(k) is not None for rr in rows)]
        out["results"][kind] = {k: {m: rc.ci95([rr[k][m] for rr in rows]) for m in rows[0][k]} for k in keys}
        out["results"][kind]["frac"] = {str(sd): rows[0][f"{sd}_frac"] for sd in SIDES}
        json.dump(out, open(os.path.join(bd.ART, "stage_ratio_weighted.json"), "w"), indent=2)
    print("Saved stage_ratio_weighted.json")


if __name__ == "__main__":
    main()
