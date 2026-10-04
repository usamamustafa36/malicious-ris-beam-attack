"""Revision E1 + E5: headline numbers with 95% confidence intervals, for both victims.

For each of 5 independent trials (victim training seed = RIS-geometry seed = attack
init seed), the attack is crafted once on the noiseless effective channel and scored
over 5 independent evaluation-noise realizations (the attacker never sees them). All
conditions within a trial use identical test users. We report mean +/- 95% t-interval
across trials of the noise-averaged metric.

Also scores two model-free baselines for the partial-measurement setting, on the same
noisy wide-beam measurements: (i) the narrow beam nearest the strongest wide beam;
(ii) a local sweep of the s=B/L narrow beams under the strongest wide beam (s extra
measurements), and the full-CSI argmax for the CSI victim.
"""
import os, json, numpy as np
import beamdata as bd
import ris
import revision_common as rc

M_LIST = [64, 128]


def classical_rsrp(H_eff, W, C, noise_seed):
    Hn = bd.add_cn_noise(H_eff, bd.PILOT_SNR_DB, np.random.default_rng(noise_seed))
    wide = np.abs(Hn.astype(np.complex128) @ C.conj().T) ** 2
    s = bd.N_BEAMS // C.shape[0]
    near = wide.argmax(1) * s
    gains = np.abs(H_eff.astype(np.complex128) @ W.conj().T) ** 2
    y = gains.argmax(1)
    meas = np.abs(Hn.astype(np.complex128) @ W.conj().T) ** 2
    cand = (near[:, None] + np.arange(-(s // 2), s - s // 2)[None, :]) % bd.N_BEAMS
    loc = cand[np.arange(len(y)), np.take_along_axis(meas, cand, 1).argmax(1)]
    return {"nearest_top1": float(np.mean(near == y)), "nearest_se": bd.se_ratio(H_eff, W, near),
            "local_top1": float(np.mean(loc == y)), "local_se": bd.se_ratio(H_eff, W, loc)}


def classical_csi(H_eff, W, noise_seed):
    Hn = bd.add_cn_noise(H_eff, bd.PILOT_SNR_DB, np.random.default_rng(noise_seed))
    pred = (np.abs(Hn.astype(np.complex128) @ W.conj().T) ** 2).argmax(1)
    y = (np.abs(H_eff.astype(np.complex128) @ W.conj().T) ** 2).argmax(1)
    return {"argmax_top1": float(np.mean(pred == y)), "argmax_se": bd.se_ratio(H_eff, W, pred)}


def avg(dicts):
    return {k: float(np.mean([x[k] for x in dicts])) for k in dicts[0]}


def main():
    d = bd.build_dataset()
    H_te, W = d["H_test"], d["W"]
    out = {"noise_seeds": rc.NOISE_SEEDS, "train_seeds": rc.TRAIN_SEEDS, "victims": {}}
    for kind in ("rsrp", "csi"):
        trials = []
        for s in rc.TRAIN_SEEDS:
            v = rc.train_victim(kind, d, seed=s)
            cls = (lambda H: avg([classical_rsrp(H, W, v.C, n) for n in rc.NOISE_SEEDS])) if kind == "rsrp" \
                else (lambda H: avg([classical_csi(H, W, n) for n in rc.NOISE_SEEDS]))
            t = {"clean": rc.evaluate_noise_avg(v, H_te, W), "clean_classical": cls(H_te)}
            for M in M_LIST:
                g = ris.build_geometry(H_te, M, 1.0, seed=s)
                Hr = rc.random_ris(g, seed=s)
                Ha = rc.ris_attack(v, g, seed=s)
                t[f"random_{M}"] = rc.evaluate_noise_avg(v, Hr, W)
                t[f"attack_{M}"] = rc.evaluate_noise_avg(v, Ha, W)
                t[f"attack_{M}_classical"] = cls(Ha)
            trials.append(t)
            print(f"[{kind}] seed={s}: clean={t['clean']['top1']*100:.1f}  "
                  f"M64={t['attack_64']['top1']*100:.1f}  M128={t['attack_128']['top1']*100:.1f}  "
                  f"rand128={t['random_128']['top1']*100:.1f}", flush=True)
        summ = {}
        for cond in trials[0]:
            summ[cond] = {}
            for met in trials[0][cond]:
                mu, h = rc.ci95([t[cond][met] for t in trials])
                summ[cond][met] = {"mean": mu, "ci95": h}
        out["victims"][kind] = {"summary": summ, "trials": trials}
        for cond, mets in summ.items():
            print(f"  {kind:4s} {cond:22s} " + "  ".join(
                f"{k}={v['mean']*100:5.1f}±{v['ci95']*100:.1f}" for k, v in mets.items()))
    json.dump(out, open(os.path.join(bd.ART, "stage_ci.json"), "w"), indent=2)
    print("Saved stage_ci.json")


if __name__ == "__main__":
    main()
