"""Revision (R2.2/R5.3): attack on fully ray-traced RIS channels (Sionna RT, Munich).

Uses artifacts/rt_ris_channels.npz from rt_trace_ris.py: ray-traced direct channels,
a ray-traced N x M BS->RIS channel G and per-user RIS->user channels r_u, with the
physical discrete-element scaling c = sqrt(zeta) 4 pi A_e / lambda^2, so
    h_eff(v) = h_d + c G diag(r_u) v
and every user's RIS-to-direct power ratio follows from the geometry. Victims are
retrained on this scene's direct channels (users with a non-zero direct channel, 70/10/20
split); the attack is scored on test users that the RIS can reach (r_u != 0).
Two analyses:
  * "norm"  (channel structure): ray-traced G and r_u on the central 32x32 block, with each
            user's reflected amplitude rescaled so that its achievable RIS-to-direct ratio
            is 0 or +6 dB, the operating points of the main results;
  * "phys"  (physical power): central 16x16, 32x32 and 64x64 blocks with the physical
            scaling, results binned by each user's achievable ratio.
Conditions: random RIS, DNN-aware attack, model-blind SNR jamming. 3 trials.
"""
import os, json, numpy as np, torch
from sklearn.model_selection import train_test_split
import beamdata as bd
import revision_common as rc
from model import DEVICE
from stage_ci import classical_rsrp, classical_csi, avg
from stage_baselines_phy import optimize, capacity_ratio

SEEDS = [42, 7, 2024]
NORM_SIDE, NORM_TARGETS = 32, [0.0, 6.0]
PHYS_SIDES = [16, 32, 64]
BINS = [(-np.inf, -20), (-20, -10), (-10, 0), (0, np.inf)]
OUT_JSON = os.environ.get("RT_JSON", "stage_raytraced.json")


def load():
    z = np.load(os.path.join(bd.ART, os.environ.get("RT_FILE", "rt_ris_channels.npz")))
    Hd, R, G = z["H_d"], z["R"], z["G"]
    keep = np.linalg.norm(Hd, axis=1) > 0
    s = 1.0 / np.median(np.linalg.norm(Hd[keep], axis=1))          # global rescale, physics-neutral
    c = float(np.sqrt(10 ** (z["zeta_db"] / 10)) * z["elem_gain"])
    return dict(H=(Hd[keep] * s).astype(np.complex64), R=(R[keep] * s).astype(np.complex64), G=G, c=c,
                rows=int(z["rows"]), cols=int(z["cols"]), users=z["users"][keep], ris_pos=z["ris_pos"],
                n_grid=len(Hd))


def block(rows, cols, side):
    """Indices of the central side x side block (element index = col*rows + row)."""
    r0, c0 = (rows - side) // 2, (cols - side) // 2
    return np.array([c * rows + r for c in range(c0, c0 + side) for r in range(r0, r0 + side)])


def geometry(H, R, G, scale, sel, W):
    t = lambda x: torch.as_tensor(x, dtype=torch.complex64, device=DEVICE)
    return dict(H_d=t(H), k_u=t(np.conj(R[:, sel])), G=t(G[:, sel]),
                scale=torch.as_tensor(np.broadcast_to(scale, (len(H),)).copy(), dtype=torch.float32, device=DEVICE),
                W=t(W), M=len(sel))


@torch.no_grad()
def ratio_db(g, iters=40):
    """Per-user achievable RIS-to-direct power ratio max_v ||scale G diag(r) v||^2 / ||h_d||^2
    (unit-modulus power iteration)."""
    r, Gm, c = g["k_u"].conj(), g["G"], g["scale"][:, None]
    A = lambda v: c * ((r * v) @ Gm.T)
    AH = lambda y: c * r.conj() * (y @ Gm.conj())
    v = torch.exp(1j * torch.angle(AH(g["H_d"])))
    for _ in range(iters):
        v = torch.exp(1j * torch.angle(AH(A(v))))
    p = (A(v).abs() ** 2).sum(1) / (g["H_d"].abs() ** 2).sum(1)
    return (10 * torch.log10(p + 1e-30)).cpu().numpy()


def metrics(v, kind, H, H_d, W, mask):
    if mask.sum() < 30:
        return None
    cls = classical_rsrp if kind == "rsrp" else classical_csi
    extra = (lambda h, n: cls(h, W, v.C, n)) if kind == "rsrp" else (lambda h, n: cls(h, W, n))
    return {**rc.evaluate_noise_avg(v, H[mask], W), **avg([extra(H[mask], n) for n in rc.NOISE_SEEDS]),
            "cap_ratio": capacity_ratio(H[mask], H_d[mask], W), "n": int(mask.sum())}


def conditions(v, g, s):
    return (("random", rc.random_ris(g, seed=s)), ("dnn_aware", rc.ris_attack(v, g, seed=s)),
            ("snr_jam", optimize(g, lambda G: G.max(1).values, seed=s)))


def main():
    D = load()
    W = bd.dft_codebook()
    idx = np.arange(len(D["H"]))
    tr, te = train_test_split(idx, test_size=0.2, random_state=bd.SEED)
    tr, va = train_test_split(tr, test_size=0.1 / 0.8, random_state=bd.SEED)
    d = dict(H_tr=D["H"][tr], H_va=D["H"][va], W=W, n_beams=bd.N_BEAMS)
    vis = np.linalg.norm(D["R"][te], axis=1) > 0                   # users the RIS can reach
    H_te, R_te = D["H"][te][vis], D["R"][te][vis]
    out = {"seeds": SEEDS, "n_grid": D["n_grid"], "n_users": len(idx), "n_test": len(te), "n_test_visible": int(vis.sum()),
           "ris_side": D["rows"], "bins": [list(b) for b in BINS], "ratio_db": {}, "results": {}}
    print(f"grid {D['n_grid']}, users with direct path {len(idx)}, test {len(te)}, RIS-visible test {vis.sum()}", flush=True)

    geos = {}
    for side in PHYS_SIDES:                                      # physical power
        sel = block(D["rows"], D["cols"], side)
        g = geometry(H_te, R_te, D["G"], D["c"], sel, W)
        rdb = ratio_db(g)
        geos[f"phys{side}"] = (g, {"all": np.ones(len(rdb), bool)} |
                               {f"bin{i}": (rdb >= lo) & (rdb < hi) for i, (lo, hi) in enumerate(BINS)})
        out["ratio_db"][str(side)] = {"pct": np.percentile(rdb, [10, 25, 50, 75, 90]).tolist(),
                                      "frac_bins": [float(np.mean((rdb >= lo) & (rdb < hi))) for lo, hi in BINS]}
        print(f"phys {side}x{side}: ratio dB pct10/50/90 = " + " / ".join(f"{x:.1f}" for x in np.percentile(rdb, [10, 50, 90]))
              + "  bins " + " ".join(f"{f*100:.0f}%" for f in out["ratio_db"][str(side)]["frac_bins"]), flush=True)
        if side == NORM_SIDE:                                     # channel structure, fixed ratio
            for T in NORM_TARGETS:
                sc = D["c"] * 10 ** ((T - rdb) / 20)
                gn = geometry(H_te, R_te, D["G"], sc, sel, W)
                geos[f"norm{int(T)}"] = (gn, {"all": np.ones(len(rdb), bool)})
    for kind in ("rsrp", "csi"):
        rows = []
        for s in SEEDS:
            v = rc.train_victim(kind, d, seed=s)
            r = {"clean": metrics(v, kind, H_te, H_te, W, np.ones(len(H_te), bool))}
            for name, (g, masks) in geos.items():
                for cond, H in conditions(v, g, s):
                    for mk, m in masks.items():
                        r[f"{name}_{cond}_{mk}"] = metrics(v, kind, H, H_te, W, m)
                for mk, m in masks.items():
                    r[f"{name}_clean_{mk}"] = metrics(v, kind, H_te, H_te, W, m)
                a, j, c0 = r[f"{name}_dnn_aware_all"], r[f"{name}_snr_jam_all"], r["clean"]
                print(f"[{kind} s={s} {name}] clean top1={c0['top1']*100:.1f} se3={c0['se_ref3']*100:.1f} | "
                      f"attack top1={a['top1']*100:.1f} se3={a['se_ref3']*100:.1f} cap={a['cap_ratio']*100:.0f} | "
                      f"jam top1={j['top1']*100:.1f} se3={j['se_ref3']*100:.1f} | "
                      f"random top1={r[f'{name}_random_all']['top1']*100:.1f}", flush=True)
            rows.append(r)
        keys = [k for k in rows[0] if all(rr.get(k) is not None for rr in rows)]
        out["results"][kind] = {k: {m: rc.ci95([rr[k][m] for rr in rows]) for m in rows[0][k]} for k in keys}
        json.dump(out, open(os.path.join(bd.ART, OUT_JSON), "w"), indent=2)
    print("Saved", OUT_JSON)


if __name__ == "__main__":
    main()
