"""Plane-wave versus spherical RIS-to-user phase on the traced Munich channels.

The stored RIS-to-user coefficients use Sionna's synthetic array, a plane-wave
phase across the aperture. For each element the geometric plane-wave phase of
the user direction is replaced by the spherical-wave phase of the same
direction. Traced amplitudes are unchanged. Diffuse scattering is not added.
The comparison is the physical 64 x 64 surface, on the test users whose
achievable ratio is at least 0 dB, which is the dominated-user headline.
"""
import json, os
import numpy as np
import torch
import beamdata as bd
import revision_common as rc
from stage_raytraced import load, block, geometry, ratio_db
from sklearn.model_selection import train_test_split

SEEDS = [42, 7, 2024]
LAM = 299792458.0 / 3.5e9


def array_axes(el, users, ris_pos, nrm, R):
    """Match local in-plane axes to the traced phase of a strong user."""
    up = np.array([0.0, 0.0, 1.0])
    horiz = np.cross(nrm, up)
    horiz = horiz / np.linalg.norm(horiz)
    i = int(np.argmax(np.linalg.norm(R, axis=1)))
    direction = users[i] - ris_pos
    direction = direction / np.linalg.norm(direction)
    center = np.argmin(el[:, 1] ** 2 + el[:, 2] ** 2)
    phase = np.angle(R[i] * np.conj(R[i, center]))

    def pred(y_hat, z_hat):
        offset = LAM * (el[:, 1:2] * y_hat + el[:, 2:3] * z_hat)
        return np.angle(np.exp(-1j * 2 * np.pi / LAM * offset @ direction))

    cands = {
        "yz": (horiz, up),
        "zy": (up, horiz),
        "y-z": (horiz, -up),
        "-yz": (-horiz, up),
        "-y-z": (-horiz, -up),
        "-zy": (-up, horiz),
    }
    scores = {}
    for name, (y_hat, z_hat) in cands.items():
        # correlation of unit phasors, immune to a single unwrapping error
        scores[name] = float(np.abs(np.mean(np.exp(1j * (phase - pred(y_hat, z_hat))))))
    best = max(scores, key=scores.get)
    if scores[best] < 0.8:
        raise RuntimeError(f"array-frame fit is too weak to trust: {scores}")
    return cands[best], scores, best


def spherical(R, el, users, ris_pos, y_hat, z_hat):
    offset = LAM * (el[:, 1:2] * y_hat[None, :] + el[:, 2:3] * z_hat[None, :])  # (M, 3)
    rel = users[:, None, :] - ris_pos[None, None, :]                              # (U, 1, 3)
    dist = np.linalg.norm(rel - offset[None, :, :], axis=2)
    dist0 = np.linalg.norm(users - ris_pos, axis=1)
    direction = rel[:, 0, :] / dist0[:, None]
    plane = offset @ direction.T                                                  # (M, U) meters
    sph = dist - dist0[:, None]                                                   # (U, M)
    delta = -2 * np.pi / LAM * (sph - plane.T)
    edge = np.argmax(el[:, 1] ** 2 + el[:, 2] ** 2)
    return R * np.exp(1j * delta), delta[:, edge]


def main():
    D = load()
    # load() dropped users the direct path misses and rescaled. Recover geometry
    # arrays on that same kept set by reloading positions from the npz through load's users.
    z = np.load(os.path.join(bd.ART, os.environ.get("RT_FILE", "rt_ris_channels.npz")))
    keep = np.linalg.norm(z["H_d"], axis=1) > 0
    el = z["el_pos"]
    (y_hat, z_hat), scores, best = array_axes(el, z["users"][keep], z["ris_pos"], z["ris_normal"], z["R"][keep])
    print("axis fit", best, {k: round(v, 3) for k, v in scores.items()}, flush=True)
    R_sph, edge_phase = spherical(z["R"][keep], el, z["users"][keep], z["ris_pos"], y_hat, z_hat)
    scale = 1.0 / np.median(np.linalg.norm(z["H_d"][keep], axis=1))
    R_sph = (R_sph * scale).astype(np.complex64)

    W = bd.dft_codebook()
    idx = np.arange(len(D["H"]))
    tr, te = train_test_split(idx, test_size=0.2, random_state=bd.SEED)
    tr, va = train_test_split(tr, test_size=0.1 / 0.8, random_state=bd.SEED)
    d = dict(H_tr=D["H"][tr], H_va=D["H"][va], W=W, n_beams=bd.N_BEAMS)
    vis = np.linalg.norm(D["R"][te], axis=1) > 0
    sel = block(D["rows"], D["cols"], 64)
    H_te, R_te = D["H"][te][vis], D["R"][te][vis]
    R_te_s = R_sph[te][vis]
    g_plane = geometry(H_te, R_te, D["G"], D["c"], sel, W)
    g_sph = geometry(H_te, R_te_s, D["G"], D["c"], sel, W)
    rdb = ratio_db(g_plane)
    dom = rdb >= 0
    print(f"dominated {int(dom.sum())} / {len(dom)}  edge phase rad "
          f"median {np.median(np.abs(edge_phase[te][vis][dom])):.2f}", flush=True)

    def run(g, mask):
        # ratio_db and the attack both batch over g's users; slice to the mask
        sub = {k: (v[mask] if torch.is_tensor(v) and v.shape[0] == len(mask) else v) for k, v in g.items()}
        return sub

    rows = []
    for s in SEEDS:
        v = rc.train_victim("rsrp", d, seed=s)
        row = {"seed": s}
        for name, g in (("plane", g_plane), ("spherical", g_sph)):
            sub = run(g, dom)
            He = rc.ris_attack(v, sub, seed=s)
            row[name] = rc.evaluate_noise_avg(v, He, W)
            row[name + "_clean"] = rc.evaluate_noise_avg(v, H_te[dom], W)
            print(f"[s={s}] {name} top1 {row[name]['top1']*100:.1f} se3 {row[name]['se_ref3']*100:.1f} "
                  f"clean {row[name + '_clean']['top1']*100:.1f}", flush=True)
        rows.append(row)
    summary = {}
    for key in ("plane", "spherical", "plane_clean"):
        summary[key] = {}
        for metric in ("top1", "se_ref3"):
            mu, h = rc.ci95([r[key][metric] for r in rows])
            summary[key][metric] = {"mean": mu, "ci95": h}
    out = {"seeds": SEEDS, "axis": best, "axis_scores": scores,
           "n_dominated": int(dom.sum()),
           "edge_phase_abs_median": float(np.median(np.abs(edge_phase[te][vis][dom]))),
           "summary": summary, "trials": rows}
    path = os.path.join(bd.ART, "stage_spherical.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
