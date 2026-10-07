"""Per-user physical RIS-to-direct power ratios from the ray-traced scenarios.

For every artifacts/rt_*.npz (and rt_ris_channels.npz) and each aperture (32x32, 64x64),
computes the achievable ratio max_v ||c G diag(r_u) v||^2 / ||h_d||^2 for every grid point
with a direct path (-inf where the RIS cannot reach the user). Output:
artifacts/rt_ratio_samples.npz, used by stage_ratio_weighted.py.
"""
import os, glob, numpy as np
import beamdata as bd
import stage_raytraced as st

SIDES = [32, 64]


def main():
    files = sorted(set(glob.glob(os.path.join(bd.ART, "rt_*.npz"))) - {os.path.join(bd.ART, "rt_ratio_samples.npz")})
    out, W = {}, bd.dft_codebook()
    for f in files:
        os.environ["RT_FILE"] = os.path.basename(f)
        D = st.load()
        name = os.path.basename(f).replace(".npz", "")
        reach = np.linalg.norm(D["R"], axis=1) > 0
        for side in SIDES:
            r = np.full(len(D["H"]), -np.inf)
            idx = np.where(reach)[0]
            for i in range(0, len(idx), 4000):
                j = idx[i:i + 4000]
                g = st.geometry(D["H"][j], D["R"][j], D["G"], D["c"], st.block(D["rows"], D["cols"], side), W)
                r[j] = st.ratio_db(g)
            out[f"{name}_{side}"] = r
            fin = r[np.isfinite(r)]
            print(f"{name} {side}x{side}: users {len(r)}, reachable {reach.mean()*100:.0f}%, "
                  f"median(reachable) {np.median(fin):.1f} dB, >=0 dB {np.mean(r >= 0)*100:.1f}% of all", flush=True)
    np.savez(os.path.join(bd.ART, "rt_ratio_samples.npz"), **out)


if __name__ == "__main__":
    main()
