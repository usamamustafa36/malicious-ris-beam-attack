"""Revision (R2.2/R5.3): ray-traced malicious-RIS channels with Sionna RT.

Runs in a separate Python>=3.11 environment (it is not imported by the other stages):
    uv venv -p 3.11 sionna_env && VIRTUAL_ENV=sionna_env uv pip install sionna-rt==1.1.0
(sionna-rt 2.x needs a newer NVIDIA driver/OptiX than 550; 1.1.0 has no diffraction).
The published channels were traced with sionna-rt 2.x and RT_DIFFRACTION=1 on a Kaggle
T4 GPU (driver 580); the CUDA variant is used by default, RT_VARIANT selects another. In the Munich scene at 3.5 GHz it ray-traces
  * h_d  : BS (64-element ULA) -> users                      (direct channel)
  * G    : BS (64-element ULA) -> each RIS element           (N x M, exact per element)
  * r    : RIS (planar array)  -> users                      (U x M)
for a 64x64 (default) half-wavelength RIS on a building facade with line of sight to the BS.
Both hops include LoS, specular reflections and diffraction (max depth 3), so the
BS->RIS link is not rank-one by construction. RIS elements have a cos(angle) power
pattern (front half-space only). The cascaded channel is
    h_ris(v) = sqrt(zeta) * (4 pi A_e / lambda^2) * G diag(r_u) v,   A_e = (lambda/2)^2,
the discrete-element RIS path-loss model of Tang et al. (2021), so the RIS-to-direct
power ratio follows from the geometry instead of being fixed. Output:
artifacts/rt_ris_channels.npz.
"""
import os, sys, time, numpy as np
import mitsuba as mi, drjit as dr
if os.environ.get("RT_VARIANT"):              # e.g. llvm_ad_mono_polarized (CPU) if OptiX is unusable
    mi.set_variant(os.environ["RT_VARIANT"])
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
from sionna.rt.antenna_pattern import register_antenna_pattern, PolarizedAntennaPattern

FREQ = 3.5e9
LAM = 299792458.0 / FREQ
BS_POS = np.array([8.5, 21.0, 27.0])
RIS_AIM = np.array([70.4, 21.0, 10.0])      # facade point in LoS of the BS (found by ray casting)
RIS_ROWS = RIS_COLS = int(os.environ.get("RT_RIS_SIDE", "64"))
ZETA_DB = -1.0
USER_RADIUS, USER_STEP, USER_H = 150.0, 1.0, 1.5
BATCH = 4000
DIFFRACTION = os.environ.get("RT_DIFFRACTION", "0") == "1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "rt_ris_channels.npz")


def v_riscos_pattern(theta, phi):
    """Power gain cos(angle to boresight) in the front half-space, 0 behind (peak 1)."""
    c = dr.sin(theta) * dr.cos(phi)
    return mi.Complex2f(dr.sqrt(dr.maximum(c, 0.0)), 0)


register_antenna_pattern("riscos", lambda *, polarization, polarization_model="tr38901_2":
                         PolarizedAntennaPattern(v_pattern=v_riscos_pattern, polarization=polarization,
                                                 polarization_model=polarization_model))


def hit(ms, o, d):
    ray = mi.Ray3f(mi.Point3f(*[mi.Float(o[:, i]) for i in range(3)]),
                   mi.Vector3f(*[mi.Float(d[:, i]) for i in range(3)]))
    si = ms.ray_intersect(ray)
    return (np.array(si.t), np.stack([np.array(si.n[i]) for i in range(3)], 1), np.array(si.is_valid()))


def narrowband(paths):
    """Sum of complex path coefficients (carrier phase included) -> numpy."""
    h = paths.cfr(frequencies=mi.Float([0.0]), normalize_delays=False, out_type="numpy")
    return h[..., 0, 0]          # [rx, rx_ant, tx, tx_ant]


def main():
    sc = load_scene(rt.scene.munich)
    sc.frequency = FREQ
    ms = sc.mi_scene
    # RIS centre and normal from the facade hit
    d0 = (RIS_AIM - BS_POS) / np.linalg.norm(RIS_AIM - BS_POS)
    t, n, _ = hit(ms, BS_POS[None], d0[None])
    p_hit, nrm = BS_POS + d0 * t[0], n[0] / np.linalg.norm(n[0])
    if nrm @ d0 > 0: nrm = -nrm
    ris_pos = p_hit + 0.05 * nrm
    print("RIS at", ris_pos.round(2), "normal", nrm.round(3), "BS-RIS", round(float(t[0]), 1), "m", flush=True)

    # user grid: outdoor points (first downward hit at ground level) within USER_RADIUS of the RIS
    g = np.mgrid[-USER_RADIUS:USER_RADIUS:USER_STEP, -USER_RADIUS:USER_RADIUS:USER_STEP].reshape(2, -1).T
    g = g[np.linalg.norm(g, axis=1) <= USER_RADIUS] + ris_pos[:2]
    tt, _, vv = hit(ms, np.c_[g, np.full(len(g), 200.0)], np.tile([0, 0, -1.0], (len(g), 1)))
    out = vv & (200.0 - tt < 0.3)
    users = np.c_[g[out], np.full(out.sum(), USER_H)]
    users = users[np.linalg.norm(users - ris_pos, axis=1) > 3.0]
    print("candidate users", len(users), flush=True)

    solver = PathSolver()
    kw = dict(max_depth=3, los=True, specular_reflection=True, refraction=False)
    if DIFFRACTION:                        # only available in sionna-rt >= 2
        kw["diffraction"] = True
    bs_array = PlanarArray(num_rows=1, num_cols=64, pattern="iso", polarization="V")
    ris_array = PlanarArray(num_rows=RIS_ROWS, num_cols=RIS_COLS, pattern="riscos", polarization="V")
    single = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    look = ris_pos + 30 * nrm          # BS boresight toward the area in front of the RIS

    def run(tx_pos, tx_look, tx_arr, rx_arr, rx_pos, synthetic, rx_look=None, samples=1_000_000):
        for o in list(sc.transmitters) + list(sc.receivers):
            sc.remove(o)
        sc.tx_array, sc.rx_array = tx_arr, rx_arr
        sc.add(Transmitter("tx", position=mi.Point3f(*map(float, tx_pos)), look_at=mi.Point3f(*map(float, tx_look))))
        for i, p in enumerate(rx_pos):
            r = Receiver(f"rx{i}", position=mi.Point3f(*map(float, p)))
            if rx_look is not None:
                r.look_at(mi.Point3f(*map(float, rx_look)))
            sc.add(r)
        return narrowband(solver(sc, synthetic_array=synthetic, samples_per_src=samples,
                                 max_num_paths_per_src=min(samples, 100_000), **kw))

    t0 = time.time()
    # G: BS -> RIS elements (exact per element pair, no plane-wave approximation)
    G = run(BS_POS, look, bs_array, ris_array, [ris_pos], False, rx_look=ris_pos + nrm,
            samples=100_000)[0, :, 0, :].T   # (N, M)
    print("G", G.shape, "rank(99% energy)", int(np.sum(np.cumsum(np.linalg.svd(G, compute_uv=False)**2) /
          np.sum(np.abs(G)**2) < 0.99) + 1), f"{time.time()-t0:.0f}s", flush=True)
    Hd, R = [], []
    for i in range(0, len(users), BATCH):
        u = users[i:i + BATCH]
        Hd.append(run(BS_POS, look, bs_array, single, u, True)[:, 0, 0, :])                       # (b, N)
        R.append(run(ris_pos, ris_pos + nrm, ris_array, single, u, True)[:, 0, 0, :])            # (b, M)
        print(f"  users {i+len(u)}/{len(users)}  {time.time()-t0:.0f}s", flush=True)
    Hd, R = np.concatenate(Hd), np.concatenate(R)
    pos_el = np.array(ris_array.normalized_positions).T          # (M, 3) in wavelengths (local frame)
    np.savez_compressed(OUT, H_d=Hd.astype(np.complex64), R=R.astype(np.complex64), G=G.astype(np.complex64),
                        users=users, bs_pos=BS_POS, ris_pos=ris_pos, ris_normal=nrm, el_pos=pos_el,
                        freq=FREQ, zeta_db=ZETA_DB, elem_gain=np.pi, rows=RIS_ROWS, cols=RIS_COLS)
    print("saved", OUT, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1:                     # quick test: few users
        USER_RADIUS, USER_STEP = 30.0, 6.0
        OUT = OUT.replace(".npz", "_test.npz")
    main()
