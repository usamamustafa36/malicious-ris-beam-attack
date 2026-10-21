"""Uplink sensing chain in front of the same per-user attack.

The surface knows its size and the BS-RIS direction. It estimates
  * the direct channel from n_pilots uplink snapshots at the BS (TDD), and
  * the user angle from n_sense contiguous RIS elements with receive chains,
and it holds an estimate that is tau milliseconds stale (Jakes, f_D = 16 Hz,
the pedestrian 3.5 GHz case used in the paper). Phases optimized on that
estimate are applied to the true channel. Ratio is the equal-power point.
"""
import json, os
import numpy as np
from scipy.special import j0
import beamdata as bd
import ris
import revision_common as rc

SEEDS = [42, 7, 2024]
FD_HZ = 16.0          # paper's pedestrian Doppler at 3.5 GHz
N_GRID = 401


def rho_of(tau_ms):
    return float(j0(2 * np.pi * FD_HZ * tau_ms * 1e-3))


def true_angle(g):
    k = g["k_u"].detach().cpu().numpy()
    dphi = np.angle(k[:, 1] * np.conj(k[:, 0]))
    u = dphi / np.pi + ris.U_BSR
    return ((u + 1) % 2) - 1


def make_geometry(H_d, u_out, M, kappa):
    U = H_d.shape[0]
    N = H_d.shape[1]
    hnorm = np.linalg.norm(H_d, axis=1)
    a_bsR = ris._steer(np.array([ris.U_BSR]), N)[0] / np.sqrt(N)
    a_in = ris._steer(np.array([ris.U_BSR]), M)[0]
    a_out = ris._steer(np.asarray(u_out, float), M)
    k_u = np.conj(a_in)[None, :] * a_out
    scale = (kappa * hnorm / ris.M_REF).astype(np.float32)
    import torch
    from model import DEVICE
    W = bd.dft_codebook()
    return dict(
        H_d=torch.as_tensor(np.asarray(H_d, np.complex64), dtype=torch.complex64, device=DEVICE),
        a_bsR=torch.as_tensor(a_bsR.astype(np.complex64), dtype=torch.complex64, device=DEVICE),
        k_u=torch.as_tensor(k_u.astype(np.complex64), dtype=torch.complex64, device=DEVICE),
        scale=torch.as_tensor(scale, dtype=torch.float32, device=DEVICE),
        W=torch.as_tensor(W, dtype=torch.complex64, device=DEVICE),
        M=M,
    )


def mle_angle(y):
    """y: (U, n_sense) complex observations of exp(j*pi*m*u), m = 0..n_sense-1.

    The template already carries the conjugate phase exp(-j*pi*m*grid), so the
    correlation is y @ steer.T. A second conjugate matches -u and leaves a
    median error of 0.5 for a uniform angle, independent of SNR.
    """
    n = y.shape[1]
    grid = np.linspace(-1, 1, N_GRID, endpoint=False)
    steer = np.exp(-1j * np.pi * grid[:, None] * np.arange(n)[None, :])  # (G, n)
    corr = y @ steer.T
    return grid[np.abs(corr).argmax(1)]


def _self_check():
    """Noiseless recovery of a known departure angle, including the k_u construction."""
    rng = np.random.default_rng(0)
    U, M, n = 500, 64, 4
    u = rng.uniform(-1, 1, size=U)
    a_in = ris._steer(np.array([ris.U_BSR]), M)[0]
    k = np.conj(a_in)[None, :] * ris._steer(u, M)
    import torch
    hat = true_angle({"k_u": torch.as_tensor(k.astype(np.complex64))})
    y = np.exp(1j * np.pi * u[:, None] * np.arange(n)[None, :])
    hat2 = mle_angle(y)
    def circ(a, b):
        e = np.abs(a - b)
        return np.minimum(e, 2 - e)
    e1, e2 = circ(hat, u), circ(hat2, u)
    # true_angle is exact; the MLE is quantized to the 401-point grid (spacing ~0.005).
    if np.median(e1) > 1e-4 or np.median(e2) > 0.005:
        raise RuntimeError(f"angle self-check failed: true {np.median(e1):.4f} mle {np.median(e2):.4f}")
    print(f"angle self-check median err true {np.median(e1):.4f} mle {np.median(e2):.4f}", flush=True)


def estimate(g, snr_db, n_pilots, n_sense, tau_ms, rng):
    """Return a geometry dict the attacker believes, built from stale noisy pilots."""
    H = g["H_d"].detach().cpu().numpy().astype(np.complex128)
    u = true_angle(g)
    M = g["M"]
    rho = rho_of(tau_ms)
    innov = (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape)) / np.sqrt(2)
    innov *= np.linalg.norm(H, axis=1, keepdims=True) / (np.linalg.norm(innov, axis=1, keepdims=True) + 1e-12)
    H_past = rho * H + np.sqrt(max(1 - rho ** 2, 0.0)) * innov
    p = np.mean(np.abs(H_past) ** 2, axis=1, keepdims=True)
    n0 = p / (10 ** (snr_db / 10.0) * n_pilots)
    noise = np.sqrt(n0 / 2) * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
    H_hat = (H_past + noise).astype(np.complex64)

    # common cascade phase the attacker missed because the estimate is stale
    z = (rng.standard_normal(len(u)) + 1j * rng.standard_normal(len(u))) / np.sqrt(2)
    phi = np.angle(rho + np.sqrt(max(1 - rho ** 2, 0.0)) * z)
    m = np.arange(n_sense)[None, :]
    y = np.exp(1j * (np.pi * m * u[:, None] + phi[:, None]))
    sig_p = 1.0
    n0e = sig_p / (10 ** (snr_db / 10.0) * n_pilots)
    y = y + np.sqrt(n0e / 2) * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))
    u_hat = mle_angle(y)
    # attacker applies the stale phase it cannot separate from the angle fit;
    # the fit absorbs a common phase, so the residual is the geometry error only.
    return make_geometry(H_hat, u_hat, M, kappa=1.0), u, u_hat


def cell(victim, g_true, snr_db, n_pilots, n_sense, tau_ms, seed):
    if snr_db is None:
        He = rc.ris_attack(victim, g_true, seed=seed)
        ang_err = 0.0
    else:
        rng = np.random.default_rng(seed)
        g_hat, u, u_hat = estimate(g_true, snr_db, n_pilots, n_sense, tau_ms, rng)
        err = np.abs(u_hat - u)
        err = np.minimum(err, 2 - err)
        ang_err = float(np.median(err))
        _, theta = rc.ris_attack(victim, g_hat, seed=seed, return_theta=True)
        He = ris._h_eff(g_true, theta).detach().cpu().numpy()
    ev = rc.evaluate_noise_avg(victim, He, bd.dft_codebook())
    ev["angle_err_median"] = ang_err
    return ev


def main():
    _self_check()
    d = bd.build_dataset()
    H_te = d["H_test"]
    # cells: tag -> kwargs. None snr means perfect CSI.
    cells = [("perfect", dict(snr_db=None, n_pilots=0, n_sense=0, tau_ms=0))]
    for snr in (0, 10, 20):
        cells.append((f"snr{snr}", dict(snr_db=snr, n_pilots=4, n_sense=4, tau_ms=2)))
    for ns in (2, 8):
        cells.append((f"sense{ns}", dict(snr_db=10, n_pilots=4, n_sense=ns, tau_ms=2)))
    for npil in (1, 8):
        cells.append((f"pilots{npil}", dict(snr_db=10, n_pilots=npil, n_sense=4, tau_ms=2)))
    for tau in (0, 5):
        cells.append((f"tau{tau}", dict(snr_db=10, n_pilots=4, n_sense=4, tau_ms=tau)))
    out = {"seeds": SEEDS, "fd_hz": FD_HZ, "rho": {str(t): rho_of(t) for t in (0, 2, 5)},
           "cells": [c[0] for c in cells], "trials": []}
    for s in SEEDS:
        v = rc.train_victim("rsrp", d, seed=s)
        g = ris.build_geometry(H_te, 64, 1.0, seed=s)
        trial = {"seed": s}
        for name, kw in cells:
            ev = cell(v, g, seed=s, **kw)
            trial[name] = ev
            print(f"[s={s}] {name:10} top1 {ev['top1']*100:5.1f}  se3 {ev['se_ref3']*100:5.1f}  "
                  f"angle {ev['angle_err_median']:.3f}", flush=True)
        out["trials"].append(trial)
    summary = {}
    for name, _ in cells:
        for key in ("top1", "se_ref3", "angle_err_median"):
            xs = [t[name][key] for t in out["trials"]]
            mu, h = rc.ci95(xs)
            summary.setdefault(name, {})[key] = {"mean": mu, "ci95": h}
    out["summary"] = summary
    path = os.path.join(bd.ART, "stage_sensing.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
