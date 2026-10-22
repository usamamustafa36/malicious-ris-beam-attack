"""Rogue sensing chain with the impairments the manuscript had left out.

Same equal-power attack and four contiguous elements at 20 dB, with no direct-channel
delay, plus one impairment at a time and one combined case:

  * mutual coupling among the sensing elements (tridiagonal, coefficient 0.3),
  * a fixed per-element calibration phase (20 degree standard deviation),
  * 3-bit I/Q quantization on those elements,
  * a sine-angle change between the listen and the measurement (the 0-5 ms
    pedestrian delay does not move the angle; these offsets do).

The attacker estimates the angle with the uncoupled, uncalibrated template.
Phases computed on that estimate are applied to the true current channel.
"""
import json, os
import numpy as np
import beamdata as bd
import ris
import revision_common as rc
from stage_sensing import make_geometry, mle_angle, true_angle, rho_of

SEEDS = [42, 7, 2024]
SNR_DB = 20.0
N_SENSE = 4
N_PILOTS = 4


def wrap(u):
    return ((u + 1.0) % 2.0) - 1.0


def coupling_matrix(n, alpha):
    i = np.arange(n)
    C = alpha ** np.abs(i[:, None] - i[None, :])
    return C.astype(np.complex128)


def quantize_iq(y, bits):
    levels = 2 ** bits
    scale = np.max(np.maximum(np.abs(y.real), np.abs(y.imag)), axis=1, keepdims=True) + 1e-12

    def q(a):
        unit = np.clip((a / scale + 1) * 0.5, 0, 1)
        return (np.round(unit * (levels - 1)) / (levels - 1)) * 2 - 1

    return (scale * q(y.real)) + 1j * (scale * q(y.imag))


def estimate(g, du, rng, alpha=0.0, cal_deg=0.0, bits=None):
    H = g["H_d"].detach().cpu().numpy().astype(np.complex128)
    u = true_angle(g)
    u_sense = wrap(u - du)
    p = np.mean(np.abs(H) ** 2, axis=1, keepdims=True)
    n0 = p / (10 ** (SNR_DB / 10.0) * N_PILOTS)
    noise = np.sqrt(n0 / 2) * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
    H_hat = (H + noise).astype(np.complex64)

    m = np.arange(N_SENSE)[None, :]
    y = np.exp(1j * np.pi * m * u_sense[:, None])
    n0e = 1.0 / (10 ** (SNR_DB / 10.0) * N_PILOTS)
    y = y + np.sqrt(n0e / 2) * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))
    if alpha:
        y = y @ coupling_matrix(N_SENSE, alpha).T
    if cal_deg:
        cal = np.exp(1j * rng.normal(0.0, np.deg2rad(cal_deg), size=N_SENSE))
        y = y * cal
    if bits is not None:
        y = quantize_iq(y, bits)
    u_hat = mle_angle(y)
    err = np.abs(wrap(u_hat - u))
    err = np.minimum(err, 2 - err)
    g_hat = make_geometry(H_hat, u_hat, g["M"], kappa=1.0)
    return g_hat, float(np.median(err))


def main():
    cells = [
        ("nominal", dict(du=0.0)),
        ("coupling", dict(du=0.0, alpha=0.3)),
        ("calibration", dict(du=0.0, cal_deg=20.0)),
        ("quant3", dict(du=0.0, bits=3)),
        ("angle02", dict(du=0.02)),
        ("angle05", dict(du=0.05)),
        ("combined", dict(du=0.05, alpha=0.3, cal_deg=20.0, bits=3)),
    ]
    d = bd.build_dataset()
    H_te = d["H_test"]
    trials = []
    for s in SEEDS:
        v = rc.train_victim("rsrp", d, seed=s)
        g = ris.build_geometry(H_te, 64, 1.0, seed=s)
        trial = {"seed": s}
        for name, kw in cells:
            rng = np.random.default_rng(s)
            g_hat, ang = estimate(g, rng=rng, **kw)
            _, theta = rc.ris_attack(v, g_hat, seed=s, return_theta=True)
            He = ris._h_eff(g, theta).detach().cpu().numpy()
            ev = rc.evaluate_noise_avg(v, He, bd.dft_codebook())
            ev["angle_err_median"] = ang
            trial[name] = ev
            print(f"[s={s}] {name:12} top1 {ev['top1']*100:5.1f}  se3 {ev['se_ref3']*100:5.1f}  "
                  f"angle {ang:.3f}", flush=True)
        trials.append(trial)
        json.dump({"seeds": SEEDS, "trials": trials},
                  open(os.path.join(bd.ART, "stage_sensing_impair.json"), "w"), indent=2)
    summary = {}
    for name, _ in cells:
        summary[name] = {}
        for key in ("top1", "se_ref3", "angle_err_median"):
            mu, h = rc.ci95([t[name][key] for t in trials])
            summary[name][key] = {"mean": mu, "ci95": h}
    out = {"seeds": SEEDS, "snr_db": SNR_DB, "n_sense": N_SENSE, "summary": summary, "trials": trials}
    path = os.path.join(bd.ART, "stage_sensing_impair.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
