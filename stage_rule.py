"""Predictive rule for the rank-one malicious RIS.

On a rank-one BS-RIS link the phases collapse to one complex gain beta added
along the BS-RIS direction, |beta| <= ||h_d|| * 10^(Gamma_dB/20). A 1-D search
over the phase of beta gives, per user:

  Gamma_w : smallest ratio at which the strongest wide beam flips
  Gamma_theta : smallest ratio at which the learned predictor's top beam changes

A closed-form lower bound on Gamma_w treats the phase as able to help one wide
beam and hurt another at once. The script checks that bound against the search
and scores the scalar attack (top-1, eta after top-3) against the same victim.
"""
import json, os
import numpy as np
import torch
import beamdata as bd
import revision_common as rc
from model import predict_probs

RATIOS = [-20, -12, -10, -6, -3, 0, 3, 6, 10]
N_PHASE = 64
U_RIS = 0.30
SEED = 42


def wide_stats(H, C, a):
    """Complex wide-beam responses of the direct path and of the RIS direction."""
    g = H.astype(np.complex128) @ C.conj().T          # (U, L)
    r = C.astype(np.complex128).conj() @ a.astype(np.complex128)  # (L,)
    return g, r


def closed_form_gamma(g, r, hnorm):
    """Optimistic wide-beam flip threshold (dB). A single phase cannot always
    align one beam and anti-align another, so this is a lower bound."""
    mag_g = np.abs(g)
    mag_r = np.abs(r)[None, :]
    ell0 = mag_g.argmax(1)
    U, L = mag_g.shape
    g0 = mag_g[np.arange(U), ell0][:, None]
    r0 = np.abs(r)[ell0][:, None]
    num = np.maximum(g0 - mag_g, 0.0)
    den = mag_r + r0
    other = np.arange(L)[None, :] != ell0[:, None]
    beta = np.where(other & (den > 1e-12), num / np.maximum(den, 1e-12), np.inf)
    beta_min = beta.min(1)
    with np.errstate(divide="ignore"):
        gamma = 20 * np.log10(beta_min / np.maximum(hnorm, 1e-12))
    gamma[~np.isfinite(gamma)] = 40.0
    return gamma


def scalar_channels(H, a, ratio_db, phases):
    """h(beta) at |beta| fixed by the power ratio, one channel per phase."""
    hnorm = np.linalg.norm(H, axis=1)
    amp = hnorm * 10 ** (ratio_db / 20.0)                 # |beta|
    # return list of (U, N) is too big to stack; yield per phase
    betas = amp[:, None] * np.exp(1j * phases[None, :])   # (U, P)
    return betas, hnorm


def main():
    d = bd.build_dataset()
    H = d["H_test"].astype(np.complex128)
    W = d["W"]
    U, N = H.shape
    C = rc.wide_codebook(rc.L_WIDE)
    a = np.exp(1j * np.pi * np.arange(N) * U_RIS) / np.sqrt(N)
    g, r = wide_stats(H, C, a)
    hnorm = np.linalg.norm(H, axis=1)
    gamma_cf = closed_form_gamma(g, r, hnorm)

    ang = np.linspace(-1, 1, bd.N_BEAMS, endpoint=False)
    u_user = ang[np.argmax(np.abs(H @ W.conj().T) ** 2, axis=1)]
    gap = np.abs(u_user - U_RIS)
    gap = np.minimum(gap, 2 - gap)
    far = gap > (2.0 / rc.L_WIDE)          # more than one wide beam away from the RIS

    print("training victim", flush=True)
    v = rc.train_victim("rsrp", d, seed=SEED)
    phases = np.linspace(0, 2 * np.pi, N_PHASE, endpoint=False)

    # clean prediction, noiseless features, so the threshold is about the channel
    clean_pred = predict_probs(v.model, v.feat_np(H.astype(np.complex64))).argmax(1)
    clean_wide = np.abs(g).argmax(1)
    y = np.abs(H @ W.conj().T).argmax(1)

    gamma_w = np.full(U, 40.0)
    gamma_th = np.full(U, 40.0)
    rows = []
    for ratio in RATIOS:
        amp = hnorm * 10 ** (ratio / 20.0)
        best_loss = np.full(U, np.inf)
        best_pred = np.zeros(U, dtype=int)
        best_wide = np.zeros(U, dtype=int)
        best_beam_gain = np.zeros(U)
        opt_gain = np.zeros(U)
        # phase loop keeps memory at one (U, N) channel
        for p in phases:
            beta = amp * np.exp(1j * p)
            He = (H + beta[:, None] * a[None, :]).astype(np.complex64)
            gp = He.astype(np.complex128) @ C.conj().T
            wide = np.abs(gp).argmax(1)
            feats = v.feat_np(He)
            probs = predict_probs(v.model, feats)
            pred = probs.argmax(1)
            gains = np.abs(He.astype(np.complex128) @ W.conj().T) ** 2
            opt = gains.max(1)
            achieved = (probs * gains).sum(1) / (opt + 1e-30)
            take = achieved < best_loss
            best_loss[take] = achieved[take]
            best_pred[take] = pred[take]
            best_wide[take] = wide[take]
            best_beam_gain[take] = gains[np.arange(U), pred][take]
            opt_gain[take] = opt[take]
        rho = 10 ** (bd.EVAL_SNR_DB / 10.0) / opt_gain.mean()
        se = np.log2(1 + rho * best_beam_gain) / np.log2(1 + rho * opt_gain)
        flip_w = best_wide != clean_wide
        flip_d = best_pred != clean_pred
        wrong = best_pred != y
        gamma_w[flip_w] = np.minimum(gamma_w[flip_w], ratio)
        gamma_th[flip_d] = np.minimum(gamma_th[flip_d], ratio)
        row = {
            "db": ratio,
            "top1": float(np.mean(best_pred == y)),
            "se": float(np.mean(se)),
            "wide_flip": float(np.mean(flip_w)),
            "pred_changed": float(np.mean(flip_d)),
            "wide_flip_far": float(np.mean(flip_w[far])) if far.any() else None,
            "wide_flip_near": float(np.mean(flip_w[~far])) if (~far).any() else None,
            "pred_changed_far": float(np.mean(flip_d[far])),
            "pred_changed_near": float(np.mean(flip_d[~far])),
        }
        rows.append(row)
        print(f"{ratio:+5.0f} dB  top1 {row['top1']*100:5.1f}  se {row['se']*100:5.1f}  "
              f"wide-flip {row['wide_flip']*100:4.1f}  pred-changed {row['pred_changed']*100:4.1f}",
              flush=True)

    def summ(x, mask=None):
        z = x if mask is None else x[mask]
        return {"median": float(np.median(z)), "p25": float(np.percentile(z, 25)),
                "p75": float(np.percentile(z, 75))}

    out = {
        "seed": SEED, "n_users": int(U), "n_phase": N_PHASE, "ratios": RATIOS,
        "frac_far": float(np.mean(far)),
        "gamma_w_search": summ(gamma_w),
        "gamma_w_search_far": summ(gamma_w, far),
        "gamma_w_search_near": summ(gamma_w, ~far),
        "gamma_theta": summ(gamma_th),
        "gamma_theta_far": summ(gamma_th, far),
        "gamma_theta_near": summ(gamma_th, ~far),
        "gamma_w_closed_form": summ(gamma_cf),
        "closed_form_minus_search_median": float(np.median(gamma_cf - gamma_w)),
        "curve": rows,
        "clean_top1_noiseless": float(np.mean(clean_pred == y)),
    }
    path = os.path.join(bd.ART, "stage_rule.json")
    json.dump(out, open(path, "w"), indent=2)
    print("medians Gamma_w", out["gamma_w_search"], "Gamma_theta", out["gamma_theta"])
    print("closed form median", out["gamma_w_closed_form"], "bias", out["closed_form_minus_search_median"])
    print("saved", path)


if __name__ == "__main__":
    main()
