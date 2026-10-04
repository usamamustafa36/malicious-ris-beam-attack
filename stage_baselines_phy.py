"""Revision E4: additional model-blind physical-layer RIS baselines (Reviewer request).

All baselines use the same RIS aperture, geometry and per-user CSI as the DNN-aware
attack; only the objective differs:
  * snr_jam      : minimize the best achievable beamforming gain max_b |w_b^H h_eff|^2
                   (destructive beamforming, Rivetti et al.)
  * beam_null    : minimize the gain of the beam the victim selects on the clean channel
                   |w_b0^H h_eff|^2 (decision-aware but gradient-free: knows b0 only)
  * beam_hijack  : maximize the normalized gain of a random wrong beam b_t,
                   |w_bt^H h_eff|^2 / max_b |w_b^H h_eff|^2 (steer the optimum elsewhere)
  * dnn_aware    : the proposed attack (victim's soft achieved/optimal gain ratio)
We also report the optimal-beam capacity ratio C_opt(h_eff)/C_opt(h_d), showing which
attacks remove channel capacity and which leave capacity intact but fool the predictor.
"""
import os, json, numpy as np, torch
import beamdata as bd
import ris
import revision_common as rc
from model import predict_probs, DEVICE
from stage_ci import classical_rsrp, classical_csi, avg

SEEDS = [42, 7, 2024]
M_LIST = [64, 128]


def optimize(g, loss_fn, iters=120, lr=0.1, seed=bd.SEED):
    torch.manual_seed(seed)
    theta = (torch.rand(g["H_d"].shape[0], g["M"], device=DEVICE) * 2 * np.pi).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    hn2 = (g["H_d"].abs() ** 2).sum(1, keepdim=True)   # per-user scale: keeps gradients >> Adam eps
    for _ in range(iters):
        opt.zero_grad()
        H = ris._h_eff(g, theta)
        loss_fn((H @ g["W"].conj().T).abs() ** 2 / hn2).mean().backward()
        opt.step()
    return ris._h_eff(g, theta).detach().cpu().numpy()


def capacity_ratio(H_eff, H_d, W, snr_db=bd.EVAL_SNR_DB):
    g = lambda H: (np.abs(H.astype(np.complex128) @ W.conj().T.astype(np.complex128)) ** 2).max(1)
    g0 = g(H_d)
    rho = 10 ** (snr_db / 10) / g0.mean()                   # clean-channel normalization
    return float(np.mean(np.log2(1 + rho * g(H_eff))) / np.mean(np.log2(1 + rho * g0)))


def main():
    d = bd.build_dataset()
    H_te, W = d["H_test"], d["W"]
    out = {"seeds": SEEDS, "results": {}}
    for kind in ("rsrp", "csi"):
        out["results"][kind] = {str(M): [] for M in M_LIST}
        for s in SEEDS:
            v = rc.train_victim(kind, d, seed=s)
            cls = (lambda H: avg([classical_rsrp(H, W, v.C, n) for n in rc.NOISE_SEEDS])) if kind == "rsrp" \
                else (lambda H: avg([classical_csi(H, W, n) for n in rc.NOISE_SEEDS]))
            b0 = predict_probs(v.model, v.feat_np(bd.eval_noise(H_te))).argmax(1)
            rng = np.random.default_rng(s)
            y0 = (np.abs(H_te.astype(np.complex128) @ W.conj().T) ** 2).argmax(1)
            bt = (y0 + rng.integers(8, bd.N_BEAMS - 8, size=len(y0))) % bd.N_BEAMS  # >= 8 beams away
            b0_t = torch.as_tensor(b0, device=DEVICE); bt_t = torch.as_tensor(bt, device=DEVICE)
            idx = torch.arange(len(b0), device=DEVICE)
            for M in M_LIST:
                g = ris.build_geometry(H_te, M, 1.0, seed=s)
                Hs = {
                    "snr_jam": optimize(g, lambda G: G.max(1).values, seed=s),
                    "beam_null": optimize(g, lambda G: G[idx, b0_t], seed=s),
                    "beam_hijack": optimize(g, lambda G: -G[idx, bt_t] / G.max(1).values.detach(), seed=s),
                    "dnn_aware": rc.ris_attack(v, g, seed=s),
                    "random": rc.random_ris(g, seed=s),
                }
                r = {"seed": s}
                for name, H in Hs.items():
                    r[name] = {**rc.evaluate_noise_avg(v, H, W), **cls(H),
                               "cap_ratio": capacity_ratio(H, H_te, W)}
                out["results"][kind][str(M)].append(r)
                print(f"[{kind} M={M} s={s}] " + "  ".join(
                    f"{n}: top1={x['top1']*100:.1f} seRef={x['se_ref3']*100:.1f} cap={x['cap_ratio']*100:.0f}"
                    for n, x in r.items() if n != "seed"), flush=True)
    json.dump(out, open(os.path.join(bd.ART, "stage_baselines_phy.json"), "w"), indent=2)
    print("Saved stage_baselines_phy.json")


if __name__ == "__main__":
    main()
