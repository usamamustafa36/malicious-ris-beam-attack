"""Structural defenses that do not train on the attack.

Two checks named in the paper, scored on the partial-measurement victim:

  top-3     the deployed refinement (19 measurements)
  union     top-3 union the 4 narrow beams under the strongest wide beam
  probes    top-3 plus 4 narrow beams drawn outside the top-3

Each is scored clean, under the attack that ignores the defense, and under an
attack that knows the defense and minimizes the gain of the beam it will serve.
The attacker still does not see the evaluation noise. 3 trials, M corresponding
to 0 dB and +6 dB.
"""
import json, os
import numpy as np
import torch
import beamdata as bd
import ris
import revision_common as rc
from model import predict_probs, DEVICE

SEEDS = [42, 7, 2024]
Q = 4
K = 3
W_LOCAL = 4


def _candidates_np(order, wide_hat, B, L, mode, rng):
    """Per-user candidate narrow beams. order is (U, B) best-first DNN ranking."""
    U = order.shape[0]
    top = order[:, :K]
    if mode == "topk":
        return top
    if mode == "union":
        s = B // L
        centre = wide_hat.argmax(1) * s
        offs = np.arange(W_LOCAL) - W_LOCAL // 2
        local = (centre[:, None] + offs[None, :]) % B
        out = np.empty((U, K + W_LOCAL), dtype=int)
        out[:, :K] = top
        out[:, K:] = local
        return out
    # probes: Q beams outside the top-3, fixed by rng
    extra = np.empty((U, Q), dtype=int)
    pool = np.arange(B)
    for i in range(U):
        ban = set(top[i].tolist())
        choices = [b for b in pool if b not in ban]
        extra[i] = rng.choice(choices, size=Q, replace=False)
    out = np.empty((U, K + Q), dtype=int)
    out[:, :K] = top
    out[:, K:] = extra
    return out


def score(victim, H, mode, noise_seeds=rc.NOISE_SEEDS, probe_seed=0):
    """Noise-averaged top-1 and SE ratio of the beam the defense actually serves,
    plus the mean number of narrow beams measured (wide beams are always L)."""
    W = None
    B = bd.N_BEAMS
    L = victim.C.shape[0]
    tops, ses, nm = [], [], []
    gains_true = None
    for n in noise_seeds:
        Hn = bd.add_cn_noise(H, bd.PILOT_SNR_DB, np.random.default_rng(n))
        probs = predict_probs(victim.model, victim.feat_np(Hn))
        order = np.argsort(-probs, axis=1)
        wide = np.abs(Hn.astype(np.complex128) @ victim.C.conj().T) ** 2
        rng = np.random.default_rng(probe_seed + n)
        cand = _candidates_np(order, wide, B, L, mode, rng)
        # unique count
        nm.append(np.mean([len(set(row.tolist())) for row in cand]))
        meas = np.abs(Hn.astype(np.complex128) @ bd.dft_codebook().conj().T) ** 2
        pick = cand[np.arange(len(cand)), np.take_along_axis(meas, cand, 1).argmax(1)]
        if gains_true is None:
            W = bd.dft_codebook()
            gains_true = np.abs(H.astype(np.complex128) @ W.conj().T) ** 2
            y = gains_true.argmax(1)
        tops.append(float(np.mean(pick == y)))
        ses.append(bd.se_ratio(H, W, pick))
    return {"top1": float(np.mean(tops)), "se": float(np.mean(ses)),
            "n_narrow": float(np.mean(nm)), "n_meas": float(L + np.mean(nm))}


def attack_adaptive(victim, g, mode, iters=80, lr=0.1, seed=bd.SEED):
    """Minimize the served gain of a defense the attacker knows. Candidate indices
    are detached; the soft objective keeps a gradient into the predictor."""
    victim.model.eval()
    torch.manual_seed(seed)
    U, M = g["H_d"].shape[0], g["M"]
    B = g["W"].shape[0]
    L = victim.C_t.shape[0]
    theta = (torch.rand(U, M, device=DEVICE) * 2 * np.pi).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    Wc = g["W"].conj()
    for _ in range(iters):
        opt.zero_grad()
        H = ris._h_eff(g, theta)
        logits = victim.model(victim.feat_t(H))
        gains = (H @ Wc.T).abs() ** 2
        opt_gain = gains.max(1, keepdim=True).values.detach()
        soft = (torch.softmax(logits, dim=1) * gains).sum(1, keepdim=True) / (opt_gain + 1e-30)
        top = logits.topk(K, dim=1).indices.detach()
        if mode == "topk":
            idx = top
        elif mode == "union":
            wide = (H @ victim.C_t.conj().T).abs() ** 2
            centre = wide.argmax(1) * (B // L)
            offs = torch.arange(W_LOCAL, device=DEVICE) - W_LOCAL // 2
            local = (centre[:, None] + offs[None, :]) % B
            idx = torch.cat([top, local], dim=1)
        else:
            score_r = torch.rand(U, B, device=DEVICE)
            score_r.scatter_(1, top, -1.0)
            extra = score_r.topk(Q, dim=1).indices
            idx = torch.cat([top, extra], dim=1)
        served = torch.gather(gains, 1, idx).max(1, keepdim=True).values / (opt_gain + 1e-30)
        (soft + served).mean().backward()
        opt.step()
    with torch.no_grad():
        return ris._h_eff(g, theta).cpu().numpy()


def main():
    d = bd.build_dataset()
    H_te, W = d["H_test"], d["W"]
    modes = ["topk", "union", "probes"]
    out = {"seeds": SEEDS, "q": Q, "trials": []}
    for s in SEEDS:
        v = rc.train_victim("rsrp", d, seed=s)
        trial = {"seed": s, "clean": {}, "unaware": {}, "adaptive": {}}
        for mode in modes:
            trial["clean"][mode] = score(v, H_te, mode)
        for M, tag in ((64, "0dB"), (128, "6dB")):
            g = ris.build_geometry(H_te, M, 1.0, seed=s)
            unaware = rc.ris_attack(v, g, seed=s)
            trial["unaware"][tag] = {mode: score(v, unaware, mode) for mode in modes}
            trial["adaptive"][tag] = {}
            for mode in modes:
                Ha = attack_adaptive(v, g, mode, seed=s)
                trial["adaptive"][tag][mode] = score(v, Ha, mode)
                print(f"[s={s}] {tag} adaptive {mode}: "
                      f"top1 {trial['adaptive'][tag][mode]['top1']*100:.1f} "
                      f"se {trial['adaptive'][tag][mode]['se']*100:.1f} "
                      f"meas {trial['adaptive'][tag][mode]['n_meas']:.1f}", flush=True)
        out["trials"].append(trial)
        c = trial["clean"]["topk"]
        print(f"[s={s}] clean top3-refine top1 {c['top1']*100:.1f} se {c['se']*100:.1f}", flush=True)
    # means
    def grab(path):
        xs = []
        for t in out["trials"]:
            node = t
            for p in path:
                node = node[p]
            xs.append(node)
        return xs
    summary = {}
    for mode in modes:
        summary[("clean", mode)] = rc.ci95([t["clean"][mode]["se"] for t in out["trials"]])
    out["summary_se"] = {}
    for kind in ("unaware", "adaptive"):
        for tag in ("0dB", "6dB"):
            for mode in modes:
                ses = [t[kind][tag][mode]["se"] for t in out["trials"]]
                t1 = [t[kind][tag][mode]["top1"] for t in out["trials"]]
                nm = [t[kind][tag][mode]["n_meas"] for t in out["trials"]]
                out["summary_se"][f"{kind}_{tag}_{mode}"] = {
                    "se_mean": rc.ci95(ses)[0], "se_ci": rc.ci95(ses)[1],
                    "top1_mean": rc.ci95(t1)[0], "top1_ci": rc.ci95(t1)[1],
                    "n_meas": float(np.mean(nm)),
                }
    for mode in modes:
        se = rc.ci95([t["clean"][mode]["se"] for t in out["trials"]])
        t1 = rc.ci95([t["clean"][mode]["top1"] for t in out["trials"]])
        out["summary_se"][f"clean_{mode}"] = {
            "se_mean": se[0], "se_ci": se[1], "top1_mean": t1[0], "top1_ci": t1[1],
            "n_meas": float(np.mean([t["clean"][mode]["n_meas"] for t in out["trials"]])),
        }
    path = os.path.join(bd.ART, "stage_structdef.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out["summary_se"], indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
