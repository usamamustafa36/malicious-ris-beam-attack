"""Sector fallback that does not use the learned scores.

The base station measures every narrow beam under the w strongest wide beams
and serves the strongest of those measurements. w = 1 is the model-free local
sweep, measured rather than trusted. An aware attacker minimizes the gain of
whatever that rule will serve. Indices are detached from the loss.
"""
import json, os
import numpy as np
import torch
import beamdata as bd
import ris
import revision_common as rc
from model import DEVICE

SEEDS = [42, 7, 2024]
WIDTHS = (1, 2, 4, 8)


def sectors(wide, w, B, L):
    """wide: (U, L) powers. Returns (U, w * s) narrow-beam indices."""
    s = B // L
    if isinstance(wide, np.ndarray):
        top = np.argpartition(-wide, w - 1, axis=1)[:, :w]
        offs = np.arange(s) - s // 2
        local = (top[:, :, None] * s + offs[None, None, :]) % B
        return local.reshape(len(wide), w * s)
    top = torch.topk(wide, w, dim=1).indices
    offs = torch.arange(s, device=wide.device) - s // 2
    local = (top[:, :, None] * s + offs) % B
    return local.reshape(wide.shape[0], w * s)


def score(victim, H, w, noise_seeds=rc.NOISE_SEEDS):
    B, L = bd.N_BEAMS, victim.C.shape[0]
    W = bd.dft_codebook()
    tops, ses, nm = [], [], []
    gains_true = np.abs(H.astype(np.complex128) @ W.conj().T) ** 2
    y = gains_true.argmax(1)
    for n in noise_seeds:
        Hn = bd.add_cn_noise(H, bd.PILOT_SNR_DB, np.random.default_rng(n))
        wide = np.abs(Hn.astype(np.complex128) @ victim.C.conj().T) ** 2
        cand = sectors(wide, w, B, L)
        nm.append(np.mean([len(set(row.tolist())) for row in cand]))
        meas = np.abs(Hn.astype(np.complex128) @ W.conj().T) ** 2
        pick = cand[np.arange(len(cand)), np.take_along_axis(meas, cand, 1).argmax(1)]
        tops.append(float(np.mean(pick == y)))
        ses.append(bd.se_ratio(H, W, pick))
    return {"top1": float(np.mean(tops)), "se": float(np.mean(ses)),
            "n_narrow": float(np.mean(nm)), "n_meas": float(L + np.mean(nm))}


def attack_aware(victim, g, w, iters=80, lr=0.1, seed=bd.SEED):
    victim.model.eval()
    torch.manual_seed(seed)
    U, M = g["H_d"].shape[0], g["M"]
    B, L = g["W"].shape[0], victim.C_t.shape[0]
    theta = (torch.rand(U, M, device=DEVICE) * 2 * np.pi).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    Wc = g["W"].conj()
    for _ in range(iters):
        opt.zero_grad()
        H = ris._h_eff(g, theta)
        gains = (H @ Wc.T).abs() ** 2
        opt_gain = gains.max(1, keepdim=True).values.detach()
        wide = (H @ victim.C_t.conj().T).abs() ** 2
        idx = sectors(wide, w, B, L).detach()
        served = torch.gather(gains, 1, idx).max(1).values / (opt_gain.squeeze(1) + 1e-30)
        served.mean().backward()
        opt.step()
    with torch.no_grad():
        return ris._h_eff(g, theta).cpu().numpy()


def main():
    d = bd.build_dataset()
    H_te = d["H_test"]
    trials = []
    for s in SEEDS:
        v = rc.train_victim("rsrp", d, seed=s)
        trial = {"seed": s, "clean": {}, "unaware": {}, "aware": {}}
        for w in WIDTHS:
            trial["clean"][str(w)] = score(v, H_te, w)
        for M, tag in ((64, "0dB"), (128, "6dB")):
            g = ris.build_geometry(H_te, M, 1.0, seed=s)
            unaware = rc.ris_attack(v, g, seed=s)
            trial["unaware"][tag] = {str(w): score(v, unaware, w) for w in WIDTHS}
            trial["aware"][tag] = {}
            for w in WIDTHS:
                Ha = attack_aware(v, g, w, seed=s)
                trial["aware"][tag][str(w)] = score(v, Ha, w)
                ev = trial["aware"][tag][str(w)]
                print(f"[s={s}] {tag} aware w={w} top1 {ev['top1']*100:.1f} "
                      f"se {ev['se']*100:.1f} meas {ev['n_meas']:.1f}", flush=True)
        trials.append(trial)
        json.dump({"seeds": SEEDS, "trials": trials},
                  open(os.path.join(bd.ART, "stage_restore.json"), "w"), indent=2)
    summary = {"clean": {}, "unaware": {}, "aware": {}}
    for w in WIDTHS:
        summary["clean"][str(w)] = {
            k: dict(zip(("mean", "ci95"), rc.ci95([t["clean"][str(w)][k] for t in trials])))
            for k in ("top1", "se", "n_meas")}
        for tag in ("0dB", "6dB"):
            for kind in ("unaware", "aware"):
                summary[kind].setdefault(tag, {})[str(w)] = {
                    k: dict(zip(("mean", "ci95"), rc.ci95([t[kind][tag][str(w)][k] for t in trials])))
                    for k in ("top1", "se", "n_meas")}
    out = {"seeds": SEEDS, "widths": list(WIDTHS), "summary": summary, "trials": trials}
    path = os.path.join(bd.ART, "stage_restore.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
