"""Review-response: sensitivity of the malicious-RIS collapse to the rank-one
LoS BS<->RIS idealisation. Replaces the coherent LoS steering vector a_bsR with
a Rician-faded version (K-factor K_db, n_scatter multipath components modelling
scattering off nearby clutter), holding the direct ray-traced channel and the
RIS<->user link unchanged. This is a sensitivity stress test of the modelling
assumption flagged in the Discussion (not a ray-traced RIS), reported at both
headline operating points (M=64/0dB and M=128/+6dB) over 3 scatter-realisation
seeds each.
"""
import os, json, numpy as np, torch
import beamdata as bd
from model import BeamMLP, DEVICE
import ris
from stage3_ris import eval_heff

N_ANT = bd.N_BS_ANT
SEEDS = [0, 1, 2]


def build_geometry_rician(H_d, M, kappa, K_db, n_scatter=8, seed=bd.SEED):
    rng = np.random.default_rng(seed)
    U = H_d.shape[0]
    hnorm = np.linalg.norm(H_d, axis=1)
    a_los = ris._steer(np.array([ris.U_BSR]), N_ANT)[0] / np.sqrt(N_ANT)
    if np.isinf(K_db):
        a_bsR = a_los
    else:
        K = 10 ** (K_db / 10.0)
        angs = rng.uniform(-1, 1, size=n_scatter)
        gpaths = (rng.standard_normal(n_scatter) + 1j * rng.standard_normal(n_scatter)) / np.sqrt(2 * n_scatter)
        a_scatter = (gpaths[:, None] * ris._steer(angs, N_ANT)).sum(axis=0) / np.sqrt(N_ANT)
        a_scatter = a_scatter / (np.linalg.norm(a_scatter) + 1e-12)
        a_bsR = np.sqrt(K / (K + 1)) * a_los + np.sqrt(1 / (K + 1)) * a_scatter
        a_bsR = a_bsR / (np.linalg.norm(a_bsR) + 1e-12)
    a_in = ris._steer(np.array([ris.U_BSR]), M)[0]
    u_out = rng.uniform(-1, 1, size=U)
    a_out = ris._steer(u_out, M)
    k_u = np.conj(a_in)[None, :] * a_out
    scale = (kappa * hnorm / ris.M_REF).astype(np.float32)
    W = bd.dft_codebook()
    return dict(
        H_d=torch.as_tensor(H_d, dtype=torch.complex64, device=DEVICE),
        a_bsR=torch.as_tensor(a_bsR, dtype=torch.complex64, device=DEVICE),
        k_u=torch.as_tensor(k_u, dtype=torch.complex64, device=DEVICE),
        scale=torch.as_tensor(scale, dtype=torch.float32, device=DEVICE),
        W=torch.as_tensor(W, dtype=torch.complex64, device=DEVICE),
        M=M,
    )


def main():
    d = bd.build_dataset()
    H_te, W = d["H_test"], d["W"]
    m = BeamMLP(d["Xte"].shape[1], d["n_beams"]).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(bd.ART, "victim_model.pt"), map_location=DEVICE)); m.eval()

    out = {"K_db_sweep": ["LoS", 10.0, 0.0], "M_sweep": [64, 128], "results": {}}
    for M in (64, 128):
        out["results"][str(M)] = {}
        for K_db in [float("inf"), 10.0, 0.0]:
            top1s, ses = [], []
            for s in SEEDS:
                g = build_geometry_rician(H_te, M, 1.0, K_db, seed=s)
                H_adv = ris.ris_attack(m, g, iters=80, seed=s)
                e = eval_heff(m, H_adv, W)
                top1s.append(e["top1"]); ses.append(e["se_ratio"])
            key = "LoS" if np.isinf(K_db) else f"{K_db:.0f}dB"
            out["results"][str(M)][key] = {
                "top1_mean": float(np.mean(top1s)), "top1_std": float(np.std(top1s)),
                "se_mean": float(np.mean(ses)), "se_std": float(np.std(ses))}
            print(f"M={M:3d} K={key:>5}: top1={np.mean(top1s)*100:5.1f}+/-{np.std(top1s)*100:.1f}%"
                  f"  SE={np.mean(ses)*100:5.1f}+/-{np.std(ses)*100:.1f}%")

    json.dump(out, open(os.path.join(bd.ART, "stage_rician.json"), "w"), indent=2)
    print("\nSaved stage_rician.json")


if __name__ == "__main__":
    main()
