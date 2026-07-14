"""Stage 3: evaluate the malicious-RIS attack (size sweep, b-bit phases, controls)."""
import os, json, numpy as np, torch
import beamdata as bd
from model import BeamMLP, predict_probs, DEVICE
import ris

M_SWEEP = [16, 32, 64, 128, 256]
KAPPA = 1.0
rng = np.random.default_rng(bd.SEED)


def eval_heff(model, H_eff, W, snr=bd.PILOT_SNR_DB):
    """Honest eval: DNN predicts from NOISY effective CSI; score on true h_eff.
    Uses the shared deterministic eval-noise realization (bd.eval_feats) so this
    number is reproducible and identical across stages."""
    Xn = bd.eval_feats(H_eff, snr)
    pred = predict_probs(model, Xn).argmax(1)
    gains = np.abs(H_eff.astype(np.complex128) @ W.conj().T) ** 2
    y_eff = gains.argmax(1)
    top1 = float(np.mean(pred == y_eff))
    return {"top1": top1, "se_ratio": bd.se_ratio(H_eff, W, pred)}


def main():
    d = bd.build_dataset()
    H_d, W, y0 = d["H_test"], d["W"], d["yte"]
    m = BeamMLP(d["Xte"].shape[1], d["n_beams"]).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(bd.ART, "victim_model.pt"))); m.eval()

    clean = eval_heff(m, H_d, W)
    out = {"clean": clean, "kappa": KAPPA, "M_sweep": M_SWEEP, "opt": [], "rand": []}
    print(f"Clean (no RIS): top1={clean['top1']*100:.1f}%  SE={clean['se_ratio']*100:.1f}%\n")
    print(f"{'M':>4} | {'random-RIS top1':>15} {'SE':>6} | {'malicious-RIS top1':>18} {'SE':>6}")
    for M in M_SWEEP:
        g = ris.build_geometry(H_d, M, KAPPA)
        # control: random (benign/unoptimised) RIS -- seeded so the control is reproducible
        torch.manual_seed(bd.SEED)
        H_rand = ris._h_eff(g, torch.rand(H_d.shape[0], M, device=DEVICE) * 2 * np.pi).detach().cpu().numpy()
        r = eval_heff(m, H_rand, W)
        # malicious optimised RIS
        H_adv = ris.ris_attack(m, g, iters=80, lr=0.1)
        a = eval_heff(m, H_adv, W)
        out["rand"].append(r); out["opt"].append(a)
        print(f"{M:4d} | {r['top1']*100:14.1f}% {r['se_ratio']*100:5.1f}% |"
              f" {a['top1']*100:17.1f}% {a['se_ratio']*100:5.1f}%")

    # b-bit hardware constraint at M=128
    print("\nDiscrete-phase (hardware) RIS at M=128:")
    g = ris.build_geometry(H_d, 128, KAPPA)
    out["bbit"] = {}
    for b in [1, 2, 3, None]:
        H_b = ris.ris_attack(m, g, iters=80, lr=0.1, bbit=b)
        e = eval_heff(m, H_b, W)
        out["bbit"][str(b)] = e
        lab = "continuous" if b is None else f"{b}-bit"
        print(f"  {lab:>10}: top1={e['top1']*100:5.1f}%  SE={e['se_ratio']*100:5.1f}%")

    json.dump(out, open(os.path.join(bd.ART, "stage3_ris.json"), "w"), indent=2)
    print("\nSaved stage3_ris.json")


if __name__ == "__main__":
    main()
