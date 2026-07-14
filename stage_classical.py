"""Review-response: classical (non-learned) argmax beam selector vs the DNN victim.

b_classical = argmax_b |w_b^H h~|^2 computed DIRECTLY on the same noisy CSI
estimate the DNN sees -- no learned model at all. Reported at the same three
operating points as the DNN (clean, DNN-aware malicious RIS at M=128/+6dB,
model-blind SNR-jamming RIS at M=128) to test whether the malicious-RIS
collapse is a property of the learned model or of the perturbed channel itself.
"""
import os, json, numpy as np, torch
import beamdata as bd
from model import BeamMLP, predict_probs, DEVICE
import ris

rng = np.random.default_rng(bd.SEED)


def eval_both(model, H_eff, W, snr=bd.PILOT_SNR_DB):
    """Paired eval: DNN and classical argmax on the SAME noisy realisation of h_eff;
    both scored against the true (noiseless) perturbed-channel optimum y_eff."""
    Hn = bd.eval_noise(H_eff, snr)   # shared deterministic eval-noise realization
    Xn = bd.complex_to_feat(Hn)
    pred_dnn = predict_probs(model, Xn).argmax(1)
    gains_n = np.abs(Hn.astype(np.complex128) @ W.conj().T) ** 2
    pred_cl = gains_n.argmax(1)
    gains_true = np.abs(H_eff.astype(np.complex128) @ W.conj().T) ** 2
    y_eff = gains_true.argmax(1)
    return {
        "dnn": {"top1": float(np.mean(pred_dnn == y_eff)), "se": bd.se_ratio(H_eff, W, pred_dnn)},
        "classical": {"top1": float(np.mean(pred_cl == y_eff)), "se": bd.se_ratio(H_eff, W, pred_cl)},
    }


def main():
    d = bd.build_dataset()
    H_te, W = d["H_test"], d["W"]
    m = BeamMLP(d["Xte"].shape[1], d["n_beams"]).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(bd.ART, "victim_model.pt"), map_location=DEVICE)); m.eval()

    out = {}
    out["clean"] = eval_both(m, H_te, W)
    g = ris.build_geometry(H_te, 128, 1.0, seed=bd.SEED)
    H_adv = ris.ris_attack(m, g, iters=80, seed=bd.SEED)
    H_jam = ris.ris_snr_jam(g, iters=120, seed=bd.SEED)
    out["malicious_M128"] = eval_both(m, H_adv, W)
    out["jamming_M128"] = eval_both(m, H_jam, W)

    print(f"{'condition':>16} | {'DNN top1':>9} {'DNN SE':>7} | {'classical top1':>15} {'classical SE':>13}")
    for k, v in out.items():
        print(f"{k:>16} | {v['dnn']['top1']*100:8.1f}% {v['dnn']['se']*100:6.1f}% |"
              f" {v['classical']['top1']*100:14.1f}% {v['classical']['se']*100:12.1f}%")

    json.dump(out, open(os.path.join(bd.ART, "stage_classical.json"), "w"), indent=2)
    print("\nSaved stage_classical.json")


if __name__ == "__main__":
    main()
