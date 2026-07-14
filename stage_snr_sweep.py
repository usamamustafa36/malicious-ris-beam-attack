"""Pilot/operating-SNR sweep: is the malicious-RIS collapse an artifact of the
single 10 dB operating point? Evaluates clean and malicious-RIS (M=128, +6 dB)
top-1 for the 10 dB-trained victim across pilot SNR in {0, 10, 20} dB."""
import os, json, numpy as np, torch
import beamdata as bd
from model import BeamMLP, DEVICE
import ris
from stage3_ris import eval_heff


def main():
    d = bd.build_dataset()
    H_te, W = d["H_test"], d["W"]   # full test set, so the 10 dB point equals the headline eval
    victim = BeamMLP(d["Xte"].shape[1], d["n_beams"]).to(DEVICE)
    victim.load_state_dict(torch.load(os.path.join(bd.ART, "victim_model.pt"), map_location=DEVICE)); victim.eval()

    g = ris.build_geometry(H_te, 128, 1.0, seed=bd.SEED)
    H_adv = ris.ris_attack(victim, g, iters=80, seed=bd.SEED)

    out = {"snr_db": [], "clean_top1": [], "malicious_top1": [], "drop": []}
    for snr in [0, 10, 20]:
        c = eval_heff(victim, H_te, W, snr=snr)["top1"]
        a = eval_heff(victim, H_adv, W, snr=snr)["top1"]
        out["snr_db"].append(snr); out["clean_top1"].append(c); out["malicious_top1"].append(a)
        out["drop"].append(c - a)
        print(f"  pilot SNR {snr:2d} dB: clean top1={c*100:5.1f}%  malicious={a*100:5.1f}%  drop={-(a-c)*100:5.1f} pts")

    json.dump(out, open(os.path.join(bd.ART, "snr_sweep.json"), "w"), indent=2)
    print("Saved snr_sweep.json")


if __name__ == "__main__":
    main()
