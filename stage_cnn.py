"""Second-architecture generality check: does the malicious-RIS collapse hold for
a CNN victim, not just the MLP? Trains a 1-D CNN beam predictor and runs the same
white-box RIS attack (M=64,128) against it."""
import os, json, numpy as np, torch
import beamdata as bd
from model import BeamCNN, train_model, predict_probs, DEVICE
import ris
from stage3_ris import eval_heff


def main():
    d = bd.build_dataset()
    H_te, W = d["H_test"], d["W"]
    cnn = BeamCNN(d["Xtr"].shape[1], d["n_beams"])
    cnn = train_model(d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["n_beams"],
                      epochs=60, seed=bd.SEED, model=cnn, verbose=False)
    torch.save(cnn.state_dict(), os.path.join(bd.ART, "cnn_victim.pt"))

    clean = eval_heff(cnn, H_te, W)
    out = {"clean": clean, "attack": {}, "random": {}}
    print(f"CNN clean: top1={clean['top1']*100:.1f}%  SE={clean['se_ratio']*100:.1f}%")
    for M in (64, 128):
        g = ris.build_geometry(H_te, M, 1.0, seed=bd.SEED)
        adv = eval_heff(cnn, ris.ris_attack(cnn, g, iters=80, seed=bd.SEED), W)
        torch.manual_seed(bd.SEED)   # seeded random-RIS control
        rnd = eval_heff(cnn, ris._h_eff(g, torch.rand(len(H_te), M, device=DEVICE) * 2 * np.pi).cpu().numpy(), W)
        out["attack"][str(M)] = adv; out["random"][str(M)] = rnd
        print(f"  M={M}: malicious top1={adv['top1']*100:5.1f}% SE={adv['se_ratio']*100:4.0f}% | "
              f"random top1={rnd['top1']*100:5.1f}%")

    json.dump(out, open(os.path.join(bd.ART, "cnn.json"), "w"), indent=2)
    print("Saved cnn.json")


if __name__ == "__main__":
    main()
