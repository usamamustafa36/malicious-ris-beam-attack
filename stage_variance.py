"""Review-response: seed variance for the black-box transfer (19.9% at M=128)
and CNN-victim (18.7% at M=128) headline numbers, so neither reads as a
single-seed fluke. Retrains the surrogate/CNN and reruns the attack across
3 seeds each, reporting mean +/- std.
"""
import os, json, numpy as np, torch
import beamdata as bd
from model import BeamMLP, BeamCNN, train_model, DEVICE
import ris
from stage3_ris import eval_heff

SEEDS = [2024, 7, 1]     # surrogate/CNN training seeds (attack seed tied to same value); 3 seeds


def blackbox_variance(d, victim, W, H_te, M=128):
    tops = []
    for s in SEEDS:
        surrogate = train_model(d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["n_beams"],
                                epochs=40, hidden=(384, 192), seed=s, verbose=False)
        g = ris.build_geometry(H_te, M, 1.0, seed=s)
        H_tr = ris.ris_attack(surrogate, g, iters=80, seed=s)
        e = eval_heff(victim, H_tr, W)
        tops.append(e["top1"])
        print(f"    surrogate seed={s}: transfer top1={e['top1']*100:.1f}%")
    return {"top1_mean": float(np.mean(tops)), "top1_std": float(np.std(tops)), "runs": tops}


def cnn_variance(d, W, H_te):
    """One CNN trained per seed, reused for both M=64 and M=128 attacks
    (was previously retraining per (M, seed) pair -- wasteful, fixed)."""
    out = {"64": {"runs": []}, "128": {"runs": []}}
    for s in SEEDS:
        cnn = BeamCNN(d["Xtr"].shape[1], d["n_beams"])
        cnn = train_model(d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["n_beams"],
                          epochs=60, seed=s, model=cnn, verbose=False)
        for M in (64, 128):
            g = ris.build_geometry(H_te, M, 1.0, seed=s)
            H_adv = ris.ris_attack(cnn, g, iters=80, seed=s)
            e = eval_heff(cnn, H_adv, W)
            out[str(M)]["runs"].append(e["top1"])
            print(f"    CNN seed={s} M={M}: top1={e['top1']*100:.1f}%")
    for M in (64, 128):
        tops = out[str(M)]["runs"]
        out[str(M)]["top1_mean"] = float(np.mean(tops))
        out[str(M)]["top1_std"] = float(np.std(tops))
    return out


def main():
    d = bd.build_dataset()
    H_te, W = d["H_test"], d["W"]
    victim = BeamMLP(d["Xte"].shape[1], d["n_beams"]).to(DEVICE)
    victim.load_state_dict(torch.load(os.path.join(bd.ART, "victim_model.pt"), map_location=DEVICE)); victim.eval()

    print("Black-box transfer variance (M=128), 3 surrogate seeds:")
    bb = blackbox_variance(d, victim, W, H_te)
    print(f"  -> {bb['top1_mean']*100:.1f} +/- {bb['top1_std']*100:.1f}%")

    print("\nCNN-victim variance (M=64,128), 3 training seeds:")
    cnn = cnn_variance(d, W, H_te)

    out = {"blackbox_transfer_M128": bb, "cnn": cnn}
    json.dump(out, open(os.path.join(bd.ART, "stage_variance.json"), "w"), indent=2)
    print("\nSaved stage_variance.json")


if __name__ == "__main__":
    main()
