"""Detector-aware (adaptive) evasion attack: does a rogue RIS that jointly fools
the classifier AND minimises the detector score defeat the 0.998-AUC detector?
Reports the detector AUC and victim top-1 as the evasion weight lam grows."""
import os, json, numpy as np, torch
import beamdata as bd
from model import BeamMLP, predict_probs, to_t, DEVICE
import ris
from stage3_ris import eval_heff
from stage4_defense import train_detector, noisy_feats, auc


def main():
    d = bd.build_dataset()
    H_tr, H_te, W = d["H_tr"], d["H_test"], d["W"]
    # subsample the test set for a fast CPU AUC estimate (detector diagnostic)
    rng = np.random.default_rng(bd.SEED)
    idx = rng.choice(len(H_te), min(2500, len(H_te)), replace=False)
    H_te, Xc = H_te[idx], d["Xte"][idx]
    victim = BeamMLP(d["Xte"].shape[1], d["n_beams"]).to(DEVICE)
    victim.load_state_dict(torch.load(os.path.join(bd.ART, "victim_model.pt"), map_location=DEVICE)); victim.eval()

    # detector trained exactly as in the defense section (clean vs non-adaptive malicious RIS)
    tri = rng.choice(len(H_tr), min(4000, len(H_tr)), replace=False)
    gtr = ris.build_geometry(H_tr[tri], 128, 1.0, seed=5)
    det = train_detector(d["Xtr"][tri], noisy_feats(ris.ris_attack(victim, gtr, iters=40, seed=5)))
    gte = ris.build_geometry(H_te, 128, 1.0, seed=bd.SEED)   # same operating point as the headline M=128 attack

    out = {"lam": [], "det_auc": [], "victim_top1": [], "victim_se": []}
    # lam = 0 is the ordinary (non-adaptive) attack; lam > 0 is detector-aware
    for lam in [0.0, 2.0, 5.0]:
        if lam == 0.0:
            H_adv = ris.ris_attack(victim, gte, iters=80, seed=99)
        else:
            H_adv = ris.ris_attack_detector_aware(victim, det, gte, lam=lam, iters=80, seed=99)
        e = eval_heff(victim, H_adv, W)
        a = auc(det, Xc, noisy_feats(H_adv))
        out["lam"].append(lam); out["det_auc"].append(a)
        out["victim_top1"].append(e["top1"]); out["victim_se"].append(e["se_ratio"])
        tag = "non-adaptive" if lam == 0 else f"adaptive lam={lam}"
        print(f"  {tag:>16}: detector AUC={a:.3f}  victim top1={e['top1']*100:5.1f}%  SE={e['se_ratio']*100:4.0f}%")

    json.dump(out, open(os.path.join(bd.ART, "detector_adaptive.json"), "w"), indent=2)
    print("Saved detector_adaptive.json")


if __name__ == "__main__":
    main()
