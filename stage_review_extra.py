"""Additional self-contained experiments (do NOT overwrite existing artifacts):

  (1) C&W sign-error substantiation (M5): run the BUGGY and the CORRECTED
      untargeted-margin C&W on the same victim and report both top-1 numbers,
      making the "sign error made the attack look ineffective" claim verifiable.
  (2) Predicted-beam distribution under the M=128 malicious-RIS attack (M6): the
      histogram behind the entropy=5.15 / 4.3%-on-a_bR statistic, for a figure.

Writes artifacts/extra.json.
"""
import os, json, numpy as np, torch
import beamdata as bd
from model import BeamMLP, predict_probs, DEVICE
import attacks, ris

rng = np.random.default_rng(bd.SEED)


def main():
    d = bd.build_dataset()
    H, W, y, X = d["H_test"], d["W"], d["yte"], d["Xte"]
    victim = BeamMLP(X.shape[1], d["n_beams"]).to(DEVICE)
    victim.load_state_dict(torch.load(os.path.join(bd.ART, "victim_model.pt"))); victim.eval()
    out = {}

    # ---- (1) C&W sign error: buggy vs corrected margin ----
    clean_top1 = float(np.mean(predict_probs(victim, X).argmax(1) == y))
    Xbug = attacks.cw(victim, X, y, c=1.0, iters=100, buggy=True)
    Xfix = attacks.cw(victim, X, y, c=1.0, iters=100, buggy=False)
    bug_top1 = float(np.mean(predict_probs(victim, Xbug).argmax(1) == y))
    fix_top1 = float(np.mean(predict_probs(victim, Xfix).argmax(1) == y))
    out["cw"] = {"clean_top1": clean_top1, "buggy_top1": bug_top1, "corrected_top1": fix_top1}
    print("M5  C&W sign error:")
    print(f"     clean                             top1={clean_top1*100:5.1f}%")
    print(f"     buggy  margin max(f_other-f_true) top1={bug_top1*100:5.1f}%  (looks ineffective)")
    print(f"     correct margin max(f_true-f_other) top1={fix_top1*100:5.1f}%  (true attack)")

    # ---- (2) predicted-beam distribution under M=128 attack ----
    g = ris.build_geometry(H, 128, 1.0, seed=bd.SEED)
    H_adv = ris.ris_attack(victim, g, iters=80, seed=bd.SEED)
    Xn = bd.eval_feats(H_adv)   # shared deterministic eval-noise realization
    pred_adv = predict_probs(victim, Xn).argmax(1)
    hist = np.bincount(pred_adv, minlength=bd.N_BEAMS).astype(int)
    p = hist / hist.sum()
    ent = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    ang = np.linspace(-1, 1, bd.N_BEAMS, endpoint=False)
    idx_bR = int(np.argmin(np.abs(ang - ris.U_BSR)))
    out["pred_hist"] = {"counts": hist.tolist(), "a_bR_beam": idx_bR,
                        "entropy_bits": ent, "max_entropy_bits": float(np.log2(bd.N_BEAMS)),
                        "frac_on_a_bR": float(np.mean(pred_adv == idx_bR))}
    print("\nM6  predicted-beam spread under M=128 attack:")
    print(f"     entropy={ent:.2f}/{np.log2(bd.N_BEAMS):.0f} bits; a_bR beam={idx_bR}; "
          f"frac on it={out['pred_hist']['frac_on_a_bR']*100:.1f}%; "
          f"nonzero beams={int((hist>0).sum())}/{bd.N_BEAMS}")

    json.dump(out, open(os.path.join(bd.ART, "extra.json"), "w"), indent=2)
    print("\nSaved extra.json")


if __name__ == "__main__":
    main()
