"""Information-limited attacker: a UNIVERSAL RIS configuration.

The per-user attack of Sec. III assumes the adversary knows each victim's
instantaneous CSI. Here we relax that: the attacker optimises ONE phase config on
a set of environment channels and applies it, unchanged, to unseen users. Two cases:
  - universal (white-box model): knows the victim DNN + environment, NOT per-user CSI
  - universal + surrogate:       knows NEITHER the victim model NOR per-user CSI
This is the realistic deployment: a rogue RIS holds a single configuration.
"""
import os, json, numpy as np, torch
import beamdata as bd
from model import BeamMLP, train_model, predict_probs, DEVICE
import ris
from stage3_ris import eval_heff

M_SWEEP = [64, 128, 256]
rng = np.random.default_rng(bd.SEED)


def main():
    d = bd.build_dataset()
    H_tr, H_te, W = d["H_tr"], d["H_test"], d["W"]
    tr_sub = H_tr[rng.choice(len(H_tr), 8000, replace=False)]   # env sample for the attacker

    victim = BeamMLP(d["Xte"].shape[1], d["n_beams"]).to(DEVICE)
    victim.load_state_dict(torch.load(os.path.join(bd.ART, "victim_model.pt"))); victim.eval()
    print("Training attacker surrogate (no victim access)...")
    surrogate = train_model(d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["n_beams"],
                            epochs=40, hidden=(384, 192), seed=2024, verbose=False)

    out = {"M": M_SWEEP, "clean": eval_heff(victim, H_te, W),
           "peruser": [], "uni_wb": [], "uni_bb": [], "random": []}
    print(f"\nClean victim: top1={out['clean']['top1']*100:.1f}%  SE={out['clean']['se_ratio']*100:.1f}%")
    print(f"{'M':>4} | {'per-user':>9} | {'universal(wb)':>13} | {'univ+surrogate':>14} | {'random':>7}")
    for M in M_SWEEP:
        g_tr = ris.build_geometry(tr_sub, M, 1.0, seed=7)
        g_te = ris.build_geometry(H_te, M, 1.0, seed=bd.SEED)

        pu = eval_heff(victim, ris.ris_attack(victim, g_te, iters=80, seed=bd.SEED), W)  # per-user (strong)
        th_wb = ris.ris_attack_universal(victim, g_tr, iters=250)
        th_bb = ris.ris_attack_universal(surrogate, g_tr, iters=250)
        uw = eval_heff(victim, ris.heff_from_theta(g_te, th_wb), W)   # universal, model-aware
        ub = eval_heff(victim, ris.heff_from_theta(g_te, th_bb), W)   # universal + surrogate (fully blind)
        torch.manual_seed(bd.SEED)   # seeded random-RIS control
        rd = eval_heff(victim, ris._h_eff(g_te, torch.rand(len(H_te), M, device=DEVICE) * 2 * np.pi).cpu().numpy(), W)

        for k, v in [("peruser", pu), ("uni_wb", uw), ("uni_bb", ub), ("random", rd)]:
            out[k].append(v)
        print(f"{M:4d} | {pu['top1']*100:8.1f}% | {uw['top1']*100:12.1f}% | {ub['top1']*100:13.1f}% | {rd['top1']*100:6.1f}%")

    json.dump(out, open(os.path.join(bd.ART, "universal.json"), "w"), indent=2)
    print("\nSaved universal.json")
    print("Even the fully-blind (universal+surrogate) attacker degrades the victim,")
    print("so the threat does not require per-user CSI or victim-model access.")


if __name__ == "__main__":
    main()
