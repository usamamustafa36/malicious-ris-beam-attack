"""Extension A: validate the malicious-RIS attack on a genuine 28 GHz mmWave scenario
(DeepMIMO city_0_newyork_28). Runs victim training, the RIS aperture sweep, and the
RIS-adversarial-training defense; saves mmwave.json for the paper."""
import os, json, numpy as np, torch
import beamdata as bd
from model import train_model, predict_probs, BeamMLP, DEVICE
import ris
from stage3_ris import eval_heff
from stage4_defense import build_augmented, eval_under_attack

SCEN, CACHE = "city_0_newyork_28", "channels_mmw.npy"
M_SWEEP = [16, 32, 64, 128, 256]


def main():
    d = bd.build_dataset(scenario=SCEN, cache_name=CACHE)
    H_te, W = d["H_test"], d["W"]
    print(f"mmWave (28 GHz) users train/val/test: {len(d['ytr'])}/{len(d['yva'])}/{len(d['yte'])}")

    victim = train_model(d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["n_beams"], epochs=60, verbose=False)
    probs = predict_probs(victim, d["Xte"]); pred = probs.argmax(1)
    acc = bd.topk_acc(probs, d["yte"])
    clean = {"top1": acc[1], "top3": acc[3], "se": bd.se_ratio(H_te, W, pred)}
    print(f"Clean victim: top1={acc[1]*100:.1f}%  top3={acc[3]*100:.1f}%  SE={clean['se']*100:.1f}%")

    out = {"clean": clean, "M_sweep": M_SWEEP, "opt": [], "rand": []}
    print(f"\n{'M':>4} | {'malicious top1':>14} {'SE':>6} | {'random top1':>12}")
    for M in M_SWEEP:
        g = ris.build_geometry(H_te, M, 1.0, seed=bd.SEED)
        a = eval_heff(victim, ris.ris_attack(victim, g, iters=80, seed=bd.SEED), W)
        torch.manual_seed(bd.SEED)   # seeded random-RIS control
        r = eval_heff(victim, ris._h_eff(g, torch.rand(H_te.shape[0], M, device=DEVICE) * 2 * np.pi).cpu().numpy(), W)
        out["opt"].append(a); out["rand"].append(r)
        print(f"{M:4d} | {a['top1']*100:13.1f}% {a['se_ratio']*100:5.1f}% | {r['top1']*100:11.1f}%")

    # defense at M=128
    print("\nRIS-adversarial training (defense) @ M=128 ...")
    Xaug, yaug = build_augmented(victim, d["H_tr"], W)
    defended = train_model(Xaug, yaug, d["Xva"], d["yva"], d["n_beams"], epochs=40, verbose=False)
    u1, uSE = eval_under_attack(victim, H_te, W, 128, seed=bd.SEED)
    d1, dSE = eval_under_attack(defended, H_te, W, 128, seed=bd.SEED)
    dc = predict_probs(defended, d["Xte"]).argmax(1)
    out["defense"] = {"undef_top1": u1, "undef_se": uSE, "def_top1": d1, "def_se": dSE,
                      "def_clean_top1": float(np.mean(dc == d["yte"]))}
    print(f"  under M=128 attack: undef={u1*100:.1f}%  ->  defended={d1*100:.1f}%  "
          f"(defended clean {out['defense']['def_clean_top1']*100:.1f}%)")

    json.dump(out, open(os.path.join(bd.ART, "mmwave.json"), "w"), indent=2)
    print("\nSaved mmwave.json")


if __name__ == "__main__":
    main()
