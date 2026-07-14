"""Extension B: black-box / transfer malicious-RIS attack.

The attacker has NO access to the victim DNN. It trains its own SURROGATE beam
predictor (different architecture + seed), optimises the RIS phases against the
surrogate, and transfers the resulting RIS configuration to the victim. We compare
white-box (attack the victim directly) vs transfer vs a random-RIS control.
"""
import os, json, numpy as np, torch
import beamdata as bd
from model import BeamMLP, train_model, predict_probs, DEVICE
import ris

KAPPA, rng = 1.0, np.random.default_rng(bd.SEED)


def best_beam(H, W):
    return np.abs(H.astype(np.complex128) @ W.conj().T).argmax(1)


def eval_on(model, H_eff, W):
    Xn = bd.eval_feats(H_eff)   # shared deterministic eval-noise realization
    pred = predict_probs(model, Xn).argmax(1)
    return {"top1": float(np.mean(pred == best_beam(H_eff, W))), "se": bd.se_ratio(H_eff, W, pred)}


def main():
    d = bd.build_dataset()
    H_te, W = d["H_test"], d["W"]
    victim = BeamMLP(d["Xte"].shape[1], d["n_beams"]).to(DEVICE)
    victim.load_state_dict(torch.load(os.path.join(bd.ART, "victim_model.pt"))); victim.eval()

    # attacker's surrogate: different architecture + seed (no victim access)
    print("Training attacker surrogate (different arch/seed)...")
    surrogate = train_model(d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["n_beams"],
                            epochs=40, hidden=(384, 192), seed=2024, verbose=False)
    ps = predict_probs(surrogate, d["Xte"]).argmax(1)
    print(f"  surrogate clean top-1: {np.mean(ps==d['yte'])*100:.1f}% (victim 89.8%)")

    res = {"M": [64, 128], "whitebox": [], "transfer": [], "random": [], "clean": eval_on(victim, H_te, W)}
    print(f"\n{'M':>4} | {'white-box':>9} | {'transfer':>9} | {'random':>7}   (victim top-1 %)")
    for M in (64, 128):
        g = ris.build_geometry(H_te, M, KAPPA, seed=bd.SEED)
        H_wb = ris.ris_attack(victim, g, iters=80, seed=bd.SEED)        # attack victim
        H_tr = ris.ris_attack(surrogate, g, iters=80, seed=bd.SEED)     # attack surrogate
        torch.manual_seed(bd.SEED)                                      # seeded random-RIS control
        H_rd = ris._h_eff(g, torch.rand(H_te.shape[0], M, device=DEVICE) * 2 * np.pi).cpu().numpy()
        wb, tr, rd = eval_on(victim, H_wb, W), eval_on(victim, H_tr, W), eval_on(victim, H_rd, W)
        res["whitebox"].append(wb); res["transfer"].append(tr); res["random"].append(rd)
        print(f"{M:4d} | {wb['top1']*100:8.1f}% | {tr['top1']*100:8.1f}% | {rd['top1']*100:6.1f}%")

    print(f"\nClean victim top-1: {res['clean']['top1']*100:.1f}%")
    print("Transfer (black-box) attack degrades the victim without any model access.")
    json.dump(res, open(os.path.join(bd.ART, "blackbox.json"), "w"), indent=2)
    print("Saved blackbox.json")


if __name__ == "__main__":
    main()
