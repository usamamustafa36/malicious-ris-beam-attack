"""Stage 2: evaluate baseline white-box attacks vs epsilon; save results + table."""
import os, json, numpy as np, torch
import beamdata as bd
from model import BeamMLP, predict_probs, DEVICE
import attacks

EPS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]


def evaluate(model, Xadv, H, W, y):
    probs = predict_probs(model, Xadv)
    pred = probs.argmax(1)
    acc = bd.topk_acc(probs, y)
    return {"top1": acc[1], "top3": acc[3], "se_ratio": bd.se_ratio(H, W, pred)}


def main():
    d = bd.build_dataset()
    H, W, y, X = d["H_test"], d["W"], d["yte"], d["Xte"]
    m = BeamMLP(X.shape[1], d["n_beams"]).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(bd.ART, "victim_model.pt"))); m.eval()

    out = {"eps": EPS, "fgsm": [], "pgd": []}
    clean = evaluate(m, X, H, W, y)
    print(f"{'eps':>6} | {'FGSM top1':>10} {'FGSM SE':>8} | {'PGD top1':>10} {'PGD SE':>8}")
    for e in EPS:
        if e == 0.0:
            f = p = clean
        else:
            f = evaluate(m, attacks.fgsm(m, X, y, e), H, W, y)
            torch.manual_seed(bd.SEED)   # seed PGD random start for reproducibility
            p = evaluate(m, attacks.pgd(m, X, y, e, iters=10), H, W, y)
        out["fgsm"].append(f); out["pgd"].append(p)
        print(f"{e:6.2f} | {f['top1']*100:9.1f}% {f['se_ratio']*100:7.1f}% |"
              f" {p['top1']*100:9.1f}% {p['se_ratio']*100:7.1f}%")

    # CW at a fixed budget (its own knob is c, not eps)
    cw = evaluate(m, attacks.cw(m, X, y, c=1.0, iters=100), H, W, y)
    out["cw"] = cw
    out["clean"] = clean
    print(f"\nClean         : top1={clean['top1']*100:.1f}%  SE={clean['se_ratio']*100:.1f}%")
    print(f"CW (c=1,100it): top1={cw['top1']*100:.1f}%  SE={cw['se_ratio']*100:.1f}%"
          f"   (correct margin objective -> should DROP accuracy, not raise it)")

    json.dump(out, open(os.path.join(bd.ART, "stage2_attacks.json"), "w"), indent=2)
    print("\nSaved stage2_attacks.json")


if __name__ == "__main__":
    main()
