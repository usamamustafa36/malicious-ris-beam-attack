"""Stage 1: train the victim beam-prediction DNN (PyTorch) and report honest metrics."""
import os, json, numpy as np, torch
import beamdata as bd
from model import train_model, predict_probs, DEVICE


def main():
    print("Device:", DEVICE)
    d = bd.build_dataset()
    m = train_model(d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["n_beams"])

    probs = predict_probs(m, d["Xte"])
    pred = np.argmax(probs, axis=1)
    acc = bd.topk_acc(probs, d["yte"])
    se = bd.se_ratio(d["H_test"], d["W"], pred)
    res = {"top1": acc[1], "top3": acc[3], "top5": acc[5], "se_ratio": se}
    print("\n=== VICTIM MODEL (clean CSI) ===")
    print(f"  Top-1 acc : {acc[1]*100:5.1f}%   (naive majority = 9.2%)")
    print(f"  Top-3 acc : {acc[3]*100:5.1f}%")
    print(f"  Top-5 acc : {acc[5]*100:5.1f}%")
    print(f"  SE ratio  : {se*100:5.1f}%   (fraction of optimal-beam spectral efficiency)")

    torch.save(m.state_dict(), os.path.join(bd.ART, "victim_model.pt"))
    json.dump(res, open(os.path.join(bd.ART, "stage1_metrics.json"), "w"), indent=2)
    print("\nSaved victim_model.pt + stage1_metrics.json")


if __name__ == "__main__":
    main()
