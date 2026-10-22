"""Second predictor on the same partial-measurement features.

The primary victim is a two-hidden-layer perceptron. This stage trains a 1-D
convolutional network on the same L wide-beam powers and, separately, a
histogram gradient-boosted tree. The convolutional network is attacked
white-box with the same phase objective. The tree is not differentiable; it is
scored on the channels produced by the attack against the perceptron.
"""
import json, os
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
import beamdata as bd
import ris
import revision_common as rc
from model import train_model, DEVICE

SEEDS = [42, 7, 2024]


class RSRPConv(nn.Module):
    """Convolution along the ordered wide-beam index. Neighboring inputs are
    neighboring angles, which the fully connected perceptron does not assume."""

    def __init__(self, L, n_beams, drop=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU())
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * L, 128), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(128, n_beams))

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


def train_conv(d, seed):
    rng = np.random.default_rng(seed)
    W, C = d["W"], rc.wide_codebook()
    lab = lambda H: np.abs(H.astype(np.complex128) @ W.conj().T).argmax(1)
    feat = lambda H: rc.rsrp_np(bd.add_cn_noise(H, bd.PILOT_SNR_DB, rng), C)
    model = RSRPConv(C.shape[0], d["n_beams"])
    m = train_model(feat(d["H_tr"]), lab(d["H_tr"]), feat(d["H_va"]), lab(d["H_va"]),
                    d["n_beams"], epochs=60, seed=seed, model=model, verbose=False)
    return rc.Victim("rsrp", m, C)


def train_tree(d, seed):
    rng = np.random.default_rng(seed)
    W, C = d["W"], rc.wide_codebook()
    X = rc.rsrp_np(bd.add_cn_noise(d["H_tr"], bd.PILOT_SNR_DB, rng), C)
    y = np.abs(d["H_tr"].astype(np.complex128) @ W.conj().T).argmax(1)
    clf = HistGradientBoostingClassifier(
        max_iter=40, max_depth=6, learning_rate=0.1, random_state=seed)
    clf.fit(X, y)
    return clf, C


def eval_tree(clf, C, H, W):
    tops, hit3, se, se3 = [], [], [], []
    for n in rc.NOISE_SEEDS:
        X = rc.rsrp_np(bd.add_cn_noise(H, bd.PILOT_SNR_DB, np.random.default_rng(n)), C)
        proba = clf.predict_proba(X)
        # classes_ may omit a beam that never appears in training
        full = np.zeros((len(X), W.shape[0]))
        full[:, clf.classes_] = proba
        order = np.argsort(-full, axis=1)
        gains = np.abs(H.astype(np.complex128) @ W.conj().T) ** 2
        y = gains.argmax(1)
        meas = np.abs(bd.add_cn_noise(H, bd.PILOT_SNR_DB, np.random.default_rng(n)).astype(np.complex128)
                      @ W.conj().T) ** 2
        top3 = order[:, :3]
        refined = top3[np.arange(len(y)), np.take_along_axis(meas, top3, 1).argmax(1)]
        tops.append(float(np.mean(order[:, 0] == y)))
        hit3.append(float(np.mean((top3 == y[:, None]).any(1))))
        se.append(bd.se_ratio(H, W, order[:, 0]))
        se3.append(bd.se_ratio(H, W, refined))
    return {"top1": float(np.mean(tops)), "top3": float(np.mean(hit3)),
            "se": float(np.mean(se)), "se_ref3": float(np.mean(se3))}


def pack(rows, key):
    out = {}
    for metric in rows[0][key]:
        mu, h = rc.ci95([r[key][metric] for r in rows])
        out[metric] = {"mean": mu, "ci95": h}
    return out


def main():
    d = bd.build_dataset()
    H_te, W = d["H_test"], d["W"]
    conv_rows, tree_rows = [], []
    for s in SEEDS:
        conv = train_conv(d, s)
        tree, C = train_tree(d, s)
        row_c = {"seed": s, "clean": rc.evaluate_noise_avg(conv, H_te, W)}
        row_t = {"seed": s, "clean": eval_tree(tree, C, H_te, W)}
        print(f"[s={s}] conv clean top1 {row_c['clean']['top1']*100:.1f}  "
              f"tree clean top1 {row_t['clean']['top1']*100:.1f}", flush=True)
        for M, tag in ((64, "0dB"), (128, "6dB")):
            g = ris.build_geometry(H_te, M, 1.0, seed=s)
            He_c = rc.ris_attack(conv, g, seed=s)
            He_m = rc.ris_attack(rc.train_victim("rsrp", d, seed=s), g, seed=s)
            row_c[tag] = rc.evaluate_noise_avg(conv, He_c, W)
            row_c[tag + "_transfer"] = rc.evaluate_noise_avg(conv, He_m, W)
            row_t[tag + "_transfer"] = eval_tree(tree, C, He_m, W)
            print(f"[s={s}] {tag} conv white-box top1 {row_c[tag]['top1']*100:.1f} "
                  f"se3 {row_c[tag]['se_ref3']*100:.1f} | transfer conv {row_c[tag + '_transfer']['top1']*100:.1f} "
                  f"tree {row_t[tag + '_transfer']['top1']*100:.1f}", flush=True)
        conv_rows.append(row_c)
        tree_rows.append(row_t)
        json.dump({"conv_trials": conv_rows, "tree_trials": tree_rows},
                  open(os.path.join(bd.ART, "stage_arch2.json"), "w"), indent=2)
    summary = {
        "conv_clean": pack(conv_rows, "clean"),
        "tree_clean": pack(tree_rows, "clean"),
    }
    for tag in ("0dB", "6dB"):
        summary["conv_" + tag] = pack(conv_rows, tag)
        summary["conv_" + tag + "_from_mlp"] = pack(conv_rows, tag + "_transfer")
        summary["tree_" + tag + "_from_mlp"] = pack(tree_rows, tag + "_transfer")
    out = {"seeds": SEEDS, "summary": summary, "conv_trials": conv_rows, "tree_trials": tree_rows}
    json.dump(out, open(os.path.join(bd.ART, "stage_arch2.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
