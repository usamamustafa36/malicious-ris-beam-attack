"""Additional experiments: link-budget/kappa sweep, model-blind SNR-jamming baseline,
detector false-positive rate, beam-confusion statistics, gradient-masking sanity
check, and attacker CSI-error sensitivity."""
import os, json, numpy as np, torch
import beamdata as bd
from model import BeamMLP, train_model, predict_probs, DEVICE
import ris
from stage3_ris import eval_heff
from stage4_defense import train_detector, best_beam, noisy_feats, auc, eval_under_attack

rng = np.random.default_rng(bd.SEED)
C128 = np.complex128


def db_ratio(kappa, M):                       # RIS-path-to-direct-path power ratio (dB)
    return 20 * np.log10(kappa * M / ris.M_REF)


def opt_se_retention(H_eff, H_d, W, snr=10):
    """mean achievable SE of best beam on h_eff vs on h_d (channel-strength effect)."""
    gd = (np.abs(H_d.astype(C128) @ W.conj().T) ** 2).max(1)
    ge = (np.abs(H_eff.astype(C128) @ W.conj().T) ** 2).max(1)
    rho = 10 ** (snr / 10) / gd.mean()
    return float(np.mean(np.log2(1 + rho * ge)) / np.mean(np.log2(1 + rho * gd)))


def main():
    d = bd.build_dataset()
    H_te, H_tr, W = d["H_test"], d["H_tr"], d["W"]
    victim = BeamMLP(d["Xte"].shape[1], d["n_beams"]).to(DEVICE)
    victim.load_state_dict(torch.load(os.path.join(bd.ART, "victim_model.pt"))); victim.eval()
    out = {}

    # ---- M1: RIS-to-direct power ratio at each M, and a kappa/ratio sweep ----
    print("M1  RIS-to-direct power ratio (dB) at kappa=1:")
    out["dB_at_M"] = {M: db_ratio(1.0, M) for M in [16, 32, 64, 128, 256]}
    for M, r in out["dB_at_M"].items():
        print(f"     M={M:3d} -> {r:+.1f} dB")
    print("\nM1  kappa sweep at M=128 (severity vs RIS-to-direct ratio):")
    out["kappa_sweep"] = []
    for kappa in [0.1, 0.25, 0.5, 1.0, 2.0]:
        g = ris.build_geometry(H_te, 128, kappa, seed=bd.SEED)
        e = eval_heff(victim, ris.ris_attack(victim, g, iters=80, seed=bd.SEED), W)
        row = {"kappa": kappa, "dB": db_ratio(kappa, 128), "top1": e["top1"], "se": e["se_ratio"]}
        out["kappa_sweep"].append(row)
        print(f"     kappa={kappa:4.2f} ({row['dB']:+5.1f} dB): top1={e['top1']*100:5.1f}%  SE={e['se_ratio']*100:5.1f}%")

    # ---- M3: DNN-blind SNR-jamming vs DNN-aware attack ----
    print("\nM3  SNR-jamming (model-blind) vs DNN-aware attack:")
    out["snr_jam"] = []
    for M in [64, 128]:
        g = ris.build_geometry(H_te, M, 1.0, seed=bd.SEED)
        Hd = ris.ris_attack(victim, g, iters=80, seed=bd.SEED)
        Hj = ris.ris_snr_jam(g, iters=120, seed=bd.SEED)
        ed, ej = eval_heff(victim, Hd, W), eval_heff(victim, Hj, W)
        rd, rj = opt_se_retention(Hd, H_te, W), opt_se_retention(Hj, H_te, W)
        out["snr_jam"].append({"M": M,
                               "dnn": {"top1": ed["top1"], "se": ed["se_ratio"], "opt_se_ret": rd},
                               "jam": {"top1": ej["top1"], "se": ej["se_ratio"], "opt_se_ret": rj}})
        print(f"     M={M}: DNN-aware top1={ed['top1']*100:5.1f}% SEr={ed['se_ratio']*100:4.0f}% optSE={rd*100:4.0f}% |"
              f" SNR-jam top1={ej['top1']*100:5.1f}% SEr={ej['se_ratio']*100:4.0f}% optSE={rj*100:4.0f}%")

    # ---- M4: detector re-evaluation (benign FPR + malicious-vs-benign separability) ----
    print("\nM4  detector re-evaluation:")
    gtr = ris.build_geometry(H_tr, 128, 1.0, seed=5)
    det = train_detector(d["Xtr"], noisy_feats(ris.ris_attack(victim, gtr, iters=60, seed=5)))
    gte = ris.build_geometry(H_te, 128, 1.0, seed=99)
    Xc = d["Xte"]
    Xmal = bd.eval_feats(ris.ris_attack(victim, gte, iters=80, seed=99))   # deterministic eval noise
    torch.manual_seed(bd.SEED)   # seeded benign-RIS control
    Xben = bd.eval_feats(ris._h_eff(gte, torch.rand(len(H_te), 128, device=DEVICE) * 2 * np.pi).cpu().numpy())
    with torch.no_grad():
        from model import to_t
        sc = lambda X: torch.sigmoid(det(to_t(X.astype(np.float32)))).cpu().numpy()
        s_c, s_m, s_b = sc(Xc), sc(Xmal), sc(Xben)
    thr = np.quantile(s_c, 0.95)                         # 5% FPR on clean
    out["detector"] = {
        "auc_clean_vs_malicious": auc(det, Xc, Xmal),
        "auc_benign_vs_malicious": auc(det, Xben, Xmal),   # can it tell malicious from benign RIS?
        "fpr_benign_at_5pct_clean": float(np.mean(s_b > thr)),
        "tpr_malicious_at_5pct_clean": float(np.mean(s_m > thr))}
    for k, v in out["detector"].items():
        print(f"     {k}: {v:.3f}")

    # ---- M6: beam-confusion under attack (rank-one collapse mechanism) ----
    print("\nM6  beam-confusion under M=128 attack:")
    g = ris.build_geometry(H_te, 128, 1.0, seed=bd.SEED)
    pred_adv = predict_probs(victim, bd.eval_feats(ris.ris_attack(victim, g, iters=80, seed=bd.SEED))).argmax(1)
    ang = np.linspace(-1, 1, bd.N_BEAMS, endpoint=False)
    idx_bR = int(np.argmin(np.abs(ang - ris.U_BSR)))     # codebook beam nearest the BS->RIS direction
    p = np.bincount(pred_adv, minlength=bd.N_BEAMS) / len(pred_adv)
    ent = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    out["confusion"] = {"a_bR_beam": idx_bR,
                        "frac_pred_on_a_bR": float(np.mean(pred_adv == idx_bR)),
                        "frac_pred_in_a_bR_pm2": float(np.mean(np.abs(pred_adv - idx_bR) <= 2)),
                        "pred_entropy_bits": ent, "max_entropy_bits": float(np.log2(bd.N_BEAMS))}
    print(f"     a_bR beam idx={idx_bR}; frac preds on it={out['confusion']['frac_pred_on_a_bR']*100:.1f}%"
          f" (+/-2: {out['confusion']['frac_pred_in_a_bR_pm2']*100:.1f}%); entropy={ent:.2f}/{np.log2(bd.N_BEAMS):.0f} bits")

    # ---- gradient-masking sanity check on the defense ----
    print("\nCheck  gradient-masking on defense (more iters + transfer):")
    defended = BeamMLP(d["Xte"].shape[1], d["n_beams"]).to(DEVICE)
    defended.load_state_dict(torch.load(os.path.join(bd.ART, "defended_model.pt"))); defended.eval()
    surrogate = train_model(d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["n_beams"], epochs=40, hidden=(384, 192), seed=2024, verbose=False)
    g = ris.build_geometry(H_te, 128, 1.0, seed=bd.SEED)
    d80 = eval_heff(defended, ris.ris_attack(defended, g, iters=80, seed=bd.SEED), W)["top1"]
    d250 = eval_heff(defended, ris.ris_attack(defended, g, iters=250, lr=0.05, seed=1), W)["top1"]
    dtr = eval_heff(defended, ris.ris_attack(surrogate, g, iters=80, seed=bd.SEED), W)["top1"]
    out["grad_mask_check"] = {"defended_top1_80it": d80, "defended_top1_250it": d250, "defended_top1_transfer": dtr}
    print(f"     defended top1: 80it={d80*100:.1f}%  250it={d250*100:.1f}%  transfer={dtr*100:.1f}%  (stable => not masking)")

    # ---- CSI-error sensitivity of the attacker ----
    print("\nCheck  attacker CSI-error sensitivity (M=128):")
    out["csi_error"] = []
    g_true = ris.build_geometry(H_te, 128, 1.0, seed=bd.SEED)
    for snr_att in [0, 5, 10, 20, 999]:
        H_att = H_te if snr_att == 999 else bd.add_cn_noise(H_te, snr_att, np.random.default_rng(bd.SEED))
        g_att = ris.build_geometry(H_att, 128, 1.0, seed=bd.SEED)
        _, theta = ris.ris_attack(victim, g_att, iters=80, seed=bd.SEED, return_theta=True)
        e = eval_heff(victim, ris.heff_from_theta(g_true, theta), W)
        out["csi_error"].append({"atk_csi_snr_db": snr_att, "top1": e["top1"], "se": e["se_ratio"]})
        lab = "perfect" if snr_att == 999 else f"{snr_att} dB"
        print(f"     attacker CSI {lab:>7}: victim top1={e['top1']*100:5.1f}%  SE={e['se_ratio']*100:5.1f}%")

    json.dump(out, open(os.path.join(bd.ART, "review_response.json"), "w"), indent=2)
    print("\nSaved review_response.json")


if __name__ == "__main__":
    main()
