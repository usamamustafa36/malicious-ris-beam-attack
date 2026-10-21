"""Figures and tables for the TCCN version, built from the revision artifacts.
Each block is skipped if its artifact is not there yet."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import j0
import beamdata as bd
import figstyle as fs

FIG = os.path.join(os.path.dirname(__file__), "figures")
A = lambda n: os.path.join(bd.ART, n)
have = lambda n: os.path.exists(A(n))
load = lambda n: json.load(open(A(n)))
fs.use_style(); C = fs.COLORS
pct = lambda x: f"{100 * x:.1f}"


def ci(xs):
    import revision_common as rc
    return rc.ci95(xs)


def fmt(mu_h):
    mu, h = mu_h
    return f"{100 * mu:.1f}$\\pm${100 * h:.1f}" if h > 0 else f"{100 * mu:.1f}"


# ------------------------------------------------------------------ Table: main results
if have("stage_ci.json"):
    S = load("stage_ci.json")["victims"]
    r, c = S["rsrp"]["summary"], S["csi"]["summary"]
    g = lambda d, cond, m: fmt((d[cond][m]["mean"], d[cond][m]["ci95"]))
    rows = [
        r"\begin{table*}[t]\centering",
        r"\caption{Results at fixed RIS-to-direct power ratios (mean $\pm$ 95\% CI over 5 trials, each averaged over 5 noise realizations). Partial-measurement victim: $L{=}16$ wide-beam RSRPs; $\eta_3$ is the SE ratio after the BS measures the DNN's top-3 narrow beams ($19$ measurements in total). The model-free local sweep measures the $4$ narrow beams under the strongest wide beam ($20$ measurements), so its top-1 already includes refinement and compares with the DNN's top-3 and $\eta_3$ rather than its top-1. Full-CSI victim: model-free argmax on the same noisy CSI.}",
        r"\label{tab:main}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l cccc cc}\toprule",
        r"& \multicolumn{4}{c}{Learned predictor} & \multicolumn{2}{c}{Model-free baseline}\\\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
        r"Condition & Top-1 (\%) & Top-3 (\%) & $\eta$ (\%) & $\eta_3$ (\%) & Top-1 (\%) & $\eta$ (\%)\\\midrule",
        r"\multicolumn{7}{l}{\emph{Partial-measurement predictor (primary)}}\\",
    ]
    for lab, cond, cc in [("No RIS", "clean", "clean_classical"), ("Random RIS, $M{=}128$", "random_128", None),
                          ("Malicious RIS, $0$ dB ($M{=}64$)", "attack_64", "attack_64_classical"),
                          ("Malicious RIS, $+6$ dB ($M{=}128$)", "attack_128", "attack_128_classical")]:
        cl = (g(r, cc, "local_top1") + " & " + g(r, cc, "local_se")) if cc else "-- & --"
        rows.append(f"\\quad {lab} & {g(r, cond, 'top1')} & {g(r, cond, 'top3')} & {g(r, cond, 'se')} & {g(r, cond, 'se_ref3')} & {cl}\\\\")
    rows.append(r"\midrule\multicolumn{7}{l}{\emph{Full-CSI predictor (stress case)}}\\")
    for lab, cond, cc in [("No RIS", "clean", "clean_classical"), ("Random RIS, $M{=}128$", "random_128", None),
                          ("Malicious RIS, $0$ dB ($M{=}64$)", "attack_64", "attack_64_classical"),
                          ("Malicious RIS, $+6$ dB ($M{=}128$)", "attack_128", "attack_128_classical")]:
        cl = (g(c, cc, "argmax_top1") + " & " + g(c, cc, "argmax_se")) if cc else "-- & --"
        rows.append(f"\\quad {lab} & {g(c, cond, 'top1')} & {g(c, cond, 'top3')} & {g(c, cond, 'se')} & {g(c, cond, 'se_ref3')} & {cl}\\\\")
    rows += [r"\bottomrule\end{tabular}}\end{table*}"]
    open(os.path.join(FIG, "table_main_tccn.tex"), "w").write("\n".join(rows) + "\n")
    print("table_main_tccn.tex")

# Legends live in a band under the axes. Nothing is drawn on the data,
# and nothing is parked in the gap under the running header.
LEG = dict(frameon=False, fontsize=6.6, handlelength=1.45, handletextpad=0.35,
           columnspacing=0.85, borderaxespad=0.0)

# ------------------------------------------------------------------ Fig: PHY baselines
if have("stage_baselines_phy.json"):
    R = load("stage_baselines_phy.json")["results"]
    conds = [("dnn_aware", "DNN-aware (ours)", C["ris"]), ("beam_null", "beam-null", "#1b6ca8"),
             ("beam_hijack", "beam-hijack", C["pgd"]), ("snr_jam", "SNR-jam", C["jam"]), ("random", "random", "#9e9e9e")]
    fig, ax = plt.subplots(1, 3, figsize=(fs.TEXT, 2.42))
    panels = [("rsrp", "se_ref3", r"$\eta_3$ (%), partial"),
              ("rsrp", "local_se", r"$\eta$ (%), sweep"),
              ("csi", "se_ref3", r"$\eta_3$ (%), full CSI")]
    w = 0.145
    for a, (kind, met, ylab) in zip(ax, panels):
        for i, (cn, lab, col) in enumerate(conds):
            mus, hs = [], []
            for M in ("64", "128"):
                mu, h = ci([row[cn][met] for row in R[kind][M]])
                mus.append(100 * mu); hs.append(100 * h)
            # Narrower than the pitch so a tall interval does not touch the next bar.
            a.bar(np.arange(2) + (i - 2) * w, mus, w * 0.74, yerr=hs, color=col, label=lab,
                  capsize=1.1, edgecolor="white", linewidth=0.35, error_kw={"lw": 0.55, "capthick": 0.55})
        a.set_xticks([0, 1]); a.set_xticklabels(["0 dB\n($M{=}64$)", "+6 dB\n($M{=}128$)"])
        # Lowest interval is 39.8. A floor of 38 put that cap on the spine.
        a.set_ylim(30, 108); a.set_yticks([40, 60, 80, 100])
        a.set_ylabel(ylab); a.grid(axis="x", visible=False); fs.despine(a)
    for a, t in zip(ax, "abc"):
        fs.tag(a, f"({t})")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.52, 0.012), **LEG)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.90, bottom=0.30, wspace=0.42)
    fig.savefig(os.path.join(FIG, "fig_phybaselines.png"), bbox_inches="tight", pad_inches=0.03); plt.close(fig)
    print("fig_phybaselines.png")

# ------------------------------------------------------------------ Fig: trade-off + aperture sweep
if have("stage_tradeoff.json"):
    T = load("stage_tradeoff.json")
    KS, WS, L = T["KS"], T["WS"], T["L"]
    fig, ax = plt.subplots(1, 2, figsize=(fs.TEXT, 2.85))
    a = ax[0]
    for cond, ls, lab in [("clean", "-", "no RIS"), ("attack_64", "--", "0 dB"), ("attack_128", ":", "+6 dB")]:
        dn = [np.mean([t[cond]["dnn"][str(k)] for t in T["tradeoff"]]) * 100 for k in KS]
        cl = [np.mean([t[cond]["classical"][str(w_)] for t in T["tradeoff"]]) * 100 for w_ in WS]
        a.plot([L + k for k in KS], dn, ls, marker="o", color=C["ris"], label=f"DNN, {lab}")
        a.plot([L + w_ for w_ in WS], cl, ls, marker="s", color=C["fgsm"], label=f"sweep, {lab}")
    a.set_xlabel("Beam measurements per user (exhaustive sweep: 64)"); a.set_ylabel("SE ratio (%)")
    a.set_xlim(16.5, 28.5); a.set_ylim(34, 102)
    h1, l1 = a.get_legend_handles_labels(); fs.despine(a); fs.tag(a, "(a)")
    a = ax[1]; Ms = T["M_sweep"]
    for key, met, col, mk, lab in [("attack", "top1", C["ris"], "o", "malicious, top-1"),
                                   ("attack", "se_ref3", C["se"], "s", r"malicious, $\eta_3$"),
                                   ("random", "se_ref3", C["rand"], "^", r"random, $\eta_3$")]:
        y = np.array([[sw[key][i][met] for i in range(len(Ms))] for sw in T["sweep"]]) * 100
        a.errorbar(Ms, y.mean(0), yerr=y.std(0, ddof=1) * 4.30 / np.sqrt(len(y)), marker=mk, color=col, label=lab, capsize=2)
    c0 = np.mean([sw["clean"]["top1"] for sw in T["sweep"]]) * 100
    a.axhline(c0, color=C["clean"], ls=":", lw=1, label="top-1, no RIS")
    a.set_xscale("log"); a.set_xticks(Ms); a.set_xticklabels([str(m) for m in Ms])
    a.set_xlabel(r"RIS elements $M$ (ratio $20\log_{10}(M/64)$ dB)")
    a.set_ylabel(r"Top-1 or $\eta_3$ (%)"); a.set_ylim(0, 100)
    # axhline is returned first by get_legend_handles_labels; keep attack series
    # in the left column and the two reference lines in the right column.
    h2, l2 = a.get_legend_handles_labels()
    order = [l2.index(s) for s in ("malicious, top-1", r"malicious, $\eta_3$", r"random, $\eta_3$", "top-1, no RIS")]
    h2, l2 = [h2[i] for i in order], [l2[i] for i in order]
    fs.despine(a); fs.tag(a, "(b)")
    # ncol=3 fills down each condition, so "DNN, 0 dB" sits over "sweep, 0 dB".
    fig.legend(h1, l1, loc="lower center", bbox_to_anchor=(0.26, 0.01), ncol=3, **LEG)
    fig.legend(h2, l2, loc="lower center", bbox_to_anchor=(0.76, 0.01), ncol=2, **LEG)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.32, wspace=0.32)
    fig.savefig(os.path.join(FIG, "fig_tradeoff.png"), bbox_inches="tight", pad_inches=0.03); plt.close(fig)
    print("fig_tradeoff.png")

# ------------------------------------------------------------------ Fig: real-time + stale CSI
if have("stage_amortized_rsrp.json") and have("stage_amortized_csi.json"):
    AM = {k: load(f"stage_amortized_{k}.json") for k in ("rsrp", "csi")}
    fig, ax = plt.subplots(1, 2, figsize=(fs.TEXT, 2.85))
    a = ax[0]; conds = [("clean", "no RIS", C["clean"]), ("random", "random RIS", C["rand"]),
                        ("iterative", "iterative", C["ris"]), ("gen_whitebox", "generator, white-box", C["se"]),
                        ("gen_blackbox", "generator, black-box", C["pgd"])]
    groups = [("rsrp", "64"), ("rsrp", "128"), ("csi", "64"), ("csi", "128")]
    w = 0.16
    for i, (cn, lab, col) in enumerate(conds):
        mus, hs = [], []
        for kind, M in groups:
            mu, h = ci([r_[cn]["top1"] for r_ in AM[kind]["results"][M]])
            mus.append(mu * 100); hs.append(h * 100)
        mus, hs = np.array(mus), np.array(hs)
        # One 95% interval on a 3-trial mean crosses 0. A rate stops at 0, so the
        # lower whisker is drawn to the axis and the upper whisker is the full CI.
        lo = np.minimum(hs, mus)
        a.bar(np.arange(4) + (i - 2) * w, mus, w * 0.74, yerr=np.vstack([lo, hs]), color=col, label=lab,
              capsize=1.1, edgecolor="white", linewidth=0.35, error_kw={"lw": 0.55, "capthick": 0.55})
    a.set_xticks(range(4)); a.set_xticklabels(["partial\n$0$ dB", "partial\n$+6$ dB", "full CSI\n$0$ dB", "full CSI\n$+6$ dB"])
    a.set_ylabel("Victim top-1 (%)"); a.set_ylim(0, 105); a.set_yticks([0, 20, 40, 60, 80, 100])
    a.grid(axis="x", visible=False)
    h1, l1 = a.get_legend_handles_labels(); fs.despine(a); fs.tag(a, "(a)")
    a = ax[1]
    rhos = AM["rsrp"]["rhos"]
    for kind, M, col, lab in [("rsrp", "64", C["ris"], "partial, $0$ dB"), ("rsrp", "128", C["se"], "partial, $+6$ dB"),
                              ("csi", "64", C["fgsm"], "full CSI, $0$ dB"), ("csi", "128", C["pgd"], "full CSI, $+6$ dB")]:
        rows_ = AM[kind]["results"][M]
        clean = np.mean([r_["clean"]["top1"] for r_ in rows_])
        y = np.array([[r_["stale_gen"][str(rh)]["top1"] for rh in rhos] for r_ in rows_]) / clean * 100
        a.plot(rhos, y.mean(0), marker="o", color=col, label=lab)
    a.set_xlabel(r"Attacker CSI correlation $\rho$"); a.set_ylabel("Top-1, % of clean")
    a.set_xlim(1.03, -0.03); a.set_ylim(0, 112)
    h2, l2 = a.get_legend_handles_labels(); fs.despine(a); fs.tag(a, "(b)")
    # top axis: Jakes delay at 3.5 GHz, 5 km/h (J0 is monotone up to its first zero)
    fD = (5 / 3.6) * 3.5e9 / 3e8
    taus = [0, 5, 10, 15, 20]
    top = a.twiny(); top.set_xlim(a.get_xlim())
    top.set_xticks([j0(2 * np.pi * fD * t * 1e-3) for t in taus]); top.set_xticklabels([str(t) for t in taus])
    top.set_xlabel("CSI delay (ms) at 3.5 GHz, 5 km/h", fontsize=7); top.tick_params(labelsize=6.5)
    top.grid(False)
    fig.legend(h1, l1, loc="lower center", bbox_to_anchor=(0.26, 0.012), ncol=2, **LEG)
    fig.legend(h2, l2, loc="lower center", bbox_to_anchor=(0.76, 0.012), ncol=2, **LEG)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.32, wspace=0.34)
    fig.savefig(os.path.join(FIG, "fig_realtime.png"), bbox_inches="tight", pad_inches=0.03); plt.close(fig)
    print("fig_realtime.png")

# ------------------------------------------------------------------ Fig: RIS channel model
if have("stage_multipath.json"):
    MP = load("stage_multipath.json")
    chans = MP["channels"]
    fig, ax = plt.subplots(1, 2, figsize=(fs.TEXT, 2.55))
    ticklab = ["LoS", r"$K{=}10$ dB", r"$K{=}0$ dB", "scattered", r"$K{=}0$+RU"]
    for a, kind, ttl in [(ax[0], "rsrp", "partial"), (ax[1], "csi", "full CSI")]:
        res = MP["results"][kind]
        for M, ls in (("64", "-"), ("128", "--")):
            for cn, col, lab in (("dnn_aware", C["ris"], "DNN-aware"), ("snr_jam", C["jam"], "SNR-jam")):
                mu = [res[f"{ch}_M{M}"][cn]["se_ref3"][0] * 100 for ch in chans]
                h = [res[f"{ch}_M{M}"][cn]["se_ref3"][1] * 100 for ch in chans]
                a.errorbar(range(len(chans)), mu, yerr=h, ls=ls, marker="o", color=col, capsize=2,
                           label=f"{lab}, {'0 dB' if M == '64' else '+6 dB'}")
        a.set_xticks(range(len(chans))); a.set_xticklabels(ticklab, fontsize=7.2)
        a.set_ylabel(rf"$\eta_3$ (%), {ttl}"); a.set_ylim(28, 112); fs.despine(a)
    fs.tag(ax[0], "(a)"); fs.tag(ax[1], "(b)")
    hnd, lab = ax[0].get_legend_handles_labels()
    fig.legend(hnd, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.52, 0.012), **LEG)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.24, wspace=0.28)
    fig.savefig(os.path.join(FIG, "fig_multipath.png"), bbox_inches="tight", pad_inches=0.03); plt.close(fig)
    print("fig_multipath.png")
