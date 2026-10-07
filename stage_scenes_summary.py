"""Summarize the ray-traced RIS study across all placements (Table: tab:scenes; writes figures/table_scenes.tex)."""
import os, json
import beamdata as bd

PLACEMENTS = [("Munich A", "stage_raytraced.json"), ("Munich B", "stage_raytraced_munichB.json"),
              ("Etoile A", "stage_raytraced_etoile.json"), ("Etoile B", "stage_raytraced_etoileB.json"),
              ("Florence", "stage_raytraced_florence.json")]


def main():
    rows = []
    for name, f in PLACEMENTS:
        p = os.path.join(bd.ART, f)
        if not os.path.exists(p):
            print(f"{name}: missing {f}"); continue
        d = json.load(open(p))
        if "csi" not in d["results"]:
            print(f"{name}: incomplete"); continue
        R, C = d["results"]["rsrp"], d["results"]["csi"]
        m = lambda res, k, met: res[k][met][0] * 100
        row = {"name": name, "n": d["n_test_visible"], "median64": d["ratio_db"]["64"]["pct"][2],
               "ge0": d["ratio_db"]["64"]["frac_bins"][3] * 100,
               "clean_top1": m(R, "clean", "top1"), "clean_se3": m(R, "clean", "se_ref3"), "clean_sweep": m(R, "clean", "local_se"),
               "n6_top1": m(R, "norm6_dnn_aware_all", "top1"), "n6_se3": m(R, "norm6_dnn_aware_all", "se_ref3"),
               "n6_sweep": m(R, "norm6_dnn_aware_all", "local_se"),
               "n6_csi": m(C, "norm6_dnn_aware_all", "top1"), "n6_argmax": m(C, "norm6_dnn_aware_all", "argmax_top1"),
               "p64_rand_top1": m(R, "phys64_random_all", "top1"), "p64_top1": m(R, "phys64_dnn_aware_all", "top1"),
               "p64_se3": m(R, "phys64_dnn_aware_all", "se_ref3"), "p64_sweep": m(R, "phys64_dnn_aware_all", "local_se")}
        rows.append(row)
        print(" | ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    json.dump(rows, open(os.path.join(bd.ART, "stage_scenes_summary.json"), "w"), indent=2)
    tex = [r"\begin{table*}[t]\centering",
           r"\caption{Ray-traced RIS placements (Sionna RT with diffraction, $3.5$\,GHz, $64{\times}64$ RIS of $2.74$\,m; 3 trials). "
           r"Users: test users the RIS reaches. Ratio: median achievable RIS-to-direct ratio and share of users at $\ge0$\,dB. "
           r"Partial-measurement victim: top-1 / $\eta_3$ / sweep $\eta$ (\%) without RIS, under the attack at a fixed $+6$\,dB ratio, and with "
           r"physical path loss under a random RIS (top-1) and the attack. Full-CSI victim at $+6$\,dB: DNN / argmax top-1 (\%).}",
           r"\label{tab:scenes}",
           r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{lccccccc}\toprule",
           r"Placement & Users & Ratio (dB / $\ge0$) & No RIS & Fixed $+6$\,dB & Physical, random & Physical, attack & Full CSI $+6$\,dB\\\midrule"]
    for r in rows:
        tex.append(f"{r['name']} & {r['n']} & ${r['median64']:+.1f}$ / {r['ge0']:.0f}\\% & "
                   f"{r['clean_top1']:.1f} / {r['clean_se3']:.1f} / {r['clean_sweep']:.1f} & "
                   f"{r['n6_top1']:.1f} / {r['n6_se3']:.1f} / {r['n6_sweep']:.1f} & {r['p64_rand_top1']:.1f} & "
                   f"{r['p64_top1']:.1f} / {r['p64_se3']:.1f} / {r['p64_sweep']:.1f} & {r['n6_csi']:.1f} / {r['n6_argmax']:.1f}\\\\")
    tex += [r"\bottomrule\end{tabular}}", r"\end{table*}"]
    open(os.path.join(os.path.dirname(bd.ART), "figures", "table_scenes.tex"), "w").write("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
