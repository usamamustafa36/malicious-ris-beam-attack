"""Revision E6: explicit link budgets that ground the RIS-to-direct power ratio.

Far-field RIS path gain (Ozdogan, Bjornson, Larsson, IEEE WCL 2020; consistent with the
measurements of Tang et al., IEEE TWC 2021): a surface of area A with reflection
efficiency zeta, incidence/reflection angles th_i/th_r, and hop lengths d1 (BS->RIS),
d2 (RIS->UE) has amplitude gain
        a_RIS = sqrt(zeta) * A * cos(th_i) cos(th_r) / (4 pi d1 d2),
while the direct path of length d0 with excess (blockage/penetration) loss L_b dB has
        a_dir = lambda / (4 pi d0) * 10^(-L_b/20).
The RIS-to-direct power ratio is 20 log10(a_RIS / a_dir). With half-wavelength elements,
A = M lambda^2 / 4. We tabulate representative deployments and the aperture needed for
the 0 dB and +6 dB operating points used in the paper.
"""
import os, json, numpy as np
import beamdata as bd

C0 = 3e8
ZETA_DB = -1.0           # reflection efficiency (~80% power)
TH_I = TH_R = np.deg2rad(30.0)

SCENARIOS = [
    # name, fc, d0, d1, d2, blockage dB
    ("3.5 GHz, LoS direct", 3.5e9, 50.0, 40.0, 10.0, 0.0),
    ("3.5 GHz, NLoS (15 dB)", 3.5e9, 50.0, 40.0, 10.0, 15.0),
    ("3.5 GHz, indoor LoS", 3.5e9, 20.0, 15.0, 5.0, 0.0),
    ("28 GHz, LoS direct", 28e9, 50.0, 40.0, 10.0, 0.0),
    ("28 GHz, NLoS (20 dB)", 28e9, 50.0, 40.0, 10.0, 20.0),
]


def ratio_db(M, fc, d0, d1, d2, Lb):
    lam = C0 / fc
    A = M * lam ** 2 / 4
    a_ris = np.sqrt(10 ** (ZETA_DB / 10)) * A * np.cos(TH_I) * np.cos(TH_R) / (4 * np.pi * d1 * d2)
    a_dir = lam / (4 * np.pi * d0) * 10 ** (-Lb / 20)
    return 20 * np.log10(a_ris / a_dir)


def m_needed(target_db, fc, d0, d1, d2, Lb):
    return ratio_db(1, fc, d0, d1, d2, Lb), int(np.ceil(10 ** ((target_db - ratio_db(1, fc, d0, d1, d2, Lb)) / 20)))


def main():
    rows = []
    for name, fc, d0, d1, d2, Lb in SCENARIOS:
        lam = C0 / fc
        r = {"name": name, "fc_GHz": fc / 1e9, "d0": d0, "d1": d1, "d2": d2, "blockage_dB": Lb}
        for tgt in (0.0, 6.0):
            _, M = m_needed(tgt, fc, d0, d1, d2, Lb)
            side = np.sqrt(M * lam ** 2 / 4)
            r[f"M_for_{tgt:+.0f}dB"] = M
            r[f"side_m_for_{tgt:+.0f}dB"] = float(side)
        for M in (256, 1024, 4096):
            r[f"ratio_dB_M{M}"] = float(ratio_db(M, fc, d0, d1, d2, Lb))
        rows.append(r)
        print(f"{name:24s} M(0dB)={r['M_for_+0dB']:6d} ({r['side_m_for_+0dB']:.2f} m side)  "
              f"M(+6dB)={r['M_for_+6dB']:6d} ({r['side_m_for_+6dB']:.2f} m)  "
              f"ratio@M=1024: {r['ratio_dB_M1024']:+.1f} dB")
    json.dump({"zeta_dB": ZETA_DB, "theta_deg": 30.0, "rows": rows},
              open(os.path.join(bd.ART, "stage_linkbudget.json"), "w"), indent=2)

    lines = [r"\begin{table}[t]\centering",
             r"\caption{Link budgets behind the RIS-to-direct power ratio: elements $M$ (and square side) needed for the $0$\,dB and $+6$\,dB operating points ($\lambda/2$ elements, $\zeta{=}{-}1$\,dB, $30^\circ$ incidence/reflection, $d_1{=}40$\,m BS--RIS unless noted).}",
             r"\label{tab:linkbudget}",
             r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{lccc}\toprule",
             r"Deployment ($d_0$/$d_1$/$d_2$, m) & $0$\,dB & $+6$\,dB & ratio @ $M{=}1024$\\\midrule"]
    for r in rows:
        lines.append(f"{r['name']} ({r['d0']:g}/{r['d1']:g}/{r['d2']:g}) & "
                     f"{r['M_for_+0dB']} ({r['side_m_for_+0dB']:.2f}\\,m) & "
                     f"{r['M_for_+6dB']} ({r['side_m_for_+6dB']:.2f}\\,m) & "
                     f"${r['ratio_dB_M1024']:+.1f}$\\,dB\\\\")
    lines += [r"\bottomrule\end{tabular}}\end{table}"]
    open(os.path.join(os.path.dirname(__file__), "figures", "table_linkbudget.tex"), "w").write("\n".join(lines) + "\n")
    print("Saved stage_linkbudget.json + figures/table_linkbudget.tex")


if __name__ == "__main__":
    main()
