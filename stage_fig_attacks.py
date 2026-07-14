"""Combined figure: (a) white-box feature-space baselines vs epsilon and
(b) black-box transferability, merged to reduce the figure count."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import beamdata as bd
import figstyle as fs

FIG = os.path.join(os.path.dirname(__file__), "figures")
s2 = json.load(open(os.path.join(bd.ART, "stage2_attacks.json")))
bx = json.load(open(os.path.join(bd.ART, "blackbox.json")))
fs.use_style(); C = fs.COLORS

fig, ax = fs.fig_2panel_wide(h=1.9)

# (a) feature-space baselines: top-1 vs epsilon
eps = s2["eps"]
ax[0].plot(eps, [x["top1"] * 100 for x in s2["fgsm"]], "o-", color=C["fgsm"], label="FGSM")
ax[0].plot(eps, [x["top1"] * 100 for x in s2["pgd"]], "s-", color=C["pgd"], label="PGD")
ax[0].axhline(s2["cw"]["top1"] * 100, ls="--", color=C["cw"], lw=1.3, label="C&W ($c{=}1$)")
ax[0].set_xlabel(r"budget $\epsilon$"); ax[0].set_ylabel("Top-1 accuracy (%)")
fs.despine(ax[0]); fs.tag(ax[0], "(a)"); ax[0].legend(loc="upper right")

# (b) black-box transferability bars
Ms = bx["M"]; x = np.arange(len(Ms)); w = 0.26
ax[1].bar(x - w, [v["top1"] * 100 for v in bx["whitebox"]], w, color=C["ris"], label="white-box")
ax[1].bar(x, [v["top1"] * 100 for v in bx["transfer"]], w, color=C["fgsm"], label="transfer")
ax[1].bar(x + w, [v["top1"] * 100 for v in bx["random"]], w, color=C["rand"], label="random")
ax[1].axhline(bx["clean"]["top1"] * 100, ls=":", color=C["clean"], lw=1.2, label="no RIS")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"$M{{=}}{m}$" for m in Ms])
ax[1].set_ylabel("Victim top-1 (%)"); ax[1].set_ylim(0, 108)
ax[1].grid(axis="x", visible=False); fs.despine(ax[1]); fs.tag(ax[1], "(b)")
ax[1].legend(loc="upper center", ncol=4, columnspacing=1.0, handlelength=1.3)

fig.savefig(os.path.join(FIG, "fig_attacks.png")); plt.close(fig)
print("Saved fig_attacks.png")
