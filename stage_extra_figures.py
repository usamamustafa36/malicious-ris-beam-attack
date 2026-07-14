"""Figures for the two extensions: 28 GHz mmWave sweep + black-box transferability."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import beamdata as bd
import figstyle as fs

FIG = os.path.join(os.path.dirname(__file__), "figures")
mm = json.load(open(os.path.join(bd.ART, "mmwave.json")))
bx = json.load(open(os.path.join(bd.ART, "blackbox.json")))
fs.use_style(); C = fs.COLORS


# ---- mmWave sweep (28 GHz) ----
M = mm["M_sweep"]
o1 = [x["top1"] * 100 for x in mm["opt"]]; r1 = [x["top1"] * 100 for x in mm["rand"]]
ose = [x["se_ratio"] * 100 for x in mm["opt"]]; rse = [x.get("se_ratio", 0) * 100 for x in mm["rand"]]
fig, ax = fs.fig_2panel()
ax[0].plot(M, o1, "o-", color=C["ris"], label="malicious RIS")
ax[0].plot(M, r1, "^--", color=C["rand"], label="random RIS")
ax[0].axhline(mm["clean"]["top1"] * 100, ls=":", color=C["clean"], lw=1.2, label="no RIS")
ax[0].set_ylabel("Top-1 accuracy (%)")
ax[1].plot(M, ose, "o-", color=C["ris"], label="malicious RIS")
ax[1].plot(M, rse, "^--", color=C["rand"], label="random RIS")
ax[1].axhline(mm["clean"]["se"] * 100, ls=":", color=C["clean"], lw=1.2, label="no RIS")
ax[1].set_ylabel("SE ratio (%)")
for a in ax:
    a.set_xscale("log"); a.set_xticks(M); a.set_xticklabels([str(m) for m in M])
    a.set_xlabel("RIS elements $M$"); fs.despine(a)
fs.tag(ax[0], "(a)"); fs.tag(ax[1], "(b)")
ax[0].legend(loc="lower left")
fig.savefig(os.path.join(FIG, "fig_mmwave.png")); plt.close(fig)

# ---- black-box transferability ----
Ms = bx["M"]
wb = [x["top1"] * 100 for x in bx["whitebox"]]
tr = [x["top1"] * 100 for x in bx["transfer"]]
rd = [x["top1"] * 100 for x in bx["random"]]
fig, ax = fs.fig_1panel(); x = np.arange(len(Ms)); w = 0.26
ax.bar(x - w, wb, w, color=C["ris"], label="white-box")
ax.bar(x, tr, w, color=C["fgsm"], label="transfer (black-box)")
ax.bar(x + w, rd, w, color=C["rand"], label="random RIS")
ax.axhline(bx["clean"]["top1"] * 100, ls=":", color=C["clean"], lw=1.2, label="no RIS")
ax.set_xticks(x); ax.set_xticklabels([f"$M{{=}}{m}$" for m in Ms])
ax.set_ylabel("Victim top-1 accuracy (%)"); ax.set_ylim(0, 116)
ax.grid(axis="x", visible=False); fs.despine(ax); ax.legend(loc="upper center", ncol=2, fontsize=8, columnspacing=0.8)
fig.savefig(os.path.join(FIG, "fig_blackbox.png")); plt.close(fig)
print("Saved fig_mmwave.png, fig_blackbox.png")
