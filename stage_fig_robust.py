"""Combined double-column robustness figure (1x3): (a) discrete b-bit RIS phases,
(b) 28 GHz mmWave top-1 vs M, (c) 28 GHz mmWave SE vs M. Merges the b-bit and
mmWave figures."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import beamdata as bd
import figstyle as fs

FIG = os.path.join(os.path.dirname(__file__), "figures")
s3 = json.load(open(os.path.join(bd.ART, "stage3_ris.json")))
mm = json.load(open(os.path.join(bd.ART, "mmwave.json")))
fs.use_style(); C = fs.COLORS

fig, ax = fs.fig_wide_grid(1, 3, h=1.8)

# (a) discrete b-bit phases (M=128)
bb = s3["bbit"]; order = ["1", "2", "3", "None"]; labels = ["1-bit", "2-bit", "3-bit", "cont."]
x = np.arange(len(order)); w = 0.38
ax[0].bar(x - w / 2, [bb[k]["top1"] * 100 for k in order], w, color=C["ris"], label="Top-1")
ax[0].bar(x + w / 2, [bb[k]["se_ratio"] * 100 for k in order], w, color=C["se"], label="SE")
ax[0].axhline(s3["clean"]["top1"] * 100, ls=":", color=C["clean"], lw=1.2, label="no RIS")
ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylabel("%"); ax[0].set_ylim(0, 108)
ax[0].grid(axis="x", visible=False); fs.despine(ax[0]); fs.tag(ax[0], "(a)")
ax[0].legend(loc="upper right", ncol=1)

# (b) mmWave top-1 vs M ; (c) mmWave SE vs M
M = mm["M_sweep"]
for a, key, ck, ylab, tg in [(ax[1], "top1", "top1", "Top-1 accuracy (%)", "(b)"),
                             (ax[2], "se_ratio", "se", "SE ratio (%)", "(c)")]:
    a.plot(M, [x[key] * 100 for x in mm["opt"]], "o-", color=C["ris"], label="malicious RIS")
    a.plot(M, [x.get(key, 0) * 100 for x in mm["rand"]], "^--", color=C["rand"], label="random RIS")
    a.axhline(mm["clean"][ck] * 100, ls=":", color=C["clean"], lw=1.2, label="no RIS")
    a.set_xscale("log"); a.set_xticks(M); a.set_xticklabels([str(m) for m in M])
    a.set_xlabel("RIS elements $M$"); a.set_ylabel(ylab); fs.despine(a); fs.tag(a, tg)
ax[1].legend(loc="lower left")

fig.savefig(os.path.join(FIG, "fig_robust.png")); plt.close(fig)
print("Saved fig_robust.png")
