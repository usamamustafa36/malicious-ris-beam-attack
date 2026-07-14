"""Figures for the revision: link-budget (kappa) sweep + SNR-jamming isolation."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import beamdata as bd
import figstyle as fs

FIG = os.path.join(os.path.dirname(__file__), "figures")
r = json.load(open(os.path.join(bd.ART, "review_response.json")))
s3 = json.load(open(os.path.join(bd.ART, "stage3_ris.json")))   # canonical DNN-aware headline numbers
_dnn_canon = {M: s3["opt"][s3["M_sweep"].index(M)]["top1"] for M in (64, 128)}
fs.use_style(); C = fs.COLORS

fig, ax = fs.fig_2panel()

# (a) severity vs RIS-to-direct power ratio (kappa sweep at M=128)
dB = [x["dB"] for x in r["kappa_sweep"]]
t1 = [x["top1"] * 100 for x in r["kappa_sweep"]]
se = [x["se"] * 100 for x in r["kappa_sweep"]]
ax[0].plot(dB, t1, "o-", color=C["ris"], label="Top-1 acc.")
ax[0].plot(dB, se, "s--", color=C["se"], label="SE ratio")
ax[0].axvline(0, color=C["clean"], ls=":", lw=1)
ax[0].text(-0.9, 42, "RIS $=$ direct", rotation=90, fontsize=8, color=C["clean"], va="center", ha="right")
ax[0].set_xlabel("RIS-to-direct power ratio (dB)"); ax[0].set_ylabel("%")
ax[0].set_ylim(0, 108)
fs.despine(ax[0]); fs.tag(ax[0], "(a)"); ax[0].legend(loc="lower left", fontsize=8)

# (b) DNN-aware vs model-blind SNR-jamming (isolates the model vulnerability)
Ms = [x["M"] for x in r["snr_jam"]]
dnn = [_dnn_canon[x["M"]] * 100 for x in r["snr_jam"]]   # matches Table I / text
jam = [x["jam"]["top1"] * 100 for x in r["snr_jam"]]
x = np.arange(len(Ms)); w = 0.36
ax[1].bar(x - w / 2, dnn, w, color=C["ris"], label="DNN-aware")
ax[1].bar(x + w / 2, jam, w, color=C["jam"], label="SNR-jam")
ax[1].axhline(90, ls=":", color=C["clean"], lw=1.2)
ax[1].text(x[-1] + 0.5, 91, "no attack", ha="right", va="bottom", fontsize=8, color=C["clean"])
ax[1].set_xticks(x); ax[1].set_xticklabels([f"$M{{=}}{m}$" for m in Ms])
ax[1].set_ylabel("Victim top-1 acc. (%)"); ax[1].set_ylim(0, 116)
ax[1].grid(axis="x", visible=False); fs.despine(ax[1]); fs.tag(ax[1], "(b)")
ax[1].legend(loc="upper center", ncol=2, fontsize=8, columnspacing=0.8, handlelength=1.2)

fig.savefig(os.path.join(FIG, "fig_review.png")); plt.close(fig)
print("Saved fig_review.png")
