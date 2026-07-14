"""Combined double-column characterization figure (2x2):
(a) top-1 vs aperture M, (b) SE ratio vs M, (c) severity vs RIS-to-direct power
ratio, (d) DNN-aware vs model-blind SNR-jamming. Merges the old aperture-sweep and
link-budget/isolation figures into one readable double-column panel."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import beamdata as bd
import figstyle as fs

FIG = os.path.join(os.path.dirname(__file__), "figures")
s3 = json.load(open(os.path.join(bd.ART, "stage3_ris.json")))
r = json.load(open(os.path.join(bd.ART, "review_response.json")))
fs.use_style(); C = fs.COLORS
_dnn = {M: s3["opt"][s3["M_sweep"].index(M)]["top1"] for M in (64, 128)}

fig, ax = fs.fig_wide_grid(2, 2, h=2.95)

# (a) top-1 vs M ; (b) SE vs M
M = s3["M_sweep"]
for a, key, ylab in [(ax[0, 0], "top1", "Top-1 accuracy (%)"), (ax[0, 1], "se_ratio", "SE ratio (%)")]:
    a.plot(M, [x[key] * 100 for x in s3["opt"]], "o-", color=C["ris"], label="malicious RIS")
    a.plot(M, [x[key] * 100 for x in s3["rand"]], "^--", color=C["rand"], label="random RIS")
    a.axhline(s3["clean"][key] * 100, ls=":", color=C["clean"], lw=1.2, label="no RIS")
    a.set_xscale("log"); a.set_xticks(M); a.set_xticklabels([str(m) for m in M])
    a.set_xlabel("RIS elements $M$"); a.set_ylabel(ylab); fs.despine(a)
fs.tag(ax[0, 0], "(a)"); fs.tag(ax[0, 1], "(b)"); ax[0, 0].legend(loc="lower left")

# (c) severity vs RIS-to-direct power ratio (kappa sweep at M=128)
dB = [x["dB"] for x in r["kappa_sweep"]]
ax[1, 0].plot(dB, [x["top1"] * 100 for x in r["kappa_sweep"]], "o-", color=C["ris"], label="Top-1 acc.")
ax[1, 0].plot(dB, [x["se"] * 100 for x in r["kappa_sweep"]], "s--", color=C["se"], label="SE ratio")
ax[1, 0].axvline(0, color=C["clean"], ls=":", lw=1)
ax[1, 0].text(0.5, 60, "RIS $=$ direct", rotation=90, fontsize=8, color=C["clean"], va="center")
ax[1, 0].set_xlabel("RIS-to-direct power ratio (dB)"); ax[1, 0].set_ylabel("%"); ax[1, 0].set_ylim(0, 108)
fs.despine(ax[1, 0]); fs.tag(ax[1, 0], "(c)"); ax[1, 0].legend(loc="lower left")

# (d) DNN-aware vs model-blind SNR-jamming
Ms = [x["M"] for x in r["snr_jam"]]; x = np.arange(len(Ms)); w = 0.36
ax[1, 1].bar(x - w / 2, [_dnn[m] * 100 for m in Ms], w, color=C["ris"], label="DNN-aware")
ax[1, 1].bar(x + w / 2, [v["jam"]["top1"] * 100 for v in r["snr_jam"]], w, color=C["jam"], label="SNR-jam")
ax[1, 1].axhline(90, ls=":", color=C["clean"], lw=1.2)
ax[1, 1].text(len(Ms) - 1 + 0.45, 91, "no attack", ha="right", va="bottom", fontsize=8, color=C["clean"])
ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels([f"$M{{=}}{m}$" for m in Ms])
ax[1, 1].set_ylabel("Victim top-1 (%)"); ax[1, 1].set_ylim(0, 108)
ax[1, 1].grid(axis="x", visible=False); fs.despine(ax[1, 1]); fs.tag(ax[1, 1], "(d)")
ax[1, 1].legend(loc="upper center", ncol=2, columnspacing=1.0)

fig.savefig(os.path.join(FIG, "fig_char.png")); plt.close(fig)
print("Saved fig_char.png")
