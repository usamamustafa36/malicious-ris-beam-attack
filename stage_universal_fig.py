"""Figure (single column, panels stacked): (a) the universal-attack negative result
and (b) the predicted-beam spread (adversarial-confusion mechanism)."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import beamdata as bd
import figstyle as fs

FIG = os.path.join(os.path.dirname(__file__), "figures")
uni = json.load(open(os.path.join(bd.ART, "universal.json")))
ext = json.load(open(os.path.join(bd.ART, "extra.json")))
fs.use_style(); C = fs.COLORS

fig, ax = fs.fig_2row(h=3.5)

# (a) only per-user CSI collapses the victim; universal / random stay near clean
Ms = uni["M"]; x = np.arange(len(Ms)); w = 0.2
series = [("peruser", "per-user CSI", C["ris"]), ("uni_wb", "univ. (model)", C["uni"]),
          ("uni_bb", "univ. (surrog.)", C["uni2"]), ("random", "random RIS", C["rand"])]
for i, (k, lab, col) in enumerate(series):
    ax[0].bar(x + (i - 1.5) * w, [rr["top1"] * 100 for rr in uni[k]], w, color=col, label=lab)
ax[0].axhline(uni["clean"]["top1"] * 100, ls=":", color=C["clean"], lw=1.2, label="clean")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"$M{{=}}{m}$" for m in Ms])
ax[0].set_ylabel("Victim top-1 acc. (%)"); ax[0].set_ylim(0, 165)
ax[0].set_yticks([0, 25, 50, 75, 100])
ax[0].grid(axis="x", visible=False); fs.despine(ax[0]); fs.tag(ax[0], "(a)")
ax[0].legend(loc="upper center", ncol=3, fontsize=8, columnspacing=0.7,
             handlelength=1.0, handletextpad=0.35)

# (b) predicted-beam distribution under M=128 attack: spread across the codebook
counts = np.array(ext["pred_hist"]["counts"], dtype=float)
frac = 100 * counts / counts.sum()
beams = np.arange(len(counts)); idx_bR = ext["pred_hist"]["a_bR_beam"]
ax[1].bar(beams, frac, width=0.9, color=C["ris"])
ax[1].bar([idx_bR], [frac[idx_bR]], width=1.8, color=C["clean"],
          label=f"BS$\\to$RIS-aligned beam ({frac[idx_bR]:.1f}%)")
ax[1].set_xlabel("Predicted beam index"); ax[1].set_ylabel("Mispredictions (%)")
ax[1].set_ylim(0, frac.max() * 1.25)
ax[1].text(0.97, 0.86, f"entropy {ext['pred_hist']['entropy_bits']:.2f}/6 bits\n"
                       f"{int((counts>0).sum())}/64 beams hit",
           ha="right", va="top", transform=ax[1].transAxes, fontsize=8)
fs.despine(ax[1]); fs.tag(ax[1], "(b)"); ax[1].legend(loc="upper left", fontsize=8)

fig.savefig(os.path.join(FIG, "fig_universal.png")); plt.close(fig)
print("Saved fig_universal.png")
