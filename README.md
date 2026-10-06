# Malicious RIS as Physical Adversaries of Learned Beam Management

Code, trained models and results for the paper *"Malicious Reconfigurable Intelligent
Surfaces as Physical Adversaries of Learned Beam Management"* (submitted to IEEE
Transactions on Cognitive Communications and Networking).

A malicious reconfigurable intelligent surface (RIS), which transmits nothing, chooses its
phases so that a learned beam predictor selects a wrong narrow beam while the channel's
capacity is preserved. The perturbation is restricted to what unit-modulus (and few-bit)
reflection can produce. All results are simulations on ray-traced channels; no
over-the-air experiment is claimed.

## Setup
Python 3.10+, PyTorch (GPU recommended; `BEAM_DEVICE=cpu` forces CPU), NumPy, SciPy,
scikit-learn, matplotlib, and `DeepMIMO` (v4) for the ray-traced channels
(`asu_campus_3p5`, `city_0_newyork_28`). Results are cached to `artifacts/*.json`,
figures to `figures/`.

## Journal experiments (run from this folder)
| Paper item | Script | What it does |
|---|---|---|
| Table IV | `stage_ci.py` | Headline results, partial-measurement and full-CSI victims, 5 seeds x 5 noise draws, 95% CIs |
| Fig. 2, Table V | `stage_tradeoff.py`, `stage_partial_L.py` | Overhead vs. SE trade-off and aperture sweep; number of wide beams L in {8, 16, 32} |
| Fig. 3 | `stage_baselines_phy.py` | Model-blind RIS baselines (SNR jamming, beam-null, beam-hijack) and capacity ratio |
| Fig. 4 | `stage_amortized.py`, `stage_latency_gen.py` | Amortized generator attack (white-/black-box), latency, stale attacker CSI |
| Table VI | `stage_rsrp_generality.py` | Phase resolution, pilot SNR, 28 GHz, RIS-adversarial training |
| Fig. 5 | `stage_multipath.py` | Rank-(P+1) multipath BS-RIS channels |
| Table VII | `rt_trace_ris.py`, `stage_raytraced.py` | Fully ray-traced RIS (Sionna RT, Munich, with diffraction) at fixed and physical power ratios |
| Table VIII | `stage_detector_rsrp.py` | Detector-aware attack on the partial-measurement victim, incl. one arms-race round |
| Table III | `stage_linkbudget.py` | Link budgets behind the RIS-to-direct power ratios |
| Figures/tables | `stage_fig_tccn.py` | Regenerates the journal figures and Table IV |

Shared code: `beamdata.py` (channels, codebook, metrics), `model.py` (victims),
`ris.py` (RIS channel models and attacks), `revision_common.py` (victims for both
settings, generic attack, noise-averaged evaluation, CIs).

`rt_trace_ris.py` runs in a separate Python 3.11 environment with `sionna-rt`; the
published channels used sionna-rt 2.x with diffraction on a GPU with a recent driver
(see its docstring). Its output `artifacts/rt_ris_channels.npz` (~430 MB) is not
committed; rerun the script to regenerate it before `stage_raytraced.py`.

## Headline results (ASU campus, 3.5 GHz, N=64, 64-beam DFT codebook)
Partial-measurement victim: L=16 wide-beam RSRPs, top-3 refinement (19 measurements).

| Condition | Top-1 | SE ratio after top-3 |
|---|---|---|
| No RIS | 52.0 ± 0.4% | 93.1 ± 0.2% |
| Random RIS (M=128) | 50.8 ± 0.3% | 92.7 ± 0.1% |
| Malicious RIS, 0 dB (M=64) | 26.9 ± 0.3% | 76.1 ± 1.0% |
| Malicious RIS, +6 dB (M=128) | 19.0 ± 0.7% | 67.0 ± 1.0% |
| Model-free local sweep (20 meas.), no RIS / 0 dB / +6 dB | n/a | 87.5 / 81.1 / 74.4% |

## Limitations (as stated in the paper)
- The main results use a geometric RIS cascade on ray-traced direct channels; the
  ray-traced RIS study covers one scene and one RIS position.
- With physical path loss at sub-6 GHz, the attack is local to users near a meter-scale
  surface.
- The attacker needs fresh per-user CSI (semi-active RIS or compromised controller), and
  a user-specific configuration targets one user or group per SSB burst.

## Conference-version scripts
`stage1_train.py` ... `stage5_figures.py`, `attacks.py`, `stage_blackbox.py`,
`stage_mmwave.py`, `stage_universal*.py`, `stage_review_*.py`, `stage_cnn.py`,
`stage_detector_adaptive.py`, `stage_snr_sweep.py` and related figure scripts reproduce
the earlier workshop submission (full-CSI victim). The journal results above supersede
them; note that the SNR-jamming baseline in that version was affected by an optimizer
scaling error fixed in `ris.ris_snr_jam`.
