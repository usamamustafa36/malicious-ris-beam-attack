# Malicious-RIS Adversarial Attacks on DL Beam Prediction

A **physically-realizable threat model** for adversarial attacks on 6G deep-learning
beam prediction: a malicious reconfigurable intelligent surface (RIS). Everything runs
on cached ray-traced DeepMIMO channels; PyTorch (GPU recommended).

## The idea in one line
A rogue RIS's unit-modulus phase shifts **are** the adversarial perturbation on the
effective channel, so unlike feature-space FGSM/PGD it is physically realizable on RIS
hardware, and its budget is the RIS aperture `M`.

## Pipeline (run in order, from this folder)
| Stage | File | What it does |
|------|------|--------------|
| 1 | `beamdata.py`, `model.py`, `stage1_train.py` | Ray-traced CSI → DFT codebook → best-beam labels → victim DNN. Metrics: top-k + SE ratio. |
| 2 | `attacks.py`, `stage2_attacks.py` | White-box FGSM/PGD/CW baselines (CW sign bug fixed), ε-sweep. |
| 3 | `ris.py`, `stage3_ris.py` | **Malicious-RIS attack**: aperture sweep, random-RIS control, b-bit phases. |
| 4 | `stage4_defense.py` | RIS-adversarial training (adaptive eval, 3-seed CI) + detector. |
| 5 | `figstyle.py`, `stage5_figures.py`, `stage_extra_figures.py` | Publication figures (`figures/`). All figures share `figstyle.py` and are authored at true IEEE column width so on-page text is readable. `table_main.tex` is curated separately (merges rows from several stages) and is **not** auto-overwritten. |
| 6 | `paper/main.tex` | IEEE conference paper (compiles with `pdflatex main.tex`). |
| B | `stage_blackbox.py` | Black-box/transfer attack (surrogate model, no victim access). |
| A | `stage_mmwave.py` | Validation on a genuine 28 GHz mmWave scenario (`city_0_newyork_28`). |
| U | `stage_universal.py`, `stage_universal_fig.py` | Information-limited (universal / surrogate) attacker, the channel-aware boundary + its figure. |
| R | `stage_review_response.py`, `stage_review_figs.py` | Additional experiments: link-budget/κ sweep, SNR-jamming baseline, detector FPR, beam-confusion, gradient-masking check, CSI-error sensitivity. |
| CR | `stage_review_extra.py` | C&W sign-error substantiation (buggy 99.8% vs corrected 1.7% top-1) and the predicted-beam histogram behind the entropy result. |
| G | `stage_cnn.py` | Architecture-generality check: trains a 1-D CNN victim (`BeamCNN` in `model.py`) and runs the same malicious-RIS attack (clean 84.2% to 19.6% at M=128, random-RIS control 83.5%). |
| DA | `stage_detector_adaptive.py` | Detector-aware (adaptive) evasion attack (`ris.ris_attack_detector_aware`): adds the detector score to the RIS objective. Evading the detector (AUC 0.998 to 0.816) forces the attack to fail (victim top-1 back to 86.8%), so evasion and effectiveness are in tension. |
| SN | `stage_snr_sweep.py` | Pilot-SNR sweep (0/10/20 dB): the collapse is not tied to one operating point (clean 76.4/90.1/91.3% to 4.6/3.7/1.3% under attack). |
| F | `stage_fig_attacks.py` | Combined figure: feature-space baselines + black-box transfer in one two-panel plot. |

```bash
python3 stage1_train.py && python3 stage2_attacks.py && \
python3 stage3_ris.py && python3 stage4_defense.py && python3 stage5_figures.py
cd paper && pdflatex main.tex && pdflatex main.tex
```
Results cache to `artifacts/*.json`; figures to `figures/`.

## Headline results (ASU 3.5 GHz, N=64, 64-beam DFT codebook)
| Scenario | Top-1 | SE ratio |
|---|---|---|
| Clean | 90.1% | 97.5% |
| White-box PGD (ε=0.1, *unrealizable*) | 0.0% | 30.5% |
| **Malicious RIS, equal-power 0 dB (M=64, headline)** | **39.0%** | **70.2%** |
| Malicious RIS, +6 dB worst case (M=128) | 3.7% | 18.8% |
| Malicious RIS, 1-bit phases (M=128) | 30.9% | 72.3% |
| Random RIS (control) | ~90% | ~97% |
| RIS-adversarial training, adaptive M=128 attack | 58.4±0.1% | 87.8% |
| Detector AUC (clean vs RIS) | n/a | 0.998 |
| **Black-box transfer** (surrogate, M=128, no victim access) | 7.9±0.8% | n/a |
| **28 GHz mmWave** clean → malicious RIS (M=128) | 87.3% → 8.6% | 95.6% → 30.1% |
| 28 GHz mmWave, RIS-adv. training under attack | 8.6% → 45.7% | n/a |

## Honest limitations (stated in the paper)
- RIS cascade uses a geometric model on top of ray-traced *direct* CSI (no ray-traced RIS scenario available).
- **The attack is channel-aware** (needs the victim's CSI, as in the malicious-RIS literature). A *blind universal* RIS config is ineffective under i.i.d. user geometry, an honest negative result reported in the paper (`stage_universal.py`); sector-wide universal attacks under correlated geometry are future work.

Extensions: **white-box model assumption removed** (black-box transfer), **band generalization shown** (3.5 GHz and genuine 28 GHz mmWave). Positioned against published prior art (malicious-RIS destructive beamforming, adversarial attacks on RIS-assisted DL).
