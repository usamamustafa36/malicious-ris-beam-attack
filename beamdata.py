"""
Stage 1 core: correct DL beam-prediction task on ray-traced DeepMIMO CSI.

Task (community-standard, Alkhateeb/DeepMIMO-style):
  - BS with an N-element ULA, DFT beam codebook of size B.
  - Ground-truth label = index of codebook beam that maximises beamforming gain
        b*(u) = argmax_b |w_b^H h_u|^2
  - Model input = a *noisy* estimate of the channel (imperfect CSI at a pilot SNR),
    so predicting the best beam is a genuine learning problem, not an identity map.
  - Metrics = top-1/3/5 beam accuracy + effective spectral-efficiency ratio
        SE_ratio = E[ log2(1+rho*|w_pred^H h|^2) / log2(1+rho*|w_opt^H h|^2) ]

This module is imported by every later stage so the data/codebook/labels are shared.
"""
import os, json
import numpy as np

SEED = 42
EVAL_NOISE_SEED = 42   # dedicated seed for the imperfect-CSI noise draw used at EVALUATION,
                       # so every stage scores channels on one identical noise realization
                       # (see eval_noise/eval_feats). Chosen a priori; not tuned.
N_BS_ANT = 64          # BS uniform linear array
N_BEAMS = 64           # DFT codebook size (standard N x N codebook)
PILOT_SNR_DB = 10.0    # SNR of the imperfect CSI the model sees (input noise)
EVAL_SNR_DB = 10.0     # SNR used when scoring spectral efficiency
ART = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ART, exist_ok=True)


def dft_codebook(n_ant=N_BS_ANT, n_beams=N_BEAMS):
    """Oversampled DFT codebook, shape (n_beams, n_ant), unit-norm columns."""
    n = np.arange(n_ant)
    ang = np.linspace(-1, 1, n_beams, endpoint=False)      # spatial frequency
    W = np.exp(1j * np.pi * np.outer(ang, n)) / np.sqrt(n_ant)
    return W.astype(np.complex64)


def load_channels(cache=True, scenario="asu_campus_3p5", cache_name="channels_64ant.npy"):
    """Load ray-traced channels for a 64-elem BS ULA -> (U,64) complex."""
    cpath = os.path.join(ART, cache_name)
    if cache and os.path.exists(cpath):
        return np.load(cpath)
    import deepmimo as dm
    ds = dm.load(scenario)
    cp = dm.ChannelParameters()
    cp.bs_antenna["shape"] = np.array([N_BS_ANT, 1])
    cp.ofdm["subcarriers"] = 64
    cp.ofdm["selected_subcarriers"] = np.array([0])        # narrowband
    ch = np.asarray(ds.compute_channels(cp))               # (U,1,64,1) or (n_bs,U,1,64,1)
    if ch.ndim == 5:                                        # MacroDataset: merge each BS->user link
        ch = ch.reshape(ch.shape[0] * ch.shape[1], *ch.shape[2:])
    H = ch.reshape(ch.shape[0], -1)[:, :N_BS_ANT]
    H = H[np.sum(np.abs(H) ** 2, axis=1) > 0]               # drop inactive users
    H = H.astype(np.complex64)
    if cache:
        np.save(cpath, H)
    return H


def add_cn_noise(H, snr_db, rng):
    """Add complex AWGN at a per-sample SNR (models imperfect CSI estimation)."""
    p = np.mean(np.abs(H) ** 2, axis=1, keepdims=True)
    n0 = p / (10 ** (snr_db / 10.0))
    noise = np.sqrt(n0 / 2) * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
    return (H + noise).astype(np.complex64)


def complex_to_feat(H):
    """Per-sample unit-norm, then real|imag concat -> (U, 2N) float32."""
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    return np.concatenate([H.real, H.imag], axis=1).astype(np.float32)


def eval_noise(H, snr_db=PILOT_SNR_DB, seed=EVAL_NOISE_SEED):
    """Deterministic imperfect-CSI channel for EVALUATION: AWGN drawn from a fresh
    generator seeded with EVAL_NOISE_SEED, so a given (H, snr) always yields the same
    noisy realization across stages. This is what makes the reported clean/attacked
    numbers reproducible and identical between stage1, stage3, classical, black-box, etc.
    (Training features must stay stochastic; use add_cn_noise for those.)"""
    return add_cn_noise(H, snr_db, np.random.default_rng(seed))


def eval_feats(H, snr_db=PILOT_SNR_DB, seed=EVAL_NOISE_SEED):
    """Model-input features for the deterministic evaluation channel (see eval_noise)."""
    return complex_to_feat(eval_noise(H, snr_db, seed))


def best_beam_labels(H, W):
    """argmax beamforming gain over the codebook -> (U,) int labels."""
    gains = np.abs(H @ W.conj().T) ** 2                     # (U, B)
    return np.argmax(gains, axis=1).astype(np.int64), gains


def se_ratio(H, W, pred_idx, snr_db=EVAL_SNR_DB):
    """Effective spectral-efficiency ratio of predicted vs optimal beam.

    rho is scaled so the *average* optimal-beam receive SNR equals `snr_db`
    (standard link-budget normalisation); float64 avoids underflow on the
    tiny ray-traced path gains.
    """
    H = H.astype(np.complex128); W = W.astype(np.complex128)
    gains = np.abs(H @ W.conj().T) ** 2
    g_pred = gains[np.arange(len(pred_idx)), pred_idx]
    g_opt = gains.max(axis=1)
    rho = 10 ** (snr_db / 10.0) / g_opt.mean()
    se_pred = np.log2(1 + rho * g_pred)
    se_opt = np.log2(1 + rho * g_opt)
    return float(np.mean(se_pred / se_opt))


def topk_acc(logits_or_probs, y_true, ks=(1, 3, 5)):
    order = np.argsort(-logits_or_probs, axis=1)
    return {k: float(np.mean([y_true[i] in order[i, :k] for i in range(len(y_true))])) for k in ks}


def build_dataset(cache=True, scenario="asu_campus_3p5", cache_name="channels_64ant.npy"):
    """Return dict with clean test channels (for physical scoring) + train/val/test tensors."""
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(SEED)
    H = load_channels(cache=cache, scenario=scenario, cache_name=cache_name)
    W = dft_codebook()
    y, _ = best_beam_labels(H, W)

    idx = np.arange(len(H))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.1 / 0.8, random_state=SEED)

    def feats(ix):
        return complex_to_feat(add_cn_noise(H[ix], PILOT_SNR_DB, rng))

    data = dict(
        W=W, n_beams=N_BEAMS, n_ant=N_BS_ANT,
        H_test=H[te], y_test=y[te],
        H_tr=H[tr], H_va=H[va],          # raw channels (for RIS-augmented training)
        Xtr=feats(tr), ytr=y[tr],
        Xva=feats(va), yva=y[va],
        Xte=eval_feats(H[te]), yte=y[te],   # deterministic eval-noise realization (shared across stages)
    )
    return data


if __name__ == "__main__":
    d = build_dataset()
    print("Users(train/val/test):", len(d["ytr"]), len(d["yva"]), len(d["yte"]))
    print("Feature dim:", d["Xtr"].shape[1], "| n_beams:", d["n_beams"])
    print("Beams used in train labels:", len(np.unique(d["ytr"])), "/", N_BEAMS)
    # naive baseline: always predict the most frequent beam
    vals, cnt = np.unique(d["ytr"], return_counts=True)
    maj = vals[np.argmax(cnt)]
    naive = np.full_like(d["yte"], maj)
    print(f"Naive top-1 acc: {np.mean(naive==d['yte'])*100:.1f}%  "
          f"| naive SE ratio: {se_ratio(d['H_test'], d['W'], naive)*100:.1f}%")
