"""Stage 3: physically-realizable MALICIOUS-RIS adversarial attack.

Threat model
------------
A passive rogue RIS with M unit-modulus elements is present. It adds a
controllable reflected path so the effective BS->user channel becomes

        h_eff(v) = h_d  +  (kappa * ||h_d|| / M_ref) * (k_u^H v) * a_bsR

where
  h_d   : ray-traced direct BS->user channel (DeepMIMO), per user
  a_bsR : BS array response toward the RIS (unit-norm)  -> rank-1 BS->RIS LoS
  k_u   : cascaded RIS response (a_in (BS->RIS) elementwise a_out (RIS->user_u)),
          |k_u[m]| = 1
  v     : RIS reflection vector, v_m = exp(j*theta_m), |v_m| = 1   (the attacker's knob)
  kappa : RIS-to-direct amplitude budget at the reference size M_ref

The perturbation lives in the RIS PHASE space and obeys the passive/unit-modulus
(and optionally b-bit) hardware constraints -- so, unlike feature-space FGSM/PGD,
it is something an over-the-air adversary can actually deploy. Attacker strength
scales with M (RIS aperture), which is the physically-meaningful budget knob.
"""
import numpy as np, torch
import beamdata as bd
from model import DEVICE

N_ANT = bd.N_BS_ANT
M_REF = 64
U_BSR = 0.30            # sin(angle) of the BS->RIS direction (fixed geometry)


def _steer(u, n):
    """ULA array response exp(j*pi*n*u), shape (len(u), n) unit-modulus entries."""
    return np.exp(1j * np.pi * np.outer(u, np.arange(n)))


def build_geometry(H_d, M, kappa, seed=bd.SEED):
    """Return torch tensors describing the RIS cascade for every user."""
    rng = np.random.default_rng(seed)
    U = H_d.shape[0]
    hnorm = np.linalg.norm(H_d, axis=1)                      # (U,)
    a_bsR = _steer(np.array([U_BSR]), N_ANT)[0] / np.sqrt(N_ANT)   # (N,) unit-norm
    a_in = _steer(np.array([U_BSR]), M)[0]                   # (M,) RIS<-BS response
    u_out = rng.uniform(-1, 1, size=U)                       # per-user RIS->user angle
    a_out = _steer(u_out, M)                                 # (U, M)
    k_u = np.conj(a_in)[None, :] * a_out                     # (U, M) cascaded response
    scale = (kappa * hnorm / M_REF).astype(np.float32)       # (U,) -> linear-in-M aperture
    W = bd.dft_codebook()
    g = dict(
        H_d=torch.as_tensor(H_d, dtype=torch.complex64, device=DEVICE),
        a_bsR=torch.as_tensor(a_bsR, dtype=torch.complex64, device=DEVICE),
        k_u=torch.as_tensor(k_u, dtype=torch.complex64, device=DEVICE),
        scale=torch.as_tensor(scale, dtype=torch.float32, device=DEVICE),
        W=torch.as_tensor(W, dtype=torch.complex64, device=DEVICE),   # codebook (B,N)
        M=M,
    )
    return g


def _feats_torch(H):
    """Per-sample unit-norm then real|imag concat -> (U,2N) float32 (matches beamdata)."""
    H = H / (torch.linalg.norm(H, dim=1, keepdim=True) + 1e-8)
    return torch.cat([H.real, H.imag], dim=1).float()


def _h_eff(g, theta):
    v = torch.exp(1j * theta)                               # (U,M) unit modulus
    c = (g["k_u"].conj() * v).sum(dim=1)                    # (U,) complex = k_u^H v
    return g["H_d"] + (g["scale"] * c)[:, None] * g["a_bsR"][None, :]


def ris_attack(model, g, iters=80, lr=0.1, bbit=None, seed=bd.SEED, return_theta=False):
    """Batched projected-gradient RIS-phase attack.

    Objective: minimise the user's *achieved / optimal* beamforming-gain ratio,
        L(theta) = E_u [ (sum_b p_b(h_eff) g_b(h_eff)) / max_b g_b(h_eff) ],
    where p_b is the victim DNN's soft beam decision and g_b the codebook gain on
    the true effective channel. This directly targets the achieved spectral
    efficiency, is scale-invariant (no blow-up when the RIS path is large), and
    gives the attacker more degrees of freedom as M grows (monotone in aperture).
    Optimises one RIS config per user; returns effective channels for honest eval.
    """
    model.eval()
    torch.manual_seed(seed)
    U, M = g["H_d"].shape[0], g["M"]
    theta = (torch.rand(U, M, device=DEVICE) * 2 * np.pi).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    Wc = g["W"].conj()                                       # (B,N)
    for _ in range(iters):
        opt.zero_grad()
        H = _h_eff(g, theta)                                # (U,N) complex
        p = torch.softmax(model(_feats_torch(H)), dim=1)    # (U,B) soft beam decision
        gains = (H @ Wc.T).abs() ** 2                       # (U,B) beamforming gains
        opt_gain = gains.max(dim=1, keepdim=True).values.detach()
        achieved = (p * gains).sum(dim=1, keepdim=True)
        loss = (achieved / (opt_gain + 1e-30)).mean()       # minimise achieved/optimal
        loss.backward(); opt.step()
    with torch.no_grad():
        if bbit is not None:                                # discrete b-bit RIS phases
            step = 2 * np.pi / (2 ** bbit)
            theta.data = torch.round(theta.data / step) * step
        heff = _h_eff(g, theta).detach().cpu().numpy()
    return (heff, theta.detach()) if return_theta else heff


def ris_attack_detector_aware(model, detector, g, lam=2.0, iters=120, lr=0.1, seed=bd.SEED):
    """Detector-aware (adaptive) RIS attack: minimise the victim's achieved/optimal
    beam-gain ratio (fool the classifier) AND the detector's malicious-score (evade
    the flag) jointly. lam trades classifier damage against detector evasion."""
    model.eval(); detector.eval(); torch.manual_seed(seed)
    U, M = g["H_d"].shape[0], g["M"]
    theta = (torch.rand(U, M, device=DEVICE) * 2 * np.pi).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    Wc = g["W"].conj()
    for _ in range(iters):
        opt.zero_grad()
        H = _h_eff(g, theta)
        feats = _feats_torch(H)
        p = torch.softmax(model(feats), dim=1)
        gains = (H @ Wc.T).abs() ** 2
        opt_gain = gains.max(dim=1, keepdim=True).values.detach()
        cls_loss = ((p * gains).sum(dim=1, keepdim=True) / (opt_gain + 1e-30)).mean()
        det_loss = torch.sigmoid(detector(feats)).mean()      # drive detector score down
        (cls_loss + lam * det_loss).backward(); opt.step()
    return _h_eff(g, theta).detach().cpu().numpy()


def ris_snr_jam(g, iters=120, lr=0.1, seed=bd.SEED):
    """DNN-BLIND destructive-beamforming RIS: minimise the best-beam receive gain
    (max_b |w_b^H h_eff|^2). This is the malicious-but-model-agnostic baseline of
    [malris]; it degrades the channel without targeting the predictor."""
    torch.manual_seed(seed)
    U, M = g["H_d"].shape[0], g["M"]
    theta = (torch.rand(U, M, device=DEVICE) * 2 * np.pi).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    Wc = g["W"].conj()
    for _ in range(iters):
        opt.zero_grad()
        H = _h_eff(g, theta)
        gains = (H @ Wc.T).abs() ** 2
        loss = gains.max(dim=1).values.mean()               # minimise best achievable gain
        loss.backward(); opt.step()
    return _h_eff(g, theta).detach().cpu().numpy()


def heff_from_theta(g, theta):
    """Effective channels (U,N) numpy from angles theta of shape (M,) or (U,M)."""
    with torch.no_grad():
        if theta.dim() == 1:
            c = (g["k_u"].conj() * torch.exp(1j * theta)[None, :]).sum(1)
        else:
            c = (g["k_u"].conj() * torch.exp(1j * theta)).sum(1)
        H = g["H_d"] + (g["scale"] * c)[:, None] * g["a_bsR"][None, :]
        return H.cpu().numpy()


def ris_attack_universal(model, g, iters=200, lr=0.05, seed=bd.SEED):
    """Optimise ONE RIS phase config shared by every user (information-limited attacker:
    knows the environment distribution and the model, but NOT any victim's per-user CSI).
    Returns the shared angle vector theta of shape (M,)."""
    model.eval(); torch.manual_seed(seed)
    theta = (torch.rand(g["M"], device=DEVICE) * 2 * np.pi).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    Wc = g["W"].conj()
    for _ in range(iters):
        opt.zero_grad()
        c = (g["k_u"].conj() * torch.exp(1j * theta)[None, :]).sum(1)   # (U,)
        H = g["H_d"] + (g["scale"] * c)[:, None] * g["a_bsR"][None, :]
        p = torch.softmax(model(_feats_torch(H)), dim=1)
        gains = (H @ Wc.T).abs() ** 2
        opt_gain = gains.max(dim=1, keepdim=True).values.detach()
        loss = ((p * gains).sum(dim=1, keepdim=True) / (opt_gain + 1e-30)).mean()
        loss.backward(); opt.step()
    return theta.detach()
