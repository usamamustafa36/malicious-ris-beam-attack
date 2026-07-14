"""Victim beam-prediction DNN (PyTorch) + shared train/eval helpers."""
import numpy as np, torch, torch.nn as nn
import beamdata as bd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BeamMLP(nn.Module):
    def __init__(self, in_dim, n_beams, hidden=(256, 256, 128), drop=0.2):
        super().__init__()
        layers, d = [], in_dim
        for i, h in enumerate(hidden):
            layers += [nn.Linear(d, h), nn.ReLU()]
            if i < len(hidden) - 1:                 # dropout on all but the last hidden layer
                layers += [nn.Dropout(drop)]
            d = h
        layers += [nn.Linear(d, n_beams)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BeamCNN(nn.Module):
    """1-D CNN victim: treats the [Re|Im] feature as a 2-channel signal over the
    N antennas and convolves along the array. A genuinely different inductive bias
    from the MLP, used to test that the malicious-RIS collapse is architecture-agnostic."""
    def __init__(self, in_dim, n_beams, drop=0.2):
        super().__init__()
        self.n_ant = in_dim // 2
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16, 128), nn.ReLU(),
                                  nn.Dropout(drop), nn.Linear(128, n_beams))

    def forward(self, x):
        x = x.view(x.shape[0], 2, self.n_ant)      # (U, 2ch=Re|Im, N antennas)
        return self.head(self.conv(x))


def to_t(a, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=DEVICE)


def train_model(Xtr, ytr, Xva, yva, n_beams, epochs=60, bs=256, lr=1e-3, patience=6,
                verbose=True, hidden=(256, 256, 128), seed=None, model=None):
    seed = bd.SEED if seed is None else seed
    torch.manual_seed(seed); np.random.seed(seed)
    m = (BeamMLP(Xtr.shape[1], n_beams, hidden=hidden) if model is None else model).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xtr_t, ytr_t = to_t(Xtr), to_t(ytr, torch.long)
    Xva_t, yva_t = to_t(Xva), to_t(yva, torch.long)
    n = len(Xtr_t); best_acc, best_state, bad = -1, None, 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(m(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            va_acc = (m(Xva_t).argmax(1) == yva_t).float().mean().item()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:2d}  val_acc={va_acc*100:5.1f}%")
        if va_acc > best_acc:
            best_acc, best_state, bad = va_acc, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m


@torch.no_grad()
def predict_probs(m, X):
    m.eval()
    return torch.softmax(m(to_t(X)), dim=1).cpu().numpy()
