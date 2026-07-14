"""White-box, feature-space adversarial attacks (baselines).

These attack the DNN's *input feature vector* directly. They are the standard
reference attacks from the literature, but note: perturbing the post-processed
256-D feature is NOT something an over-the-air adversary can do exactly -- that
is the motivation for the physically-realizable malicious-RIS attack (Stage 3).

CW is implemented with the correct untargeted margin objective (the earlier
codebase minimised MSE, which *improves* the model -- a sign bug).
"""
import numpy as np, torch, torch.nn as nn
from model import to_t, DEVICE

_CE = nn.CrossEntropyLoss()


def fgsm(model, X, y, eps):
    model.eval()
    x = to_t(X).clone().requires_grad_(True)
    yt = to_t(y, torch.long)
    _CE(model(x), yt).backward()
    x_adv = x + eps * x.grad.sign()
    return x_adv.detach().cpu().numpy()


def pgd(model, X, y, eps, alpha=None, iters=10):
    model.eval()
    alpha = alpha if alpha is not None else eps / 4
    x0 = to_t(X); yt = to_t(y, torch.long)
    x = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    for _ in range(iters):
        x = x.detach().requires_grad_(True)
        loss = _CE(model(x), yt)
        g, = torch.autograd.grad(loss, x)
        x = x.detach() + alpha * g.sign()
        x = x0 + torch.clamp(x - x0, -eps, eps)          # project to L-inf ball
    return x.detach().cpu().numpy()


def cw(model, X, y, c=1.0, iters=100, lr=0.01, kappa=0.0, buggy=False):
    """Untargeted CW-L2: minimise ||delta||^2 + c * margin.

    Correct untargeted margin: f = max(f_true - f_other, 0), which is positive
    while the input is still correctly classified and therefore *drives the true
    logit down* (increasing misprediction). The `buggy` sign that appears in some
    reused code, f = max(f_other - f_true, 0), is instead zero on every correctly
    classified sample, so the optimiser only ever minimises ||delta||^2 -> delta~0
    and the attack looks ineffective (accuracy stays near clean).
    """
    model.eval()
    x0 = to_t(X); yt = to_t(y, torch.long)
    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    n_classes = model(x0[:1]).shape[1]
    onehot = torch.zeros(len(yt), n_classes, device=DEVICE)
    onehot.scatter_(1, yt[:, None], 1.0)
    for _ in range(iters):
        logits = model(x0 + delta)
        true = (logits * onehot).sum(1)
        other = (logits - 1e9 * onehot).max(1).values
        margin = (other - true) if buggy else (true - other)   # buggy = wrong sign
        f = torch.clamp(margin + kappa, min=0.0)
        loss = (delta.pow(2).sum(1) + c * f).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return (x0 + delta).detach().cpu().numpy()
