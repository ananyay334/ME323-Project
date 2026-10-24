"""Shallow ANN ensemble: 3 inputs -> 3-4 tanh units -> 2 outputs (log10 k, COF_steady).

* inputs and targets standardised on the training rows; masked MSE (a missing / censored
  target contributes nothing), so rows with only one target still train the other output;
* Adam (lr 1e-2, weight decay from the grid), at most ``max_epochs`` full-batch epochs;
* early stopping with ``patience`` on a validation set of ``val_frac`` of the training
  *groups* (both checkpoints of a run, and the replicate group, stay on one side), best
  weights restored;
* ``n_members`` seeds (own initialisation and validation groups) -> prediction mean and SD.

The members are trained together as one batched network (one weight tensor per layer with a
leading member axis). Adam is element-wise and the loss is a sum over members, so this is
the same as training them one by one, only faster.

Backends: ``numpy`` (default; hand-written gradients of the same float64 network and the same
Adam update as ``torch.optim.Adam`` with L2 weight decay - no torch needed, identical results
on every machine) or ``torch`` (autograd; the original implementation).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .registry import ModelSpec


def n_weights(n_in: int = 3, hidden: int = 4, n_out: int = 2) -> int:
    """Trainable parameters of one member."""
    return n_in * hidden + hidden + hidden * n_out + n_out


def _loss_and_grads(params, Z, t, mtr, ntr):
    """Per-member masked training MSE, squared errors and the gradients of sum_m loss_m.

    Network per member m: out = tanh(Z W1_m + b1_m) W2_m + b2_m; loss_m = sum(mask * (out - t)^2) / n_m.
    """
    W1, b1, W2, b2 = params
    h = np.tanh(Z[None] @ W1 + b1)                                               # (M, n, H)
    out = h @ W2 + b2                                                            # (M, n, o)
    diff = out - t
    se = diff ** 2
    loss = (se * mtr).sum((1, 2)) / ntr
    dout = 2.0 * diff * mtr / ntr[:, None, None]
    gW2 = np.einsum("mnh,mno->mho", h, dout)
    gb2 = dout.sum(1, keepdims=True)
    da = np.einsum("mno,mho->mnh", dout, W2) * (1.0 - h ** 2)
    gW1 = np.einsum("nd,mnh->mdh", Z, da)
    gb1 = da.sum(1, keepdims=True)
    return loss, se, (gW1, gb1, gW2, gb2)


class ANNEnsemble(BaseEstimator, RegressorMixin):
    """Multi-output (n, 2) regressor; NaN in ``y`` = masked."""

    multi_output = True

    def __init__(self, hidden: int = 3, weight_decay: float = 1e-3, lr: float = 1e-2,
                 max_epochs: int = 3000, patience: int = 200, val_frac: float = 0.2,
                 n_members: int = 20, random_state: int = 0, backend: str = "numpy"):
        self.hidden = hidden
        self.weight_decay = weight_decay
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.n_members = n_members
        self.random_state = random_state
        self.backend = backend

    def fit(self, X, y, groups=None):
        X = np.asarray(X, float)
        Y = np.asarray(y, float)
        Y = Y[:, None] if Y.ndim == 1 else Y
        self.n_outputs_ = Y.shape[1]
        self.x_mean_, self.x_sd_ = X.mean(0), X.std(0)
        self.x_sd_[self.x_sd_ == 0] = 1.0
        self.y_mean_, self.y_sd_ = np.nanmean(Y, 0), np.nanstd(Y, 0)
        self.y_mean_ = np.nan_to_num(self.y_mean_)
        self.y_sd_ = np.where(np.isfinite(self.y_sd_) & (self.y_sd_ > 0), self.y_sd_, 1.0)
        Z = (X - self.x_mean_) / self.x_sd_
        T = (Y - self.y_mean_) / self.y_sd_
        M, n, H, d, o = int(self.n_members), len(Z), int(self.hidden), Z.shape[1], self.n_outputs_
        groups = np.arange(n).astype(str) if groups is None else np.asarray(groups).astype(str)
        ug = np.unique(groups)
        n_val = int(round(self.val_frac * len(ug))) if len(ug) >= 5 else 0
        # per-member train / validation masks (by group) and initial weights (torch Linear default)
        tr = np.ones((M, n), bool)
        rngs = [np.random.default_rng([int(self.random_state), m]) for m in range(M)]
        for m in range(M):
            if n_val:
                vg = rngs[m].choice(ug, n_val, replace=False)
                tr[m] = ~np.isin(groups, vg)

        def init(shape, fan_in, m_rng):
            b = 1.0 / np.sqrt(fan_in)
            return m_rng.uniform(-b, b, shape)

        W1 = np.stack([init((d, H), d, r) for r in rngs])
        b1 = np.stack([init((1, H), d, r) for r in rngs])
        W2 = np.stack([init((H, o), H, r) for r in rngs])
        b2 = np.stack([init((1, o), H, r) for r in rngs])
        if self.backend == "torch":
            self._fit_torch(Z, T, tr, (W1, b1, W2, b2))
        else:
            self._fit_numpy(Z, T, tr, (W1, b1, W2, b2))
        return self

    def _fit_numpy(self, Z, T, tr, init_w):
        """Full-batch Adam (torch semantics: L2 weight decay added to the gradient) in numpy."""
        M = tr.shape[0]
        W1, b1, W2, b2 = (a.copy() for a in init_w)
        t = np.nan_to_num(T)
        obs = np.isfinite(T).astype(float)                                           # (n, o)
        mtr = tr[:, :, None] * obs                                                   # (M, n, o)
        mva = (~tr)[:, :, None] * obs
        ntr = np.maximum(mtr.sum((1, 2)), 1.0)
        nva = np.maximum(mva.sum((1, 2)), 1.0)
        has_val = mva.sum((1, 2)) > 0
        params = [W1, b1, W2, b2]
        m1 = [np.zeros_like(p) for p in params]
        m2 = [np.zeros_like(p) for p in params]
        b1c, b2c, eps, lr, wd = 0.9, 0.999, 1e-8, float(self.lr), float(self.weight_decay)
        best = np.full(M, np.inf)
        best_w = [p.copy() for p in params]
        since = np.zeros(M, int)
        epoch = -1
        for epoch in range(int(self.max_epochs)):
            loss_tr, se, grads = _loss_and_grads(params, Z, t, mtr, ntr)
            v = np.where(has_val, (se * mva).sum((1, 2)) / nva, loss_tr)
            imp = v < best - 1e-12
            if imp.any():
                for bw, p in zip(best_w, params):
                    bw[imp] = p[imp]
                best[imp] = v[imp]
            since = np.where(imp, 0, since + 1)
            if (since > self.patience).all():
                break
            step = epoch + 1
            for i, (p, g) in enumerate(zip(params, grads)):
                g = g + wd * p
                m1[i] = b1c * m1[i] + (1 - b1c) * g
                m2[i] = b2c * m2[i] + (1 - b2c) * g * g
                denom = np.sqrt(m2[i]) / np.sqrt(1 - b2c ** step) + eps
                params[i] = p - (lr / (1 - b1c ** step)) * m1[i] / denom
        self.n_epochs_ = epoch + 1
        self.weights_ = [w.copy() for w in best_w]

    def _fit_torch(self, Z, T, tr, init_w):
        import torch
        M = tr.shape[0]
        W1, b1, W2, b2 = init_w
        torch.manual_seed(int(self.random_state))
        params = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in (W1, b1, W2, b2)]
        z = torch.tensor(Z, dtype=torch.float64)
        t = torch.tensor(np.nan_to_num(T), dtype=torch.float64)
        obs = torch.tensor(np.isfinite(T), dtype=torch.float64)                       # (n, o)
        mtr = torch.tensor(tr, dtype=torch.float64)[:, :, None] * obs                 # (M, n, o)
        mva = torch.tensor(~tr, dtype=torch.float64)[:, :, None] * obs
        ntr, nva = mtr.sum((1, 2)).clamp(min=1), mva.sum((1, 2)).clamp(min=1)
        has_val = (mva.sum((1, 2)) > 0).numpy()
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        best = np.full(M, np.inf)
        best_w = [p.detach().clone() for p in params]
        since = np.zeros(M, int)
        for epoch in range(int(self.max_epochs)):
            opt.zero_grad()
            out = torch.tanh(z @ params[0] + params[1]) @ params[2] + params[3]        # (M, n, o)
            se = (out - t) ** 2
            loss_tr = (se * mtr).sum((1, 2)) / ntr
            loss_tr.sum().backward()
            with torch.no_grad():
                v = ((se * mva).sum((1, 2)) / nva).numpy() if has_val.any() else loss_tr.detach().numpy()
                v = np.where(has_val, v, loss_tr.detach().numpy())
                imp = v < best - 1e-12
                if imp.any():
                    idx = torch.tensor(np.flatnonzero(imp))
                    for bw, p in zip(best_w, params):
                        bw[idx] = p.detach()[idx]
                    best[imp] = v[imp]
                since = np.where(imp, 0, since + 1)
            if (since > self.patience).all():
                break
            opt.step()
        self.n_epochs_ = epoch + 1
        self.weights_ = [w.numpy().copy() for w in best_w]

    def _members(self, X) -> np.ndarray:
        Z = (np.asarray(X, float) - self.x_mean_) / self.x_sd_
        W1, b1, W2, b2 = self.weights_
        out = np.tanh(Z[None] @ W1 + b1) @ W2 + b2                                       # (M, n, o)
        return out * self.y_sd_ + self.y_mean_

    def predict(self, X) -> np.ndarray:
        p = self._members(X).mean(0)
        return p[:, 0] if self.n_outputs_ == 1 else p

    def predict_std(self, X) -> np.ndarray:
        s = self._members(X).std(0, ddof=1) if self.weights_[0].shape[0] > 1 else \
            np.zeros((len(X), self.n_outputs_))
        return s[:, 0] if self.n_outputs_ == 1 else s


def build_ann(params: dict, seed: int, ctx: dict):
    return ANNEnsemble(random_state=seed, **params)


SPECS = [ModelSpec("ann", build_ann, "ANN ensemble: 3 -> {3, 4} tanh -> 2 outputs (log10 k, COF), masked "
                   "MSE, Adam, early stopping on 20 % of the training groups, 20 seeds",
                   has_std=True, multi_output=True)]
