"""Shared torch machinery for the neural models (MLP, KAN, multi-task).

* preprocessing fitted on the training fold only (median imputation + scaling);
* early stopping on a *group-aware* validation split (whole runs are held out, so
  seconds of one run never sit on both sides);
* deterministic training: explicit seeds, single CPU thread, no MPS/CUDA;
* seed ensembles: ``predict`` is the ensemble mean and ``predict_std`` the square
  root of (ensemble variance + mean validation MSE), i.e. epistemic + an aleatoric
  estimate, in the units of y.
"""
from __future__ import annotations

import contextlib
import copy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@contextlib.contextmanager
def single_thread():
    """Run torch on one CPU thread (deterministic; parallelism is across CV tasks)."""
    import torch
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n)


class Preprocessor:
    """Median imputer + StandardScaler (``"standard"``) or MinMax to [-1, 1] (``"minmax"``)."""

    def __init__(self, kind: str = "standard"):
        self.kind = kind

    def fit(self, X) -> "Preprocessor":
        X = np.asarray(X, float)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        Xi = self.imputer.transform(X)
        self.scaler = (StandardScaler() if self.kind == "standard"
                       else MinMaxScaler(feature_range=(-1, 1), clip=False)).fit(Xi)
        return self

    def transform(self, X) -> np.ndarray:
        return self.scaler.transform(self.imputer.transform(np.asarray(X, float))).astype(np.float32)

    def inverse_column(self, j: int, z: np.ndarray) -> np.ndarray:
        """Map scaled values of input ``j`` back to original units."""
        if self.kind == "standard":
            return z * self.scaler.scale_[j] + self.scaler.mean_[j]
        return (z - self.scaler.min_[j]) / self.scaler.scale_[j]


def group_val_split(n: int, groups: np.ndarray | None, val_frac: float,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Train/validation indices; whole groups go to validation when groups are given."""
    idx = np.arange(n)
    if val_frac <= 0 or n < 5:
        return idx, idx[:0]
    if groups is None:
        perm = rng.permutation(n)
        k = max(1, int(round(val_frac * n)))
        return np.sort(perm[k:]), np.sort(perm[:k])
    g = np.asarray(groups)
    uniq = np.unique(g)
    if len(uniq) < 3:
        return idx, idx[:0]
    k = max(1, int(round(val_frac * len(uniq))))
    val_groups = rng.choice(uniq, size=k, replace=False)
    va = np.isin(g, val_groups)
    return idx[~va], idx[va]


def train_net(net, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, *,
              lr: float, weight_decay: float, epochs: int, patience: int, batch_size: int,
              seed: int, reg_fn=None, epoch_hook=None) -> float:
    """AdamW + early stopping on validation MSE; restores the best state. Returns best val MSE."""
    import torch
    gen = torch.Generator().manual_seed(seed)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    Xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva.astype(np.float32))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(Xt)
    bs = n if batch_size is None or batch_size >= n else int(batch_size)
    best, best_state, bad = np.inf, copy.deepcopy(net.state_dict()), 0
    for ep in range(int(epochs)):
        if epoch_hook is not None:
            epoch_hook(net, ep, Xt)
        net.train()
        perm = torch.randperm(n, generator=gen) if bs < n else torch.arange(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = torch.mean((net(Xt[b]).squeeze(-1) - yt[b]) ** 2)
            if reg_fn is not None:
                loss = loss + reg_fn(net)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            if len(Xv):
                val = float(torch.mean((net(Xv).squeeze(-1) - yv) ** 2))
            else:
                val = float(torch.mean((net(Xt).squeeze(-1) - yt) ** 2))
        if not np.isfinite(val):
            break
        if val < best - 1e-7:
            best, best_state, bad = val, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return float(best)


class TorchEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Base class: subclasses define ``__init__`` (hyperparameters) and ``_make_net``.

    Required attributes on subclasses: ``n_seeds, val_frac, lr, weight_decay, epochs,
    patience, batch_size, random_state``. Optional hooks: ``_reg_fn``, ``_epoch_hook``,
    ``_scaling`` ("standard" | "minmax").
    """

    _scaling = "standard"

    def _make_net(self, n_in: int):
        raise NotImplementedError

    def _reg_fn(self):
        return None

    def _epoch_hook(self):
        return None

    def _validate(self) -> None:
        pass

    def fit(self, X, y, groups=None):
        import torch
        self._validate()
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.pre_ = Preprocessor(self._scaling).fit(X)
        Z = self.pre_.transform(X)
        y = np.asarray(y, float)
        self.y_mu_, self.y_sd_ = float(y.mean()), float(y.std() or 1.0)
        ys = ((y - self.y_mu_) / self.y_sd_).astype(np.float32)
        self.members_, vals = [], []
        with single_thread():
            for s in range(int(self.n_seeds)):
                seed = int(self.random_state) * 7919 + s
                rng = np.random.default_rng(seed)
                tr, va = group_val_split(len(Z), groups, self.val_frac, rng)
                torch.manual_seed(seed)
                net = self._make_net(Z.shape[1])
                vals.append(train_net(net, Z[tr], ys[tr], Z[va], ys[va], lr=self.lr,
                                      weight_decay=self.weight_decay, epochs=self.epochs,
                                      patience=self.patience, batch_size=self.batch_size, seed=seed,
                                      reg_fn=self._reg_fn(), epoch_hook=self._epoch_hook()))
                self.members_.append(net)
        self.val_mse_ = float(np.mean(vals))
        self.noise_var_ = self.val_mse_ * self.y_sd_ ** 2
        return self

    def member_predictions(self, X) -> np.ndarray:
        """(n_members, n) predictions in the units of y."""
        import torch
        Z = torch.from_numpy(self.pre_.transform(X))
        with single_thread(), torch.no_grad():
            out = np.stack([m(Z).squeeze(-1).numpy() for m in self.members_])
        return out * self.y_sd_ + self.y_mu_

    def predict(self, X) -> np.ndarray:
        return self.member_predictions(X).mean(axis=0)

    def predict_std(self, X) -> np.ndarray:
        p = self.member_predictions(X)
        return np.sqrt(p.var(axis=0) + self.noise_var_)
