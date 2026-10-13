"""Multi-task shared-trunk network for COF(t) and log10 k jointly (MLP and KAN versions).

    run features --trunk--> embedding e (tanh-bounded)
        e                --wear head-->  log10 k      (one prediction per run)
        [e, time enc.]   --COF head -->  COF(t)       (one prediction per second)

Loss per step = w_cof * MSE(COF(t) over a minibatch of seconds)
              + w_wear * MSE(log10 k over the training runs),
with both targets standardised on the training fold. The wear term is computed over
*runs* (each run counted once per step), never repeated per second. ``loss_weighting``
is ``"fixed"`` (w_cof = 1, w_wear tuned) or ``"uncertainty"`` (Kendall et al. 2018:
learned log-variances s, loss = sum 0.5 exp(-s) MSE + 0.5 s). Runs with a censored or
missing wear value still train the trunk through the COF head.

Joint interface (used by :func:`ml.evaluate.cross_validate_joint`)::

    est.fit(run_X, wear_y, row_run, row_T, cof_y, groups)   # wear_y may contain NaN
    est.predict_wear(run_X), est.predict_wear_std(run_X)
    est.predict_cof(run_X, row_run, row_T), est.predict_cof_std(...)
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ._torch import Preprocessor, group_val_split, single_thread
from .kan import grid_update_hook, kan_layers, kan_regularization, make_kan
from .mlp import make_mlp
from .registry import ModelSpec


_NET = None


def __getattr__(name: str):
    # lets pickle find ml.models.multitask.MultiTaskNet without importing torch at import
    if name == "MultiTaskNet":
        return _net_class()
    raise AttributeError(name)


def _net_class():
    global _NET
    if _NET is not None:
        return _NET
    import torch
    nn = torch.nn

    class MultiTaskNet(nn.Module):
        def __init__(self, n_run, n_time, arch, trunk, head, embedding, dropout, grid_size,
                     spline_order, weighting):
            super().__init__()
            self.arch = arch
            if arch == "mlp":
                self.trunk = make_mlp(n_run, trunk, dropout, n_out=embedding)
                self.wear_head = make_mlp(embedding, head, dropout)
                self.cof_head = make_mlp(embedding + n_time, head, dropout)
            else:
                self.trunk = make_kan([n_run, *trunk, embedding], grid_size, spline_order)
                self.wear_head = make_kan([embedding, *head, 1], grid_size, spline_order)
                self.cof_head = make_kan([embedding + n_time, *head, 1], grid_size, spline_order)
            # learned log-variances (used only with uncertainty weighting)
            self.log_var = nn.Parameter(torch.zeros(2), requires_grad=weighting == "uncertainty")

        def embed(self, R):
            return torch.tanh(self.trunk(R))

        def wear(self, E):
            return self.wear_head(E).squeeze(-1)

        def cof(self, E_rows, T):
            return self.cof_head(torch.cat([E_rows, T], dim=-1)).squeeze(-1)

    MultiTaskNet.__module__, MultiTaskNet.__qualname__ = __name__, "MultiTaskNet"
    _NET = MultiTaskNet
    return MultiTaskNet


class MultiTaskRegressor(BaseEstimator):
    """Seed ensemble of shared-trunk two-head networks (``arch`` = "mlp" | "kan")."""

    def __init__(self, arch: str = "mlp", trunk=(16,), head=(16,), embedding: int = 8,
                 dropout: float = 0.1, weight_decay: float = 1e-4, lr: float = 3e-3,
                 epochs: int = 300, patience: int = 40, batch_size: int = 1024, n_seeds: int = 3,
                 val_frac: float = 0.2, loss_weighting: str = "uncertainty", w_wear: float = 1.0,
                 w_cof: float = 1.0, grid_size: int = 5, spline_order: int = 3, lamb: float = 1e-4,
                 lamb_entropy: float = 2.0, grid_updates: tuple = (0, 20, 60), random_state: int = 0):
        self.arch = arch
        self.trunk = trunk
        self.head = head
        self.embedding = embedding
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.n_seeds = n_seeds
        self.val_frac = val_frac
        self.loss_weighting = loss_weighting
        self.w_wear = w_wear
        self.w_cof = w_cof
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.lamb = lamb
        self.lamb_entropy = lamb_entropy
        self.grid_updates = grid_updates
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, run_X: pd.DataFrame, wear_y: np.ndarray, row_run: np.ndarray,
            row_T: pd.DataFrame, cof_y: np.ndarray, groups: np.ndarray | None = None):
        import torch
        if self.arch not in ("mlp", "kan"):
            raise ValueError(f"arch must be 'mlp' or 'kan', got {self.arch!r}")
        if self.arch == "mlp" and (max(self.trunk) > 31 or max(self.head) > 31):
            raise ValueError("MLP trunk/head layers must have < 32 units")
        kind = "minmax" if self.arch == "kan" else "standard"
        self.feature_names_ = list(run_X.columns)
        self.time_names_ = list(row_T.columns)
        self.pre_run_ = Preprocessor(kind).fit(run_X)
        self.pre_time_ = Preprocessor(kind).fit(row_T)
        R = self.pre_run_.transform(run_X)
        T = self.pre_time_.transform(row_T)
        wy = np.asarray(wear_y, float)
        cy = np.asarray(cof_y, float)
        has_w = np.isfinite(wy)
        self.w_mu_, self.w_sd_ = float(wy[has_w].mean()), float(wy[has_w].std() or 1.0)
        self.c_mu_, self.c_sd_ = float(cy.mean()), float(cy.std() or 1.0)
        wys = np.where(has_w, (wy - self.w_mu_) / self.w_sd_, 0.0).astype(np.float32)
        cys = ((cy - self.c_mu_) / self.c_sd_).astype(np.float32)
        row_run = np.asarray(row_run, int)
        Net = _net_class()
        self.members_, vw, vc = [], [], []
        with single_thread():
            for s in range(int(self.n_seeds)):
                seed = int(self.random_state) * 7919 + s
                rng = np.random.default_rng(seed)
                tr_runs, va_runs = group_val_split(len(R), groups, self.val_frac, rng)
                torch.manual_seed(seed)
                net = Net(R.shape[1], T.shape[1], self.arch, tuple(self.trunk), tuple(self.head),
                          int(self.embedding), float(self.dropout), int(self.grid_size),
                          int(self.spline_order), self.loss_weighting)
                bw, bc = self._train(net, R, wys, has_w, row_run, T, cys, tr_runs, va_runs, seed)
                self.members_.append(net)
                vw.append(bw)
                vc.append(bc)
        self.noise_var_wear_ = float(np.mean(vw)) * self.w_sd_ ** 2
        self.noise_var_cof_ = float(np.mean(vc)) * self.c_sd_ ** 2
        return self

    def _train(self, net, R, wys, has_w, row_run, T, cys, tr_runs, va_runs, seed):
        import torch
        gen = torch.Generator().manual_seed(seed)
        Rt, Wt, Mw = torch.from_numpy(R), torch.from_numpy(wys), torch.from_numpy(has_w)
        Tt, Ct = torch.from_numpy(T), torch.from_numpy(cys)
        rr = torch.from_numpy(row_run)
        tr_mask_run = np.zeros(len(R), bool)
        tr_mask_run[tr_runs] = True
        rows_tr = np.flatnonzero(tr_mask_run[row_run])
        rows_va = np.flatnonzero(~tr_mask_run[row_run])
        w_tr = torch.from_numpy(tr_mask_run & has_w)
        w_va = torch.from_numpy(~tr_mask_run & has_w)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n = len(rows_tr)
        bs = n if self.batch_size >= n else int(self.batch_size)
        hooks = None
        if self.arch == "kan":
            hooks = grid_update_hook(tuple(self.grid_updates))
        best, best_state, bad, best_parts = np.inf, copy.deepcopy(net.state_dict()), 0, (1.0, 1.0)
        rows_tr_t = torch.from_numpy(rows_tr)
        for ep in range(int(self.epochs)):
            if hooks is not None:
                with torch.no_grad():
                    hooks(net.trunk, ep, Rt[torch.from_numpy(tr_runs)])
                    E = net.embed(Rt)
                    hooks(net.wear_head, ep, E[w_tr])
                    hooks(net.cof_head, ep, torch.cat([E[rr[rows_tr_t]], Tt[rows_tr_t]], -1))
            net.train()
            perm = rows_tr_t[torch.randperm(n, generator=gen)] if bs < n else rows_tr_t
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                E = net.embed(Rt)
                mse_c = torch.mean((net.cof(E[rr[b]], Tt[b]) - Ct[b]) ** 2)
                mse_w = (torch.mean((net.wear(E[w_tr]) - Wt[w_tr]) ** 2) if bool(w_tr.any())
                         else torch.zeros(()))
                if self.loss_weighting == "uncertainty":
                    lv = net.log_var
                    loss = 0.5 * (torch.exp(-lv[0]) * mse_c + lv[0] + torch.exp(-lv[1]) * mse_w + lv[1])
                else:
                    loss = self.w_cof * mse_c + self.w_wear * mse_w
                if self.arch == "kan":
                    loss = loss + sum(kan_regularization(m, self.lamb, self.lamb_entropy)
                                      for m in (net.trunk, net.wear_head, net.cof_head))
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                E = net.embed(Rt)
                if len(rows_va):
                    rv = torch.from_numpy(rows_va)
                    vc = float(torch.mean((net.cof(E[rr[rv]], Tt[rv]) - Ct[rv]) ** 2))
                else:
                    vc = float(torch.mean((net.cof(E[rr[rows_tr_t]], Tt[rows_tr_t]) - Ct[rows_tr_t]) ** 2))
                mask = w_va if bool(w_va.any()) else w_tr
                vw = float(torch.mean((net.wear(E[mask]) - Wt[mask]) ** 2)) if bool(mask.any()) else 0.0
            val = vc + vw                         # equal weight on the standardised scales
            if not np.isfinite(val):
                break
            if val < best - 1e-7:
                best, best_state, bad, best_parts = val, copy.deepcopy(net.state_dict()), 0, (vw, vc)
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        return best_parts

    # ------------------------------------------------------------------ predict
    def _members(self, run_X, row_run=None, row_T=None):
        import torch
        R = torch.from_numpy(self.pre_run_.transform(run_X))
        outs_w, outs_c = [], []
        with single_thread(), torch.no_grad():
            T = torch.from_numpy(self.pre_time_.transform(row_T)) if row_T is not None else None
            rr = torch.from_numpy(np.asarray(row_run, int)) if row_run is not None else None
            for m in self.members_:
                E = m.embed(R)
                outs_w.append(m.wear(E).numpy())
                if T is not None:
                    outs_c.append(m.cof(E[rr], T).numpy())
        w = np.stack(outs_w) * self.w_sd_ + self.w_mu_
        c = np.stack(outs_c) * self.c_sd_ + self.c_mu_ if outs_c else None
        return w, c

    def predict_wear(self, run_X) -> np.ndarray:
        return self._members(run_X)[0].mean(0)

    def predict_wear_std(self, run_X) -> np.ndarray:
        w = self._members(run_X)[0]
        return np.sqrt(w.var(0) + self.noise_var_wear_)

    def predict_cof(self, run_X, row_run, row_T) -> np.ndarray:
        return self._members(run_X, row_run, row_T)[1].mean(0)

    def predict_cof_std(self, run_X, row_run, row_T) -> np.ndarray:
        c = self._members(run_X, row_run, row_T)[1]
        return np.sqrt(c.var(0) + self.noise_var_cof_)

    def predict_both(self, run_X, row_run, row_T) -> dict:
        w, c = self._members(run_X, row_run, row_T)
        return {"wear": w.mean(0), "wear_std": np.sqrt(w.var(0) + self.noise_var_wear_),
                "cof": c.mean(0), "cof_std": np.sqrt(c.var(0) + self.noise_var_cof_)}

    def trunk_kan(self, member: int = 0):
        """First trunk layer of a KAN member (for learned-function plots)."""
        return kan_layers(self.members_[member].trunk)[0] if self.arch == "kan" else None


def _build(arch):
    def build(params: dict, seed: int, ctx: dict):
        for k in ("trunk", "head", "grid_updates"):
            if k in params and isinstance(params[k], (list, tuple)):
                params[k] = tuple(params[k])
            elif k in ("trunk", "head") and k in params:
                params[k] = (int(params[k]),)
        return MultiTaskRegressor(arch=arch, random_state=seed, **params)
    return build


SPECS = [
    ModelSpec("mt_mlp", _build("mlp"), "Multi-task shared trunk (MLP): run features -> embedding "
              "-> wear head + COF(t) head", regimes=("joint",), requires=("torch",), has_std=True,
              needs_groups=True),
    ModelSpec("mt_kan", _build("kan"), "Multi-task shared trunk (KAN layers)", regimes=("joint",),
              requires=("torch",), has_std=True, needs_groups=True),
]
