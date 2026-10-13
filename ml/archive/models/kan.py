"""Kolmogorov-Arnold network (KAN): shallow [n_in, 4-8, 1], B-spline edge functions.

Native torch implementation of the efficient-KAN formulation, so the only requirement
is torch. Each edge carries ``phi(x) = w_b * silu(x) + sum_c w_c B_c(x)`` (cubic
B-splines on a grid of ``grid_size`` intervals); a node sums its incoming edges.
Inputs are min-max scaled to [-1, 1] on the training fold (the initial grid range),
and the grids of every layer are re-fitted to the range of their actual inputs
during training (``grid_updates``), keeping the represented function unchanged.

Regularisation (pykan style): ``lamb * (L1 + lamb_entropy * entropy)`` of the mean
absolute edge activations, plus AdamW weight decay; both tuned in the inner CV.
Set ``backend: efficient_kan`` to use the ``efficient_kan`` package's layers instead
(if installed); the learned-function plots work with either backend.
"""
from __future__ import annotations

import numpy as np

from ._torch import TorchEnsembleRegressor, single_thread
from .registry import ModelSpec


def _torch():
    import torch
    return torch


_KAN_LAYER = None


def __getattr__(name: str):
    # lets pickle find ml.models.kan.KANLayer without importing torch at module import
    if name == "KANLayer":
        return make_kan_layer_class()
    raise AttributeError(name)


def make_kan_layer_class():
    """Define the layer lazily so that importing this module never imports torch."""
    global _KAN_LAYER
    if _KAN_LAYER is not None:
        return _KAN_LAYER
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional

    class KANLayer(nn.Module):
        def __init__(self, in_features: int, out_features: int, grid_size: int = 5,
                     spline_order: int = 3, grid_range=(-1.0, 1.0), grid_eps: float = 1.0,
                     noise_scale: float = 0.1):
            super().__init__()
            self.in_features, self.out_features = in_features, out_features
            self.grid_size, self.spline_order, self.grid_eps = grid_size, spline_order, grid_eps
            h = (grid_range[1] - grid_range[0]) / grid_size
            knots = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32)
            grid = knots * h + grid_range[0]
            self.register_buffer("grid", grid.expand(in_features, -1).contiguous())
            self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
            self.spline_weight = nn.Parameter(
                noise_scale / grid_size * torch.randn(out_features, in_features,
                                                      grid_size + spline_order))
            self._acts = None

        def b_splines(self, x):
            """B-spline bases, (batch, in) -> (batch, in, grid_size + spline_order)."""
            g = self.grid
            x = x.unsqueeze(-1)
            b = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
            for k in range(1, self.spline_order + 1):
                left = (x - g[:, :-(k + 1)]) / (g[:, k:-1] - g[:, :-(k + 1)]) * b[..., :-1]
                right = (g[:, k + 1:] - x) / (g[:, k + 1:] - g[:, 1:-k]) * b[..., 1:]
                b = left + right
            return b

        def edge_activations(self, x):
            """phi_oi(x_i) for every edge: (batch, out, in)."""
            base = self.base_weight.unsqueeze(0) * F.silu(x).unsqueeze(1)
            spline = torch.einsum("bic,oic->boi", self.b_splines(x), self.spline_weight)
            return base + spline

        def forward(self, x):
            acts = self.edge_activations(x)
            self._acts = acts
            return acts.sum(-1)

        def regularization(self, lamb_entropy: float = 2.0):
            if self._acts is None:
                return 0.0
            l1 = self._acts.abs().mean(0)                       # (out, in)
            tot = l1.sum()
            p = l1 / (tot + 1e-12)
            return tot + lamb_entropy * (-(p * torch.log(p + 1e-12)).sum())

        @torch.no_grad()
        def update_grid(self, x, margin: float = 0.05):
            """Refit the grid to the range of ``x`` while preserving the spline function."""
            old = torch.einsum("bic,oic->bio", self.b_splines(x), self.spline_weight)   # (b,in,out)
            xs = torch.sort(x, dim=0)[0]
            G, k = self.grid_size, self.spline_order
            step = (xs[-1] - xs[0] + 2 * margin) / G
            uniform = torch.arange(G + 1, dtype=x.dtype).unsqueeze(1) * step + xs[0] - margin
            idx = torch.linspace(0, len(xs) - 1, G + 1).long()
            grid = self.grid_eps * uniform + (1 - self.grid_eps) * xs[idx]
            ext = torch.arange(1, k + 1, dtype=x.dtype).unsqueeze(1) * step
            grid = torch.cat([grid[:1] - ext.flip(0), grid, grid[-1:] + ext], dim=0)
            self.grid.copy_(grid.T.contiguous())
            A = self.b_splines(x).transpose(0, 1)                # (in, b, coef)
            sol = torch.linalg.lstsq(A, old.transpose(0, 1)).solution   # (in, coef, out)
            self.spline_weight.copy_(sol.permute(2, 0, 1))

    KANLayer.__module__, KANLayer.__qualname__ = __name__, "KANLayer"
    _KAN_LAYER = KANLayer
    return KANLayer


def make_kan(widths: list[int], grid_size: int, spline_order: int, backend: str = "native"):
    torch = _torch()
    if backend == "efficient_kan":
        from efficient_kan import KAN                       # optional dependency
        return KAN(widths, grid_size=grid_size, spline_order=spline_order)
    Layer = make_kan_layer_class()
    return torch.nn.Sequential(*[Layer(a, b, grid_size, spline_order)
                                 for a, b in zip(widths[:-1], widths[1:])])


def kan_layers(net) -> list:
    return list(net.layers) if hasattr(net, "layers") else list(net)


def edge_activations(layer, x):
    """Per-edge activations for a native or efficient_kan layer: (batch, out, in)."""
    torch = _torch()
    if hasattr(layer, "edge_activations"):
        return layer.edge_activations(x)
    base = layer.base_weight.unsqueeze(0) * layer.base_activation(x).unsqueeze(1)
    w = layer.scaled_spline_weight if hasattr(layer, "scaled_spline_weight") else layer.spline_weight
    return base + torch.einsum("bic,oic->boi", layer.b_splines(x), w)


def kan_regularization(net, lamb: float, lamb_entropy: float):
    if lamb <= 0:
        return 0.0
    if hasattr(net, "regularization_loss"):                  # efficient_kan
        return lamb * net.regularization_loss(1.0, lamb_entropy)
    return lamb * sum(layer.regularization(lamb_entropy) for layer in kan_layers(net))


def grid_update_hook(epochs_at: tuple[int, ...]):
    def hook(net, epoch, X):
        if epoch not in epochs_at:
            return
        x = X
        for layer in kan_layers(net):
            if hasattr(layer, "update_grid"):
                layer.update_grid(x)
            x = layer(x)
    return hook


def layer_functions(layer, pre, names: list[str] | None, Z_train=None, n: int = 101,
                    spline_order: int = 3) -> list[dict]:
    """Univariate edge functions of a KAN layer whose inputs were scaled by ``pre``.

    Returns one dict per (input i, node j): ``x`` in original units over the grid range,
    ``phi`` = phi_ij(x), and ``importance`` = mean |phi_ij| over ``Z_train`` (scaled inputs).
    """
    torch = _torch()
    n_in = layer.in_features
    names = names or [f"x{i}" for i in range(n_in)]
    lo = float(layer.grid[0, spline_order]) if hasattr(layer, "grid") else -1.0
    hi = float(layer.grid[0, -spline_order - 1]) if hasattr(layer, "grid") else 1.0
    z = np.linspace(max(lo, -1.1), min(hi, 1.1), n).astype(np.float32)
    imp = None
    with single_thread(), torch.no_grad():
        acts = edge_activations(layer, torch.from_numpy(np.repeat(z[:, None], n_in, 1))).numpy()
        if Z_train is not None:
            At = edge_activations(layer, torch.from_numpy(np.asarray(Z_train, np.float32))).numpy()
            imp = np.abs(At).mean(0)                                   # (out, in)
    out = []
    for i, name in enumerate(names):
        xo = pre.inverse_column(i, z)
        for j in range(acts.shape[1]):
            out.append({"input": name, "node": j, "x": xo, "phi": acts[:, j, i],
                        "importance": float(imp[j, i]) if imp is not None else np.nan})
    return out


class KANRegressor(TorchEnsembleRegressor):
    """Shallow KAN seed ensemble; see :func:`KANRegressor.edge_functions` for plots."""

    _scaling = "minmax"

    def __init__(self, hidden=6, grid_size: int = 5, spline_order: int = 3, lamb: float = 1e-4,
                 lamb_entropy: float = 2.0, weight_decay: float = 1e-5, lr: float = 1e-2,
                 epochs: int = 600, patience: int = 60, batch_size: int = 512, n_seeds: int = 3,
                 val_frac: float = 0.2, grid_updates: tuple = (0, 20, 60), backend: str = "native",
                 random_state: int = 0):
        self.hidden = hidden
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.lamb = lamb
        self.lamb_entropy = lamb_entropy
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.n_seeds = n_seeds
        self.val_frac = val_frac
        self.grid_updates = grid_updates
        self.backend = backend
        self.random_state = random_state

    def _widths(self, n_in: int) -> list[int]:
        h = list(self.hidden) if isinstance(self.hidden, (list, tuple)) else [int(self.hidden)]
        return [n_in, *h, 1]

    def _make_net(self, n_in: int):
        return make_kan(self._widths(n_in), int(self.grid_size), int(self.spline_order), self.backend)

    def _reg_fn(self):
        lamb, ent = float(self.lamb), float(self.lamb_entropy)
        return lambda net: kan_regularization(net, lamb, ent)

    def _epoch_hook(self):
        return grid_update_hook(tuple(self.grid_updates)) if self.backend == "native" else None

    def edge_functions(self, X_train=None, n: int = 101, member: int = 0) -> list[dict]:
        """First-layer learned univariate functions phi_ij of ensemble member ``member``."""
        layer = kan_layers(self.members_[member])[0]
        Zt = self.pre_.transform(X_train) if X_train is not None else None
        return layer_functions(layer, self.pre_, self.feature_names_, Zt, n, self.spline_order)

    def input_importance(self, X_train) -> dict[str, float]:
        """Sum over hidden nodes of the mean |first-layer edge activation| (member average)."""
        tot: dict[str, float] = {}
        for m in range(len(self.members_)):
            for e in self.edge_functions(X_train, n=3, member=m):
                tot[e["input"]] = tot.get(e["input"], 0.0) + e["importance"] / len(self.members_)
        return tot


def build_kan(params: dict, seed: int, ctx: dict):
    if isinstance(params.get("hidden"), list):
        params["hidden"] = tuple(params["hidden"])
    if isinstance(params.get("grid_updates"), list):
        params["grid_updates"] = tuple(params["grid_updates"])
    return KANRegressor(random_state=seed, **params)


SPECS = [ModelSpec("kan", build_kan, "KAN [n_in, 4-8, 1], cubic B-spline edges (grid, L1/entropy "
                   "and weight decay tuned); seed ensemble",
                   targets=("log10_k", "log10_V", "COF_t", "log10_Vdot_steady", "log10_dV_run",
                            "log10_tau_trans"),
                   requires=("torch",), has_std=True, needs_groups=True)]
