"""Invertible per-dimension normalization of state/action vectors from LeRobot ``stats.json``."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import torch

Array = TypeVar("Array", np.ndarray, torch.Tensor)

MODES = ("quantile", "mean_std")


class Normalizer:
    """``quantile`` maps [q01, q99] to [-1, 1] (no clipping, so it stays invertible);
    ``mean_std`` standardizes. Works on numpy arrays and torch tensors of shape [..., D]."""

    def __init__(self, params: dict[str, tuple[np.ndarray, np.ndarray]], mode: str):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        # Stored as (offset, scale): normalized = (x - offset) / scale
        self.params = {k: (np.asarray(o, np.float32), np.asarray(s, np.float32)) for k, (o, s) in params.items()}

    @classmethod
    def from_stats(
        cls,
        stats: dict,
        mode: str = "quantile",
        keys: tuple[str, ...] = ("observation.state", "action"),
        eps: float = 1e-6,
    ) -> Normalizer:
        params = {}
        for k in keys:
            s = stats[k]
            if mode == "quantile":
                lo, hi = np.asarray(s["q01"], np.float64), np.asarray(s["q99"], np.float64)
                offset, scale = (hi + lo) / 2, np.maximum((hi - lo) / 2, eps)
            elif mode == "mean_std":
                offset, scale = np.asarray(s["mean"], np.float64), np.maximum(np.asarray(s["std"], np.float64), eps)
            else:
                raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
            params[k] = (offset, scale)
        return cls(params, mode)

    def _get(self, key: str, like: Array) -> tuple[Array, Array]:
        offset, scale = self.params[key]
        if isinstance(like, torch.Tensor):
            return (
                torch.as_tensor(offset, dtype=like.dtype, device=like.device),
                torch.as_tensor(scale, dtype=like.dtype, device=like.device),
            )
        return offset, scale

    def normalize(self, x: Array, key: str) -> Array:
        offset, scale = self._get(key, x)
        return (x - offset) / scale

    def unnormalize(self, x: Array, key: str) -> Array:
        offset, scale = self._get(key, x)
        return x * scale + offset

    def state_dict(self) -> dict:
        """JSON-serializable form, to save next to a checkpoint for evaluation."""
        return {"mode": self.mode, "params": {k: [o.tolist(), s.tolist()] for k, (o, s) in self.params.items()}}

    @classmethod
    def from_state_dict(cls, d: dict) -> Normalizer:
        return cls({k: (np.asarray(o), np.asarray(s)) for k, (o, s) in d["params"].items()}, d["mode"])
