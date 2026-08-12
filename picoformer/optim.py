"""Optimizer compatibility helpers for Picoformer's mixed-precision models."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from nemo_automodel.components.optim.optimizer import MuonConfig


def split_param_groups_by_dtype(
    param_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split optimizer groups so Dion foreach kernels see one dtype at a time."""
    homogeneous_groups: list[dict[str, Any]] = []
    for group in param_groups:
        params_by_dtype: dict[torch.dtype, list[torch.Tensor]] = defaultdict(list)
        for param in group["params"]:
            local_param = param.to_local() if isinstance(param, DTensor) else param
            params_by_dtype[local_param.dtype].append(param)

        for params in params_by_dtype.values():
            homogeneous_groups.append({**group, "params": params})

    return homogeneous_groups


class MixedPrecisionMuonConfig(MuonConfig):
    """AutoModel Muon config with dtype-homogeneous Dion parameter groups."""

    def _make_optimizer(self, param_groups, ctor_kwargs):
        from dion import Muon

        return Muon(split_param_groups_by_dtype(param_groups), **ctor_kwargs)
