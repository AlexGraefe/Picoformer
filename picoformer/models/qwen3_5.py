"""Compatibility fixes for NeMo AutoModel's native Qwen3.5 implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.distributed.parallelizer import (
    Qwen3_5ParallelizationStrategy,
    register_parallel_strategy,
)
from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForCausalLM


class PicoformerQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    """Qwen3.5 with corrected embedding initialization."""

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().initialize_weights(buffer_device=buffer_device, dtype=dtype)

        # NeMo 0.5 initializes the native backbone embedding with normal_(weight),
        # whose implicit std is 1.0. The outer initializer deliberately skips the
        # backbone, so correct the tied embedding/head after backbone initialization.
        init_std = float(getattr(self.config, "initializer_range", 0.02))
        nn.init.normal_(self.model.embed_tokens.weight, mean=0.0, std=init_std)
        if self.model.embed_tokens.padding_idx is not None:
            self.model.embed_tokens.weight[self.model.embed_tokens.padding_idx].zero_()


@register_parallel_strategy(name="PicoformerQwen3_5ForCausalLM")
class PicoformerQwen3_5ParallelizationStrategy(Qwen3_5ParallelizationStrategy):
    """Use Qwen3.5's mixed-dtype FSDP strategy for the project subclass."""


# Importing this project module makes NeMo's normal AutoModel resolution select
# the derived implementation. This retains NeMo's materialization, sharding, and
# checkpoint infrastructure without modifying an installed package or class.
ModelRegistry.register(
    "Qwen3_5ForCausalLM",
    PicoformerQwen3_5ForCausalLM,
    exist_ok=True,
)


# Keep the factory identical to NeMo's class so its training recipe recognizes the
# target and passes the distributed setup into from_config. Importing this module
# still performs both project-specific registrations above.
PicoformerAutoModelForCausalLM = NeMoAutoModelForCausalLM
