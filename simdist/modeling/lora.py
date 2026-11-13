"""Low-rank adapters (LoRA) on the linear layers of a module tree (design doc 20).

``apply_lora(root, rank, rngs)`` swaps every ``nnx.Linear`` / ``nnx.LinearGeneral`` under ``root`` for a
subclass that adds ``scale * (x @ lora_a) @ lora_b`` to its output. ``lora_b`` is zero-initialized, so the
wrapped module computes exactly the original function until the adapter is trained. The base kernel/bias keep
their parameter paths (the class is swapped in place), so a pretrained checkpoint restores unchanged and the
adapter parameters live at ``.../<layer>/lora_a`` and ``.../<layer>/lora_b`` (heads filter ``lora``).
"""
from __future__ import annotations

import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp


def _normalize_axes(axes, ndim: int) -> tuple[int, ...]:
    return tuple(sorted(ax if ax >= 0 else ndim + ax for ax in axes))


class LoRALinear(nnx.Linear):
    """``nnx.Linear`` plus a rank-r update on the last axis."""

    def __call__(self, inputs):
        y = super().__call__(inputs)
        delta = (inputs @ self.lora_a.value) @ self.lora_b.value
        return y + self.lora_scale * delta


class LoRALinearGeneral(nnx.LinearGeneral):
    """``nnx.LinearGeneral`` plus a rank-r update over the contracted axes (no batch axes)."""

    def __call__(self, inputs):
        y = super().__call__(inputs)
        axis = _normalize_axes(self.axis, inputs.ndim)
        n_in = len(axis)
        assert axis == tuple(range(inputs.ndim - n_in, inputs.ndim)), "LoRA expects the contracted axes to be the trailing axes"
        x_flat = inputs.reshape(inputs.shape[: inputs.ndim - n_in] + (-1,))
        delta = (x_flat @ self.lora_a.value) @ self.lora_b.value
        delta = delta.reshape(delta.shape[:-1] + tuple(self.out_features))
        return y + self.lora_scale * delta


def _attach(layer: nnx.Module, in_dim: int, out_dim: int, rank: int, scale: float, key) -> None:
    # A ~ N(0, 1/in) (LoRA's small random input projection), B = 0 so the adapted layer starts at the base layer
    layer.lora_a = nnx.Param(jax.random.normal(key, (in_dim, rank), jnp.float32) / math.sqrt(in_dim))
    layer.lora_b = nnx.Param(jnp.zeros((rank, out_dim), jnp.float32))
    layer.lora_scale = float(scale)


def apply_lora(root: nnx.Module, rank: int, rngs: nnx.Rngs, scale: float = 1.0) -> int:
    """Attach adapters to every linear layer under ``root`` (in place). Returns the number of layers wrapped."""
    count = 0

    def visit(obj):
        nonlocal count
        if isinstance(obj, (LoRALinear, LoRALinearGeneral)):
            return
        if isinstance(obj, nnx.Linear):
            _attach(obj, int(obj.in_features), int(obj.out_features), rank, scale, rngs.params())
            obj.__class__ = LoRALinear
            count += 1
            return
        if isinstance(obj, nnx.LinearGeneral):
            in_f = obj.in_features if isinstance(obj.in_features, (tuple, list)) else (obj.in_features,)
            out_f = obj.out_features if isinstance(obj.out_features, (tuple, list)) else (obj.out_features,)
            assert len(obj.batch_axis) == 0, "LoRA on LinearGeneral with batch axes is not supported"
            _attach(obj, int(math.prod(in_f)), int(math.prod(out_f)), rank, scale, rngs.params())
            obj.__class__ = LoRALinearGeneral
            count += 1
            return
        if isinstance(obj, nnx.Module):
            for name, child in list(vars(obj).items()):
                if name.startswith("_"):
                    continue
                visit(child)
        elif isinstance(obj, (list, tuple)):
            for child in obj:
                visit(child)
        elif isinstance(obj, dict):
            for child in obj.values():
                visit(child)

    visit(root)
    return count


def lora_param_count(root: nnx.Module) -> int:
    n = 0

    def visit(obj):
        nonlocal n
        if isinstance(obj, (LoRALinear, LoRALinearGeneral)):
            n += obj.lora_a.value.size + obj.lora_b.value.size
        if isinstance(obj, nnx.Module):
            for name, child in vars(obj).items():
                if not name.startswith("_"):
                    visit(child)
        elif isinstance(obj, (list, tuple)):
            for child in obj:
                visit(child)

    visit(root)
    return n
