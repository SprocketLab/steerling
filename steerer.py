import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class LowRankAdapter(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        alpha: float = 1.0,
        activation: str = "silu",
        dtype=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha

        # Choose activation
        if activation is None:
            self.act = None
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        elif activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "tanh":
            self.act = nn.Tanh()
        else:
            # default: SiLU (like many modern adapters / LoRA variants)
            self.act = nn.SiLU()

        # Linear layers: (H -> r) then (r -> H)
        # Use dtype from base model if provided
        self.down = nn.Linear(hidden_size, rank, bias=False, dtype=dtype)
        self.up = nn.Linear(rank, hidden_size, bias=False, dtype=dtype)

        self.reset_parameters()

    def reset_parameters(self):
        # LoRA-style init:
        # - down: small random
        # - up: zeros so the adapter starts as a no-op
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        self.down.weight.data *= 1e-3  # make it small
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (..., H)
        returns Δh of same shape as h
        """
        orig_shape = h.shape
        H = orig_shape[-1]
        assert H == self.hidden_size, f"Expected last dim {self.hidden_size}, got {H}"

        x = h.view(-1, H)  # (N, H)
        x = self.down(x)   # (N, r)
        if self.act is not None:
            x = self.act(x)
        x = self.up(x)     # (N, H)

        if self.alpha != 1.0:
            x = self.alpha * x

        return x.view(orig_shape)


class AdapterSteerer(nn.Module):
    """
    Wraps a causal LM and injects learned low-rank adapters at:
      - block output
      - attention output
      - MLP output

    For each (layer, target) we have a LowRankAdapter:
        Δh = α * W_up( act(W_down(h)) )

    `apply_to` controls where within the sequence the adapter is applied:
      - "last": only to the last token (position -1)
      - "all":  to all tokens in the sequence
    """

    def __init__(
        self,
        base_model,
        layers_to_steer="all",
        targets=("block",),
        rank: int = 4,
        apply_to: str = "last",  # "last" or "all"
        alpha: float = 1.0,
        activation: str = "silu",
    ):
        super().__init__()
        assert hasattr(base_model, "model"), "Assuming LLaMA-style with .model.layers"
        assert apply_to in ("last", "all")

        self.base_model = base_model
        self.config = base_model.config
        self.apply_to = apply_to
        self.rank = rank
        self.alpha = alpha
        self.activation = activation

        if layers_to_steer == "all":
            try:
                self.layers_to_steer = list(range(self.config.num_hidden_layers))
            except:
                self.layers_to_steer = list(range(self.config.text_config.num_hidden_layers))
        else:
            self.layers_to_steer = list(layers_to_steer)

        # normalize targets to a set
        self.targets = targets  # e.g. {"block", "attn", "mlp"}
        try:
            hidden_size = self.config.hidden_size
        except:
            hidden_size = self.config.text_config.hidden_size
        param_dtype = next(base_model.parameters()).dtype

        # One adapter per (layer, target)
        self.adapters = nn.ModuleDict()
        for layer_idx in self.layers_to_steer:
            if "block" in self.targets:
                self.adapters[f"layer{layer_idx}_block"] = LowRankAdapter(
                    hidden_size=hidden_size,
                    rank=rank,
                    alpha=alpha,
                    activation=activation,
                    dtype=param_dtype,
                )
            if "attn" in self.targets:
                self.adapters[f"layer{layer_idx}_attn"] = LowRankAdapter(
                    hidden_size=hidden_size,
                    rank=rank,
                    alpha=alpha,
                    activation=activation,
                    dtype=param_dtype,
                )
            if "mlp" in self.targets:
                self.adapters[f"layer{layer_idx}_mlp"] = LowRankAdapter(
                    hidden_size=hidden_size,
                    rank=rank,
                    alpha=alpha,
                    activation=activation,
                    dtype=param_dtype,
                )
        # Register hooks
        self._hooks = []
        for layer_idx in self.layers_to_steer:
            try:
                layer = self.base_model.model.layers[layer_idx]
            except:
                layer = self.base_model.model.language_model.layers[layer_idx]

            # 1) Hook on whole block output
            if "block" in self.targets:
                h = layer.register_forward_hook(
                    self._make_block_hook(layer_idx)
                )
                self._hooks.append(h)

            # 2) Hook on attention output
            if "attn" in self.targets:
                attn = layer.post_attention_layernorm  # LLaMA-style
                h = attn.register_forward_hook(
                    self._make_submodule_hook(layer_idx, "attn")
                )
                self._hooks.append(h)

            # 3) Hook on MLP output
            if "mlp" in self.targets:
                mlp = layer.mlp
                h = mlp.register_forward_hook(
                    self._make_submodule_hook(layer_idx, "mlp")
                )
                self._hooks.append(h)

    # ---------- helpers: how to apply adapters over sequence ----------

    def _apply_adapter_last(self, tensor, adapter: nn.Module):
        """
        tensor: (B, S, H)
        apply adapter only to last token (position -1) in a functional way
        """
        if tensor.ndim != 3:
            return tensor

        B, S, H = tensor.shape
        h_last = tensor[:, -1, :]           # (B, H)
        delta_last = adapter(h_last)        # (B, H)
        new_last = h_last + delta_last      # (B, H)

        # build a new tensor, no in-place assignment
        new_last = new_last.unsqueeze(1)    # (B, 1, H)
        prefix = tensor[:, :-1, :]          # (B, S-1, H)
        return torch.cat([prefix, new_last], dim=1)


    def _apply_adapter_all(self, tensor, adapter: nn.Module):
        """
        tensor: (B, S, H)
        apply adapter to all tokens individually
        """
        if tensor.ndim != 3:
            return tensor

        B, S, H = tensor.shape
        h = tensor.view(B * S, H)           # (B*S, H)
        delta = adapter(h).view(B, S, H)    # (B, S, H)
        return tensor + delta               # functional, no in-place


    def _apply_adapter(self, tensor, adapter: nn.Module):
        if self.apply_to == "last":
            return self._apply_adapter_last(tensor, adapter)
        else:  # "all"
            return self._apply_adapter_all(tensor, adapter)

    # ---------- Hook creators ----------

    def _make_block_hook(self, layer_idx):
        key = f"layer{layer_idx}_block"

        def hook(module, inputs, output):
            adapter = self.adapters[key]
            if isinstance(output, torch.Tensor):
                return self._apply_adapter(output, adapter)
            if isinstance(output, tuple):
                first = self._apply_adapter(output[0], adapter)
                return (first,) + output[1:]
            return output

        return hook

    def _make_submodule_hook(self, layer_idx, which):
        key = f"layer{layer_idx}_{which}"

        def hook(module, inputs, output):
            adapter = self.adapters[key]
            if isinstance(output, torch.Tensor):
                return self._apply_adapter(output, adapter)
            if isinstance(output, tuple):
                first = self._apply_adapter(output[0], adapter)
                return (first,) + output[1:]
            return output

        return hook

    # ---------- Forward / generate ----------

    def forward(self, *args, **kwargs):
        # Trainer / your code will call this; hooks do the steering
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        # Generation also flows through the same hooks
        return self.base_model.generate(*args, **kwargs)


class FixedVectorSteerer(nn.Module):
    """
    Wraps a causal LM and injects trainable additive vectors at:
      - block output
      - attention output
      - MLP output
    based on `targets`.

    `apply_to` controls where the delta is added within the sequence:
      - "last": only to the last token (position -1)
      - "all":  to all tokens in the sequence
    """

    def __init__(
        self,
        base_model,
        layers_to_steer="all",
        targets=("block",),
        apply_to: str = "last",     # "last" or "all"
    ):
        super().__init__()
        assert hasattr(base_model, "model"), "Assuming LLaMA-style with .model.layers"
        assert apply_to in ("last", "all")
        self.base_model = base_model
        self.config = base_model.config
        self.apply_to = apply_to

        if layers_to_steer == "all":
            self.layers_to_steer = list(range(self.config.num_hidden_layers))
        else:
            self.layers_to_steer = list(layers_to_steer)

        # normalize targets to a set for membership checks
        self.targets = targets  # e.g. {"block", "attn", "mlp"}

        hidden_size = self.config.hidden_size
        param_dtype = next(base_model.parameters()).dtype

        # One delta per (layer, target)
        self.deltas = nn.ParameterDict()
        for layer_idx in self.layers_to_steer:
            if "block" in self.targets:
                self.deltas[f"layer{layer_idx}_block"] = nn.Parameter(
                    torch.zeros(hidden_size, dtype=param_dtype)
                    # or: torch.randn(hidden_size, dtype=param_dtype) * 1e-3
                )
            if "attn" in self.targets:
                self.deltas[f"layer{layer_idx}_attn"] = nn.Parameter(
                    torch.zeros(hidden_size, dtype=param_dtype)
                )
            if "mlp" in self.targets:
                self.deltas[f"layer{layer_idx}_mlp"] = nn.Parameter(
                    torch.zeros(hidden_size, dtype=param_dtype)
                )
        # Register hooks
        self._hooks = []
        for layer_idx in self.layers_to_steer:
            layer = self.base_model.model.layers[layer_idx]

            # 1) Hook on whole block output
            if "block" in self.targets:
                h = layer.register_forward_hook(
                    self._make_block_hook(layer_idx)
                )
                self._hooks.append(h)

            # 2) Hook on attention output
            if "attn" in self.targets:
                attn = layer.post_attention_layernorm  # LLaMA-style
                h = attn.register_forward_hook(
                    self._make_submodule_hook(layer_idx, "attn")
                )
                self._hooks.append(h)

            # 3) Hook on MLP output
            if "mlp" in self.targets:
                mlp = layer.mlp
                h = mlp.register_forward_hook(
                    self._make_submodule_hook(layer_idx, "mlp")
                )
                self._hooks.append(h)

    # ---------- helpers: how to apply delta over sequence ----------

    def _add_delta_to_last_token(self, tensor, delta_vec):
        """
        tensor: (B, S, H)
        delta_vec: (H,)
        returns new tensor with last token shifted
        """
        if tensor.ndim != 3:
            return tensor
        out = tensor.clone()
        out[:, -1, :] = out[:, -1, :] + delta_vec
        return out

    def _add_delta_to_all_tokens(self, tensor, delta_vec):
        """
        tensor: (B, S, H)
        delta_vec: (H,)
        returns new tensor with ALL tokens shifted
        """
        if tensor.ndim != 3:
            return tensor
        return tensor + delta_vec.view(1, 1, -1)

    def _apply_delta(self, tensor, delta_vec):
        if self.apply_to == "last":
            return self._add_delta_to_last_token(tensor, delta_vec)
        else:  # "all"
            return self._add_delta_to_all_tokens(tensor, delta_vec)

    # ---------- Hook creators ----------

    def _make_block_hook(self, layer_idx):
        key = f"layer{layer_idx}_block"

        def hook(module, inputs, output):
            delta_vec = self.deltas[key]  # (H,)
            if isinstance(output, torch.Tensor):
                return self._apply_delta(output, delta_vec)
            if isinstance(output, tuple):
                first = self._apply_delta(output[0], delta_vec)
                return (first,) + output[1:]
            return output

        return hook

    def _make_submodule_hook(self, layer_idx, which):
        key = f"layer{layer_idx}_{which}"

        def hook(module, inputs, output):
            delta_vec = self.deltas[key]  # (H,)
            if isinstance(output, torch.Tensor):
                return self._apply_delta(output, delta_vec)
            if isinstance(output, tuple):
                first = self._apply_delta(output[0], delta_vec)
                return (first,) + output[1:]
            return output

        return hook

    # ---------- Forward / generate ----------

    def forward(self, *args, **kwargs):
        # Trainer will call this; hooks do the steering
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        # Generation also flows through the same hooks
        return self.base_model.generate(*args, **kwargs)