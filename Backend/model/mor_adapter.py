# Backend/model/mor_adapter.py
"""
Robust, optimized Mixture-of-Recursions (MoR) adapter
- Stable numerical guards
- Softplus-normalized residual scaling
- Vectorized expert-choice (uses scatter_reduce where possible)
- Safe FP16/AMP-friendly operations
- KV cache offload support
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Try importing user config values; fall back to sensible defaults
try:
    from Backend.model.config import (
        EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_HIDDEN_DIM,
        MAX_LEN, DROPOUT, RECURSION_PENALTY_WEIGHT, MAX_RECURSION_STEPS,
        ENTROPY_REGULARIZATION, PAD_TOKEN_ID, USE_TRITON_OPTIMIZATIONS
    )
except Exception:
    logger.warning("Config not found, using defaults")
    EMBED_DIM = 192
    NUM_HEADS = 4
    NUM_LAYERS = 3
    FF_HIDDEN_DIM = 768
    MAX_LEN = 512
    DROPOUT = 0.1
    RECURSION_PENALTY_WEIGHT = 0.005
    MAX_RECURSION_STEPS = 3
    ENTROPY_REGULARIZATION = 0.01
    PAD_TOKEN_ID = 0

KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


def move_kv_cache(present_kv_cache: KVCache, device: str = "cuda", to_dtype: Optional[torch.dtype] = None):
    """
    Move kv cache tuples to device and optionally cast dtype.
    Works in-place returning a new list; preserves shape/order.
    """
    if present_kv_cache is None:
        return None
    out = []
    for k, v in present_kv_cache:
        k2 = k.detach().to(device)
        v2 = v.detach().to(device)
        if to_dtype is not None:
            k2 = k2.to(to_dtype)
            v2 = v2.to(to_dtype)
        out.append((k2, v2))
    return out


@dataclass
class MoRConfig:
    vocab_size: int
    embed_dim: int = EMBED_DIM
    num_heads: int = NUM_HEADS
    num_layers: int = NUM_LAYERS
    ff_hidden_dim: int = FF_HIDDEN_DIM
    max_len: int = MAX_LEN
    dropout: float = DROPOUT
    recursion_penalty_weight: float = RECURSION_PENALTY_WEIGHT
    max_recursion_steps: int = MAX_RECURSION_STEPS
    use_gradient_checkpointing: bool = False
    router_temperature: float = 1.0
    router_noise_std: float = 0.05
    entropy_regularization_weight: float = ENTROPY_REGULARIZATION
    residual_connection_strength: float = 1.0
    use_kv_cache: bool = True
    use_expert_choice: bool = False
    expert_capacity_factor: float = 1.2
    router_detach_inputs: bool = True
    kv_cache_device: str = "cuda"   # "cuda" or "cpu" or "auto"

    def __post_init__(self):
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.max_recursion_steps < 1:
            raise ValueError("max_recursion_steps must be >= 1")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")


class MoRBlock(nn.Module):
    """Single transformer block optimized for numerical stability and small GPUs."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, use_flash_attention: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_flash_attention = use_flash_attention and hasattr(F, "scaled_dot_product_attention")

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.ff[0], self.ff[3]]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x: (B, S, D)
        attn_mask: additive mask (S,S) with -inf on disallowed
        key_padding_mask: boolean (B, S) True where token is pad
        layer_past: optional (k, v) pair to prepend
        returns (x_out, present_kv) where present_kv is (k, v) for this layer
        """
        try:
            bsz, seqlen, _ = x.shape
            normed = self.norm1(x)

            q = self.q_proj(normed)
            k = self.k_proj(normed)
            v = self.v_proj(normed)

            # shape -> (B, H, S, head)
            q = q.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

            if layer_past is not None:
                past_k, past_v = layer_past
                # concat on seq dim (assumed shape (B, H, S_past, head_dim))
                k = torch.cat((past_k, k), dim=-2)
                v = torch.cat((past_v, v), dim=-2)

            present = (k, v)

            if self.use_flash_attention:
                # Ensure causal attention for autoregressive generation
                flash_attn_mask = None
                if key_padding_mask is not None:
                    # flash expects True for allowed positions -> invert pad mask
                    flash_attn_mask = ~key_padding_mask
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=flash_attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=True
                )
            else:
                # stable scaled dot-product attention
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(max(1, self.head_dim))
                if attn_mask is not None:
                    scores = scores + attn_mask
                if key_padding_mask is not None:
                    scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), torch.finfo(scores.dtype).min)
                # clamp to safe range (helps FP16)
                scores = torch.clamp(scores, min=-100.0, max=100.0)
                # subtract max for numerical stability
                scores = scores - scores.amax(dim=-1, keepdim=True)
                attn_weights = torch.softmax(scores, dim=-1)
                attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
                attn_weights = self.dropout(attn_weights)
                attn_output = torch.matmul(attn_weights, v)

            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, self.embed_dim)
            attn_out = self.out_proj(attn_output)
            x = x + self.dropout(attn_out)

            normed2 = self.norm2(x)
            ff_out = self.ff(normed2)
            x = x + ff_out

            return x, present

        except Exception as e:
            logger.error(f"MoRBlock forward error: {e}")
            raise


class AdvancedTokenRouter(nn.Module):
    """
    Router that can operate in token-choice (soft routing per-token per-step)
    or expert-choice (assign tokens to experts) modes.
    Numerically stable softmax, Gumbel-softmax for training, deterministic argmax for inference.
    """
    def __init__(self, embed_dim: int, max_steps: int, temperature: float = 1.0,
                 noise_std: float = 0.05, use_expert_choice: bool = False, expert_capacity_factor: float = 1.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_steps = max_steps
        self.temperature = max(1e-6, float(temperature))
        self.noise_std = float(noise_std)
        self.use_expert_choice = use_expert_choice
        self.expert_capacity_factor = float(expert_capacity_factor)

        self.router_net = nn.Linear(self.embed_dim, self.max_steps)
        nn.init.xavier_uniform_(self.router_net.weight)
        if self.router_net.bias is not None:
            nn.init.zeros_(self.router_net.bias)

        self.register_buffer("step_counts", torch.zeros(self.max_steps, dtype=torch.long), persistent=False)
        self.register_buffer("total_tokens", torch.tensor(0, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor, detach_inputs: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: (B, S, D)
        bsz, seqlen, _ = x.shape
        num_tokens = bsz * seqlen
        x_flat = x.view(num_tokens, -1)

        logits = self.router_net(x_flat.detach() if detach_inputs else x_flat)

        if self.use_expert_choice:
            # expert mode expects logits shape (num_tokens, num_experts)
            return self._expert_choice_routing(logits, num_tokens)
        else:
            # token-choice: (B, S, steps)
            logits_ts = logits.view(bsz, seqlen, -1)
            return self._token_choice_routing(logits_ts)

    def _token_choice_routing(self, logits: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # logits: (B, S, steps)
        device = logits.device
        training = self.training

        # stable: subtract max and optionally add noise
        logits = logits - logits.amax(dim=-1, keepdim=True)
        if training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * (self.noise_std)

        logits = torch.clamp(logits, min=-50.0, max=50.0)
        soft_probs = torch.softmax(logits / (self.temperature + 1e-8), dim=-1)
        soft_probs = torch.nan_to_num(soft_probs, nan=1e-8, posinf=1e-8, neginf=1e-8)

        # entropy metric (clamped)
        entropy = - (soft_probs * torch.log(soft_probs + 1e-8)).sum(dim=-1).mean()
        entropy = torch.clamp(entropy, min=0.0, max=20.0)

        if training:
            # differentiable-ish routing via gumbel-softmax (soft)
            gumbel = self._sample_gumbel(soft_probs.shape).to(device) * 0.5
            gumbel_logits = logits + gumbel
            routing_weights = torch.softmax(gumbel_logits / (self.temperature + 1e-8), dim=-1)
        else:
            max_idx = soft_probs.argmax(dim=-1)
            routing_weights = F.one_hot(max_idx, num_classes=self.max_steps).float()

        step_range = torch.arange(self.max_steps, device=device, dtype=logits.dtype)
        expected_steps = (soft_probs * step_range).sum(dim=-1).mean()

        metrics = {
            "entropy_loss": entropy.detach(),
            "expected_steps": expected_steps.detach(),
            "load_balance_loss": torch.tensor(0.0, device=device),
            "routing_distribution": soft_probs.detach()
        }

        # update non-differentiable counters safely
        with torch.no_grad():
            counts = routing_weights.sum(dim=(0, 1))  # per-step token counts
            if counts.numel() == self.step_counts.numel():
                self.step_counts += counts.to(self.step_counts.dtype)
                self.total_tokens += torch.tensor(routing_weights.shape[0] * routing_weights.shape[1], dtype=self.total_tokens.dtype, device=self.total_tokens.device)

        return routing_weights, metrics

    def _expert_choice_routing(self, logits: torch.Tensor, num_tokens: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        logits: (num_tokens, num_experts)
        returns ((assigned_tokens, assigned_experts), metrics)
        Vectorized attempt using scatter_reduce / scatter_max; fallback safe on-device grouping
        """
        device = logits.device
        num_experts = logits.size(-1)
        # top-k per expert candidate set size; capacity heuristic
        k = max(1, int((num_tokens / max(1, num_experts)) * self.expert_capacity_factor))
        k = min(k, max(1, num_tokens))

        # transpose for easy expert-first topk: (num_experts, num_tokens)
        expert_logits = logits.transpose(0, 1)
        topk_vals, topk_idx = torch.topk(expert_logits, k=min(k, expert_logits.size(-1)), dim=1)  # (E, k)

        candidate_token = topk_idx.reshape(-1)          # (E*k,)
        candidate_expert = torch.arange(num_experts, device=device).unsqueeze(1).expand(-1, topk_idx.size(1)).reshape(-1)
        candidate_logit = topk_vals.reshape(-1)

        # Attempt vectorized best-per-token selection using scatter_reduce (amax)
        # target: best_logit_for_token[token] = max(candidate_logit where candidate_token==token)
        try:
            best_val = torch.full((num_tokens,), float("-inf"), device=device)
            if hasattr(best_val, "scatter_reduce_"):
                best_val.scatter_reduce_(0, candidate_token, candidate_logit, reduce="amax", include_self=True)
            else:
                # fallback: use scatter_max if available
                best_val = best_val.scatter(0, candidate_token, candidate_logit)
            # select candidates that match best_val
            eq_mask = candidate_logit == best_val[candidate_token]
            if eq_mask.any():
                sel_idx = torch.nonzero(eq_mask, as_tuple=True)[0]
                sel_tokens = candidate_token[sel_idx]
                sel_experts = candidate_expert[sel_idx]
                # deduplicate tokens keeping first occurrence (stable)
                order = torch.argsort(sel_tokens)
                sel_tokens_sorted = sel_tokens[order]
                sel_experts_sorted = sel_experts[order]
                # unique consecutive
                if sel_tokens_sorted.numel() > 0:
                    keep_mask = torch.cat([torch.tensor([True], device=device), sel_tokens_sorted[1:] != sel_tokens_sorted[:-1]])
                    final_tokens = sel_tokens_sorted[keep_mask]
                    final_experts = sel_experts_sorted[keep_mask]
                else:
                    final_tokens = sel_tokens_sorted
                    final_experts = sel_experts_sorted
                assigned_tokens = final_tokens
                assigned_experts = final_experts
            else:
                assigned_tokens = torch.arange(num_tokens, device=device)
                assigned_experts = (assigned_tokens % num_experts).long()
        except Exception:
            # safe on-device fallback grouping (no CPU copies)
            token_best = {}
            # iterate in chunks to avoid memory blow; operate on CPU-less lists
            cand_t = candidate_token
            cand_e = candidate_expert
            cand_v = candidate_logit
            for ct, ce, cv in zip(cand_t.tolist(), cand_e.tolist(), cand_v.tolist()):
                prev = token_best.get(ct)
                if prev is None or cv > prev[0]:
                    token_best[ct] = (cv, ce)
            if len(token_best) == 0:
                assigned_tokens = torch.arange(num_tokens, device=device)
                assigned_experts = (assigned_tokens % num_experts).long()
            else:
                items = sorted(token_best.items(), key=lambda x: x[0])
                assigned_tokens = torch.tensor([k for k, v in items], dtype=torch.long, device=device)
                assigned_experts = torch.tensor([v[1] for k, v in items], dtype=torch.long, device=device)

        # compute load balancing
        expert_counts = torch.bincount(assigned_experts, minlength=num_experts).float()
        mean_load = expert_counts.mean() if expert_counts.numel() > 0 else torch.tensor(0.0, device=device)
        load_balance_loss = ((expert_counts - mean_load) ** 2).mean() / (mean_load + 1e-9)

        metrics = {
            "entropy_loss": torch.tensor(0.0, device=device),
            "expected_steps": float(expert_counts.mean().item()) if expert_counts.numel() > 0 else 0.0,
            "load_balance_loss": load_balance_loss.detach(),
            "routing_indices": torch.stack([assigned_tokens, assigned_experts], dim=1) if assigned_tokens.numel() > 0 else torch.empty((0, 2), device=device)
        }
        # update counters (non-diff)
        with torch.no_grad():
            if hasattr(self, "step_counts"):
                try:
                    self.step_counts += expert_counts.to(self.step_counts.dtype)
                    self.total_tokens += torch.tensor(num_tokens, dtype=self.total_tokens.dtype, device=self.total_tokens.device)
                except Exception:
                    pass

        return (assigned_tokens, assigned_experts), metrics

    def _sample_gumbel(self, shape: torch.Size) -> torch.Tensor:
        u = torch.rand(shape, device=self.router_net.weight.device)
        return -torch.log(-torch.log(u + 1e-8) + 1e-8)


class RecursiveStack(nn.Module):
    """
    The recursive stack: executes up to max_steps, using token-sparse processing to only compute
    on selected tokens per step. Returns accumulated residuals + metrics + kv cache.
    """
    def __init__(self, depth: int, embed_dim: int, num_heads: int, ff_dim: int, dropout: float, max_steps: int, config: MoRConfig):
        super().__init__()
        self.depth = depth
        self.max_steps = max_steps
        self.config = config

        self.layers = nn.ModuleList([MoRBlock(embed_dim, num_heads, ff_dim, dropout, use_flash_attention=USE_TRITON_OPTIMIZATIONS) for _ in range(depth)])
        self.router = AdvancedTokenRouter(embed_dim, max_steps, temperature=config.router_temperature, noise_std=config.router_noise_std,
                                          use_expert_choice=config.use_expert_choice, expert_capacity_factor=config.expert_capacity_factor)

        # Use softplus-bounded residual scales (positive and smooth)
        self._raw_scale = nn.Parameter(torch.zeros(max_steps))
        self.step_norms = nn.ModuleList([nn.LayerNorm(embed_dim, eps=1e-5) for _ in range(max_steps)])

    def _get_scales(self):
        sp = F.softplus(self._raw_scale) + 1e-6
        # normalize mean to 1.0 to keep accumulated magnitude stable
        sp_mean = sp.mean().clamp(min=1e-6)
        return sp / sp_mean

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None, past_kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], KVCache]:
        """
        x: (B, S, D)
        returns: (output, metrics, present_kv_cache)
        """
        try:
            device = x.device
            bsz, seqlen, d = x.shape

            routing_result, router_metrics = self.router(x, detach_inputs=self.config.router_detach_inputs)
            accum = torch.zeros_like(x)

            final_present_kv = None
            entropy_loss = router_metrics.get("entropy_loss", torch.tensor(0.0, device=device))
            load_balance_loss = router_metrics.get("load_balance_loss", torch.tensor(0.0, device=device))

            scales = self._get_scales()

            # Token-choice path (dense routing weights)
            if not self.router.use_expert_choice:
                routing_weights = routing_result  # (B, S, steps)
                # quick early guard: if nothing selected, return x unchanged (prevents zero-collapse)
                if routing_weights.sum() == 0:
                    router_metrics["routing_penalty"] = torch.tensor(0.0, device=device)
                    router_metrics["entropy_loss"] = entropy_loss
                    router_metrics["load_balance_loss"] = load_balance_loss
                    return x, router_metrics, None

                # iterate steps and process compact mini-batches per step
                for step in range(self.max_steps):
                    # find positions per batch for this step
                    # mask shape (B, S)
                    step_mask = routing_weights[..., step].bool()
                    # vectorized extraction of indices
                    batch_idxs, seq_idxs = torch.nonzero(step_mask, as_tuple=True)

                    if batch_idxs.numel() == 0:
                        continue

                    # group tokens by batch into contiguous padded mini-batch
                    # compute counts per batch
                    counts = torch.bincount(batch_idxs, minlength=bsz).tolist()
                    max_count = max(counts)
                    # create tensor of shape (B, max_count, D) filled with 0 and mask
                    mini_inputs = torch.zeros((bsz, max_count, d), device=device, dtype=x.dtype)
                    mini_mask = torch.zeros((bsz, max_count), device=device, dtype=torch.bool)
                    # fill per-batch
                    # use indexing to avoid python-level gather per token
                    ptrs = [0] * bsz
                    for b, s in zip(batch_idxs.tolist(), seq_idxs.tolist()):
                        i = ptrs[b]
                        mini_inputs[b, i] = x[b, s]
                        mini_mask[b, i] = True
                        ptrs[b] += 1

                    key_padding_mask_for_step = ~mini_mask  # True where padded
                    # process mini-batch (B, max_count, D)
                    if self.config.use_gradient_checkpointing and self.training:
                        mini_out, present_kv = checkpoint(self._process_step, mini_inputs, None, key_padding_mask_for_step, None)
                    else:
                        mini_out, present_kv = self._process_step(mini_inputs, None, key_padding_mask_for_step, None)

                    # scatter outputs back into accum
                    ptrs = [0] * bsz
                    for b in range(bsz):
                        n = (mini_mask[b].sum().item() if mini_mask.dtype == torch.bool else int(mini_mask[b].sum()))
                        if n == 0:
                            continue
                        # find original indices for this batch in step_mask
                        # efficient extraction: use torch.nonzero on step_mask[b]
                        idxs = torch.nonzero(step_mask[b], as_tuple=True)[0]
                        out_seg = mini_out[b, :n, :]  # (n, D)
                        # apply normalized scale
                        accum[b, idxs, :] += out_seg * scales[step] * self.config.residual_connection_strength
                    # handle present_kv caching (choose device)
                    if present_kv is not None:
                        if self.config.kv_cache_device == "cpu":
                            final_present_kv = move_kv_cache(present_kv, device="cpu", to_dtype=torch.float16 if d <= 1024 else None)
                        elif self.config.kv_cache_device == "auto":
                            try:
                                total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
                                if total_mem_gb > 8:
                                    final_present_kv = move_kv_cache(present_kv, device="cuda")
                                else:
                                    final_present_kv = move_kv_cache(present_kv, device="cpu", to_dtype=torch.float16)
                            except Exception:
                                final_present_kv = move_kv_cache(present_kv, device="cpu", to_dtype=torch.float16)
                        else:
                            final_present_kv = move_kv_cache(present_kv, device="cuda")

                total_mask = routing_weights.sum(dim=-1, keepdim=True).clamp(max=1.0)
                output = accum + x * (1.0 - total_mask)
                expected_steps = router_metrics.get("expected_steps", torch.tensor(0.0, device=device))
                # normalize penalty by sequence length to keep scale consistent
                router_metrics["routing_penalty"] = (expected_steps * self.config.recursion_penalty_weight) / max(1.0, float(seqlen))
                router_metrics["entropy_loss"] = entropy_loss
                router_metrics["load_balance_loss"] = load_balance_loss
                return output, router_metrics, final_present_kv

            else:
                # expert-choice path (index assignments)
                assigned_tokens, assigned_experts = routing_result
                if assigned_tokens.numel() == 0:
                    router_metrics["routing_penalty"] = torch.tensor(0.0, device=device)
                    router_metrics["entropy_loss"] = entropy_loss
                    router_metrics["load_balance_loss"] = load_balance_loss
                    return x, router_metrics, None

                b_idx = assigned_tokens // seqlen
                s_idx = assigned_tokens % seqlen

                # process per expert
                for step in range(self.max_steps):
                    mask_ex = (assigned_experts == step)
                    if not mask_ex.any():
                        continue
                    tokens_for_ex = assigned_tokens[mask_ex]
                    b_for = tokens_for_ex // seqlen
                    s_for = tokens_for_ex % seqlen
                    selected = x[b_for, s_for, :].unsqueeze(1)  # (N_e,1,D)
                    if self.config.use_gradient_checkpointing and self.training:
                        compact_output, present_kv = checkpoint(self._process_step, selected, None, None, None)
                    else:
                        compact_output, present_kv = self._process_step(selected, None, None, None)
                    compact_output = compact_output.squeeze(1)
                    accum[b_for, s_for, :] += compact_output * scales[step] * self.config.residual_connection_strength
                    final_present_kv = present_kv

                output = accum + x
                router_metrics["routing_penalty"] = torch.tensor(0.0, device=device)
                router_metrics["entropy_loss"] = entropy_loss
                router_metrics["load_balance_loss"] = load_balance_loss
                return output, router_metrics, final_present_kv

        except Exception as e:
            logger.error(f"RecursiveStack forward error: {e}")
            raise

    def _process_step(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor], past_kv_cache: Optional[KVCache]) -> Tuple[torch.Tensor, KVCache]:
        # x: (B_small, S_small, D)
        present_kv_cache = []
        out = x
        for i, layer in enumerate(self.layers):
            layer_past = None
            if past_kv_cache is not None and i < len(past_kv_cache):
                layer_past = past_kv_cache[i]
            out, present_kv = layer(out, attn_mask=attn_mask, key_padding_mask=key_padding_mask, layer_past=layer_past)
            present_kv_cache.append(present_kv)
        return out, present_kv_cache


class MoRLanguageModel(nn.Module):
    def __init__(self, config: MoRConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim

        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=PAD_TOKEN_ID if PAD_TOKEN_ID is not None else None)
        self.pos_emb = nn.Embedding(config.max_len, config.embed_dim)
        self.emb_norm = nn.LayerNorm(config.embed_dim, eps=1e-5)
        self.emb_dropout = nn.Dropout(config.dropout)

        self.recursive_stack = RecursiveStack(
            depth=config.num_layers,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_hidden_dim,
            dropout=config.dropout,
            max_steps=config.max_recursion_steps,
            config=config
        )

        self.norm = nn.LayerNorm(config.embed_dim, eps=1e-5)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        # tie weights
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()
        logger.info(f"Initialized MoRLanguageModel with {self.count_parameters():,} parameters")

    def _init_weights(self):
        nn.init.xavier_uniform_(self.token_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None, past_kv_cache: Optional[KVCache] = None,
                return_metrics: bool = True, use_cache: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor], KVCache]]:
        try:
            bsz, seq_len = input_ids.shape
            device = input_ids.device

            if seq_len > self.config.max_len:
                raise ValueError("Sequence length exceeds model max_len")

            if position_ids is None:
                past_len = past_kv_cache[0][0].shape[-2] if past_kv_cache is not None else 0
                position_ids = torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(0).expand(bsz, -1)

            token_embeds = self.token_emb(input_ids)
            pos_embeds = self.pos_emb(position_ids)
            x = token_embeds + pos_embeds
            x = self.emb_norm(x)
            x = self.emb_dropout(x)

            causal_mask = None
            if not (use_cache and past_kv_cache is not None):
                causal_mask = torch.triu(torch.full((seq_len, seq_len), torch.finfo(x.dtype).min, device=device), diagonal=1)

            key_padding_mask = (~attention_mask.bool()) if (attention_mask is not None) else None

            x, metrics, present_kv_cache = self.recursive_stack(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask, past_kv_cache=past_kv_cache)

            x = self.norm(x)
            logits = self.lm_head(x)

            if use_cache:
                return logits, metrics, present_kv_cache
            elif return_metrics:
                return logits, metrics
            else:
                return logits

        except Exception as e:
            logger.error(f"MoRLanguageModel forward error: {e}")
            raise


    def generate(self, input_ids: torch.Tensor, max_length: int = 100, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9, do_sample: bool = True, pad_token_id: Optional[int] = None, eos_token_id: Optional[int] = None) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        bsz, seq_len = input_ids.shape
        max_gen_len = min(max_length, self.config.max_len - seq_len)

        generated = input_ids.clone()
        past_kv_cache = None

        with torch.no_grad():
            for i in range(max_gen_len):
                current_ids = generated if i == 0 else generated[:, -1:].contiguous()
                attention_mask = torch.ones_like(current_ids, device=device)
                logits, _, past_kv_cache = self.forward(input_ids=current_ids, attention_mask=attention_mask, past_kv_cache=past_kv_cache, use_cache=True)
                next_token_logits = logits[:, -1, :].float()  # work in fp32 for sampling stability
                # apply temperature
                temp = max(1e-8, float(temperature))
                next_token_logits = next_token_logits / temp
                # clamp logits
                next_token_logits = torch.clamp(next_token_logits, min=-100.0, max=100.0)

                if do_sample:
                    # Top-k
                    if top_k > 0:
                        kth = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)), dim=-1)[0][..., -1, None]
                        mask = next_token_logits < kth
                        next_token_logits = next_token_logits.masked_fill(mask, torch.finfo(next_token_logits.dtype).min)
                    # Top-p
                    if top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(next_token_logits, descending=True)
                        probs = torch.softmax(sorted_logits, dim=-1)
                        cumprobs = torch.cumsum(probs, dim=-1)
                        sorted_mask = cumprobs > top_p
                        sorted_mask[..., 0] = False
                        remove_idx = sorted_mask.scatter(1, sorted_idx, sorted_mask)
                        next_token_logits = next_token_logits.masked_fill(remove_idx, torch.finfo(next_token_logits.dtype).min)
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_tokens.to(device)], dim=-1)

                if eos_token_id is not None and (next_tokens == eos_token_id).all():
                    break

        return generated

    def save_pretrained(self, save_directory: Union[str, Path]):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")
        import json
        config_dict = self.config.__dict__
        with open(save_directory / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_directory: Union[str, Path]):
        model_directory = Path(model_directory)
        import json
        with open(model_directory / "config.json", "r") as f:
            config_dict = json.load(f)
        config = MoRConfig(**config_dict)
        model = cls(config)
        state_dict = torch.load(model_directory / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info(f"Model loaded from {model_directory}")
        return model


class MoRAdapter(nn.Module):
    """Adapter wrapper: training-friendly interface returning losses and metrics."""
    def __init__(self, vocab_size: int, config: Optional[MoRConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = MoRConfig(vocab_size=vocab_size, **kwargs)
        else:
            config.vocab_size = vocab_size
        self.config = config
        self.model = MoRLanguageModel(config)
        self.training_step = 0
        self.metrics_history: List[Dict[str, float]] = []
        logger.info(f"MoRAdapter initialized with config: {config}")

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, return_loss: bool = True, return_metrics: bool = True) -> Dict[str, torch.Tensor]:
        try:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_metrics=return_metrics, use_cache=False)
            if return_metrics:
                logits, metrics = outputs
            else:
                logits = outputs
                metrics = {}

            results: Dict[str, torch.Tensor] = {"logits": logits}

            if labels is not None and return_loss:
                # standard causal shift
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
                lm_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                routing_penalty = metrics.get("routing_penalty", torch.tensor(0.0, device=lm_loss.device))
                entropy_loss = metrics.get("entropy_loss", torch.tensor(0.0, device=lm_loss.device))
                load_balance_loss = metrics.get("load_balance_loss", torch.tensor(0.0, device=lm_loss.device))

                # Normalize and clamp penalty contributions to avoid domination
                lm_detached = lm_loss.detach().clamp(min=1e-8)
                routing_penalty = torch.clamp(routing_penalty, max=(lm_detached * 3.0))
                entropy_term = torch.clamp(entropy_loss, max=(lm_detached * 3.0)) * self.config.entropy_regularization_weight
                load_balance_term = torch.clamp(load_balance_loss, max=(lm_detached * 3.0)) * 0.01

                total_loss = lm_loss + routing_penalty + entropy_term + load_balance_term

                results["loss"] = total_loss
                results["lm_loss"] = lm_loss
                results["routing_penalty"] = routing_penalty
                results["entropy_loss"] = entropy_loss
                results["load_balance_loss"] = load_balance_loss

            if return_metrics:
                if "routing_distribution" in metrics:
                    metrics["routing_distribution"] = metrics["routing_distribution"].detach()
                results.update(metrics)

            if self.training:
                self.training_step += 1
                if len(self.metrics_history) >= 1000:
                    self.metrics_history.pop(0)
                hist_entry = {}
                for k, v in metrics.items():
                    try:
                        hist_entry[k] = float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)
                    except Exception:
                        pass
                if hist_entry:
                    self.metrics_history.append(hist_entry)

            return results

        except Exception as e:
            logger.error(f"MoRAdapter forward error: {e}")
            raise

    def generate(self, *args, **kwargs) -> torch.Tensor:
        return self.model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: Union[str, Path]):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_directory)
        adapter_state = {"training_step": self.training_step, "metrics_history": self.metrics_history[-100:]}
        torch.save(adapter_state, save_directory / "adapter_state.bin")
        logger.info(f"MoRAdapter saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_directory: Union[str, Path]):
        model_directory = Path(model_directory)
        model = MoRLanguageModel.from_pretrained(model_directory)
        adapter = cls(vocab_size=model.config.vocab_size, config=model.config)
        adapter.model = model
        adapter_state_path = model_directory / "adapter_state.bin"
        if adapter_state_path.exists():
            adapter_state = torch.load(adapter_state_path, map_location="cpu")
            adapter.training_step = adapter_state.get("training_step", 0)
            adapter.metrics_history = adapter_state.get("metrics_history", [])
        logger.info(f"MoRAdapter loaded from {model_directory}")
        return adapter

    def get_routing_efficiency_stats(self) -> Dict[str, float]:
        if not self.metrics_history:
            return {}
        recent = self.metrics_history[-100:]
        avg_expected_steps = sum(m.get("expected_steps", 0) for m in recent) / len(recent)
        avg_entropy = sum(m.get("entropy_loss", 0) for m in recent) / len(recent)
        return {
            "avg_expected_steps": float(avg_expected_steps),
            "avg_entropy": float(avg_entropy),
            "routing_efficiency": float(1.0 / max(avg_expected_steps, 1.0)),
            "total_training_steps": self.training_step
        }

    def reset_routing_stats(self):
        try:
            if hasattr(self.model.recursive_stack.router, "step_counts"):
                self.model.recursive_stack.router.step_counts.zero_()
                self.model.recursive_stack.router.total_tokens.zero_()
        except Exception:
            pass

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "MoRLanguageModel",
            "vocab_size": self.config.vocab_size,
            "embed_dim": self.config.embed_dim,
            "num_heads": self.config.num_heads,
            "num_layers": self.config.num_layers,
            "ff_hidden_dim": self.config.ff_hidden_dim,
            "max_len": self.config.max_len,
            "max_recursion_steps": self.config.max_recursion_steps,
            "total_parameters": self.model.count_parameters(),
            "training_step": self.training_step,
            "config": self.config.__dict__
        }

    def set_gradient_checkpointing(self, enable: bool = True):
        self.config.use_gradient_checkpointing = enable
        logger.info(f"Gradient checkpointing {'enabled' if enable else 'disabled'}")

    def freeze_embeddings(self):
        for p in self.model.token_emb.parameters():
            p.requires_grad = False
        for p in self.model.pos_emb.parameters():
            p.requires_grad = False
        logger.info("Embeddings frozen")

    def unfreeze_embeddings(self):
        for p in self.model.token_emb.parameters():
            p.requires_grad = True
        for p in self.model.pos_emb.parameters():
            p.requires_grad = True
        logger.info("Embeddings unfrozen")

    def get_parameter_groups(self, learning_rate: float = 1e-4) -> List[Dict[str, Any]]:
        no_decay = ["bias", "LayerNorm.weight", "norm.weight"]
        router_params = [p for n, p in self.named_parameters() if "router" in n]
        transformer_decay = [p for n, p in self.named_parameters() if "router" not in n and not any(nd in n for nd in no_decay)]
        transformer_no_decay = [p for n, p in self.named_parameters() if "router" not in n and any(nd in n for nd in no_decay)]
        return [
            {"params": transformer_decay, "weight_decay": 0.01, "lr": learning_rate},
            {"params": transformer_no_decay, "weight_decay": 0.0, "lr": learning_rate},
            {"params": router_params, "weight_decay": 0.01, "lr": learning_rate * 2.0}
        ]
