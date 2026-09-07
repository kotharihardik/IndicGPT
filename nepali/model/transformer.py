"""Decoder-only Transformer language model.

Shapes: B batch, T sequence length, C model dimension, h heads, dk = C//h,
V vocabulary.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Configuration


@dataclass
class ModelConfig:
    """Every architectural choice in one place, saved with each checkpoint.

    Attributes:
        vocab_size: Size of this language's own vocabulary (Phase 1).
        d_model: Model dimension C, the width of every residual stream.
        n_layers: Number of stacked Transformer blocks.
        n_heads: Attention heads per block; `d_model` must divide by it.
        d_ff: Inner width of the feed-forward network.
        context_length: Longest sequence trained on, used to size the mask.
        dropout: Rate on the embedding and both sublayer outputs.
        attn_dropout: Rate on the attention probabilities.
        positional: `rope`, `learned`, `sinusoidal` or `none`.
        rope_theta: RoPE base frequency.
        ffn_activation: `swiglu` or `gelu`.
        norm_eps: LayerNorm epsilon.
        tie_embeddings: Reuse the token embedding as the output projection.
        bias: Add bias terms to the linear projections.
        init_std: Standard deviation of the normal weight initialisation.
        pad_id, unk_id, bos_id, eos_id: Fixed special-token ids from Phase 1.
    """

    vocab_size: int
    d_model: int = 384
    n_layers: int = 11
    n_heads: int = 6
    d_ff: int = 1024
    context_length: int = 512
    dropout: float = 0.0
    attn_dropout: float = 0.0
    positional: str = "rope"
    rope_theta: float = 10000.0
    ffn_activation: str = "swiglu"
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    bias: bool = False
    init_std: float = 0.02
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    #: Fields that must be numeric whatever the config said.
    _INT_FIELDS = ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff",
                   "context_length", "pad_id", "unk_id", "bos_id", "eos_id")
    _FLOAT_FIELDS = ("dropout", "attn_dropout", "rope_theta", "norm_eps", "init_std")

    def __post_init__(self) -> None:
        # PyYAML reads 1e-05 as a string, not a float (YAML 1.1 needs 1.0e-05).
        for name in self._INT_FIELDS:
            setattr(self, name, int(getattr(self, name)))
        for name in self._FLOAT_FIELDS:
            setattr(self, name, float(getattr(self, name)))
        self.tie_embeddings = bool(self.tie_embeddings)
        self.bias = bool(self.bias)

        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}")
        if self.positional not in ("rope", "learned", "sinusoidal", "none"):
            raise ValueError(f"unknown positional scheme: {self.positional}")
        if self.ffn_activation not in ("swiglu", "gelu"):
            raise ValueError(f"unknown ffn activation: {self.ffn_activation}")
        # RoPE rotates the head dimension in pairs, so it must be even.
        if self.positional == "rope" and self.head_dim % 2:
            raise ValueError(f"RoPE needs an even head dimension, got {self.head_dim}")

    @property
    def head_dim(self) -> int:
        """Width of the subspace each attention head works in."""
        return self.d_model // self.n_heads

    @classmethod
    def from_dict(cls, blob: Dict) -> "ModelConfig":
        """Build a config from a dict, ignoring keys this version does not know."""
        known = {k: v for k, v in blob.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> Dict:
        """Plain dict for JSON / checkpoint storage."""
        return asdict(self)


# Positional information


class RotaryPositionalEmbedding(nn.Module):
    """RoPE: rotate Q and K by an angle proportional to position.

    The dot product of two rotated vectors depends only on the difference of
    their angles, so attention sees relative distance. Chosen over a learned
    table because it costs zero parameters and is not limited to one sequence
    length. Applied to Q and K only; V carries content, not position.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        cos, sin = self._build(max_seq_len)
        # Not persistent: lets a checkpoint be reloaded at a different length.
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _build(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache cos/sin of `position * frequency` for every position and pair."""
        # One frequency per channel pair, geometrically spaced.
        exponent = torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        inv_freq = 1.0 / (self.theta ** exponent)              # (dk/2,)
        pos = torch.arange(seq_len, dtype=torch.float32)       # (T,)
        angles = torch.outer(pos, inv_freq)                    # (T, dk/2)
        return angles.cos(), angles.sin()

    def _cache(self, seq_len: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin covering `seq_len`, extending the cache if needed."""
        if seq_len > self.cos.shape[0]:
            cos, sin = self._build(seq_len)
            self.cos, self.sin = cos.to(device), sin.to(device)
        return self.cos[:seq_len].to(dtype), self.sin[:seq_len].to(dtype)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Rotate one projected tensor in place of its positions.

        Args:
            x: Queries or keys, shape (B, h, T, dk).
            offset: Position of the first token, non-zero when continuing a
                sequence during generation.

        Returns:
            The same shape, with each channel pair rotated by its position angle.
        """
        seq_len = x.shape[-2]
        cos, sin = self._cache(offset + seq_len, x.device, torch.float32)
        cos = cos[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)   # (1,1,T,dk/2)
        sin = sin[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)

        # Split-half: channel i pairs with i + dk/2. Rotate in float32.
        x1, x2 = x.float().chunk(2, dim=-1)                            # (B,h,T,dk/2)
        rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return rotated.to(x.dtype)


def sinusoidal_table(seq_len: int, d_model: int) -> torch.Tensor:
    """Fixed sinusoidal position table, kept as a selectable alternative."""
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    exponent = torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
    inv_freq = 1.0 / (10000.0 ** exponent)
    table = torch.zeros(seq_len, d_model)
    table[:, 0::2] = torch.sin(pos * inv_freq)
    table[:, 1::2] = torch.cos(pos * inv_freq)
    return table


# Attention


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, written out operation by operation.

    Three separate projections rather than a fused Linear(C, 3C): identical
    arithmetic, but it maps directly onto Q = X W_Q.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        # 1/sqrt(dk): without it scores grow like dk and softmax saturates.
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        self.rope = (RotaryPositionalEmbedding(cfg.head_dim, cfg.context_length,
                                               cfg.rope_theta)
                     if cfg.positional == "rope" else None)

        # Additive causal mask: 0 where a query may look, -inf on the future.
        mask = torch.full((cfg.context_length, cfg.context_length), float("-inf"))
        self.register_buffer("mask", torch.triu(mask, diagonal=1), persistent=False)

    def _causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """Additive mask for a window of `seq_len` tokens: 0 allowed, -inf future."""
        if seq_len > self.mask.shape[0]:
            full = torch.full((seq_len, seq_len), float("-inf"), device=device)
            self.mask = torch.triu(full, diagonal=1)
        return self.mask[:seq_len, :seq_len]

    def forward(self, x: torch.Tensor, offset: int = 0,
                need_weights: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run one attention sublayer.

        Args:
            x: Input activations, shape (B, T, C).
            offset: Position of the first token of `x` in the wider sequence.
                Only shifts the RoPE phase; the model has no key/value cache, so
                every forward pass attends within `x` alone.
            need_weights: Also return the attention probabilities, for analysis.

        Returns:
            (output of shape (B, T, C), attention weights (B, h, T, T) or None).
        """
        B, T, C = x.shape

        # (B,T,C) -> (B,T,h,dk) -> (B,h,T,dk): one dk-wide slice per head.
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

        # Scores: (B,h,T,dk) @ (B,h,dk,T) -> (B,h,T,T).
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Mask, then softmax in float32: fp16 loses accuracy here first.
        scores = scores.float() + self._causal_mask(T, x.device)
        weights = torch.softmax(scores, dim=-1).to(v.dtype)

        y = torch.matmul(self.attn_dropout(weights), v)          # (B,h,T,dk)

        # Concatenate the heads, then mix them with W_O.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))
        return y, (weights if need_weights else None)


# Feed-forward


class FeedForward(nn.Module):
    """Position-wise feed-forward network, applied at every token.

    swiglu uses 3 matrices of width d_ff, gelu uses 2, so d_ff 1024 under
    SwiGLU costs what 1536 costs under GELU at d_model 384.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.activation = cfg.ffn_activation
        if self.activation == "swiglu":
            self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
            self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        else:
            self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Widen, gate or squash, project back, drop out. Shapes: (B,T,C)->(B,T,C)."""
        if self.activation == "swiglu":
            hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        else:
            hidden = F.gelu(self.up_proj(x), approximate="tanh")
        return self.dropout(self.down_proj(hidden))


# Block


class TransformerBlock(nn.Module):
    """One pre-norm decoder block: attention sublayer, then feed-forward.

    Pre-norm: x = x + Attn(LN(x)); x = x + FFN(LN(x)). The LayerNorm sits
    inside each residual branch, leaving an unbroken identity path through the
    stack, which is what makes a deep stack train without a long warmup.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, offset: int = 0,
                need_weights: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply both sublayers with their residual connections."""
        attn_out, weights = self.attn(self.ln1(x), offset=offset,
                                      need_weights=need_weights)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, weights


# Language model


class DecoderLM(nn.Module):
    """The full decoder-only language model.

    The output projection reuses the token-embedding matrix (weight tying),
    saving 6.14M parameters at V=16,000 and C=384.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_dropout = nn.Dropout(cfg.dropout)

        # RoPE lives inside attention; the other schemes are additive tables.
        if cfg.positional == "learned":
            self.position_embedding = nn.Embedding(cfg.context_length, cfg.d_model)
        elif cfg.positional == "sinusoidal":
            self.register_buffer("position_table",
                                 sinusoidal_table(cfg.context_length, cfg.d_model),
                                 persistent=False)

        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        scale = self.cfg.init_std / math.sqrt(2 * cfg.n_layers)
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=scale)

    def _init_weights(self, module: nn.Module) -> None:
        """Normal(0, init_std) for weights, zeros for biases."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Trainable parameter count, optionally excluding the lookup tables."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if not non_embedding:
            return total
        total -= self.token_embedding.weight.numel()
        if self.cfg.positional == "learned":
            total -= self.position_embedding.weight.numel()
        if not self.cfg.tie_embeddings:
            total -= self.lm_head.weight.numel()
        return total

    def parameter_table(self) -> List[Tuple[str, int]]:
        """Parameter count grouped by component, for the report."""
        groups: Dict[str, int] = {}
        for name, param in self.named_parameters():
            if name.startswith("blocks."):
                # blocks.<i>.<component>... -> one row per component
                part = name.split(".", 2)[2]
                key = "block." + part.rsplit(".", 1)[0]
            else:
                key = name.rsplit(".", 1)[0]
            groups[key] = groups.get(key, 0) + param.numel()
        return sorted(groups.items(), key=lambda kv: -kv[1])

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                offset: int = 0, need_weights: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """Map token ids to next-token logits, and optionally to a loss.

        Args:
            idx: Token ids, shape (B, T).
            targets: Next-token ids, shape (B, T). When given, the mean
                cross-entropy over all positions is returned as well.
            offset: Position of the first token, used during generation.
            need_weights: Collect per-layer attention probabilities.

        Returns:
            (logits (B, T, V), loss or None, list of (B, h, T, T) or None).
        """
        B, T = idx.shape
        x = self.token_embedding(idx)                       # (B, T, C)

        if self.cfg.positional == "learned":
            pos = torch.arange(offset, offset + T, device=idx.device)
            x = x + self.position_embedding(pos)[None, :, :]
        elif self.cfg.positional == "sinusoidal":
            x = x + self.position_table[offset:offset + T].to(x.dtype)[None, :, :]

        x = self.embed_dropout(x)

        attentions: List[torch.Tensor] = [] if need_weights else None
        for block in self.blocks:
            x, weights = block(x, offset=offset, need_weights=need_weights)
            if need_weights:
                attentions.append(weights)

        x = self.ln_f(x)
        logits = self.lm_head(x)                            # (B, T, V)

        loss = None
        if targets is not None:
            # The t -> t+1 shift is applied by the data pipeline, not here.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss, attentions

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, greedy: bool = False,
                 top_k: Optional[int] = None,
                 eos_id: Optional[int] = None) -> torch.Tensor:
        """Continue each row of `idx` for `max_new_tokens` steps.

        Args:
            idx: Prompt token ids, shape (B, T0).
            max_new_tokens: How many tokens to append.
            temperature: Divides the logits before softmax. Below 1 sharpens
                the distribution, above 1 flattens it.
            greedy: Take the argmax instead of sampling; `temperature` is then
                irrelevant, since argmax is invariant to positive scaling.
            top_k: Optionally keep only the k most likely tokens per step.
            eos_id: Stop a row once it emits this token (it is still appended).

        Returns:
            (B, T0 + max_new_tokens) token ids.
        """
        self.eval()
        finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            # Only the most recent `context_length` tokens are fed back in.
            window = idx[:, -self.cfg.context_length:]
            logits, _, _ = self(window)
            step = logits[:, -1, :].float()                  # (B, V)

            if greedy:
                nxt = step.argmax(dim=-1, keepdim=True)
            else:
                step = step / max(temperature, 1e-6)
                if top_k:
                    kth = torch.topk(step, min(top_k, step.size(-1)), dim=-1).values[:, -1:]
                    step = step.masked_fill(step < kth, float("-inf"))
                nxt = torch.multinomial(torch.softmax(step, dim=-1), num_samples=1)

            if eos_id is not None:
                # A finished row keeps emitting eos, so the batch stays rectangular.
                nxt = torch.where(finished.unsqueeze(1),
                                  torch.full_like(nxt, eos_id), nxt)
                finished |= nxt.squeeze(1) == eos_id
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and bool(finished.all()):
                break
        return idx


def build_model(cfg: ModelConfig) -> DecoderLM:
    """Construct the model described by `cfg`."""
    return DecoderLM(cfg)


# Self-test


def _self_test(vocab_size: int = 512, verbose: bool = True) -> bool:
    """Check the properties that make this a correct causal Transformer.

    Fast enough to run before every training job.
    """
    ok = True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(passed)
        if verbose:
            print(f"  {name:<44}{detail:<26}{'ok' if passed else 'FAILED'}")

    torch.manual_seed(0)

    # 1. RoPE is relative: the score must depend only on m - n.
    rope = RotaryPositionalEmbedding(64, 512)
    q, k = torch.randn(1, 1, 1, 64), torch.randn(1, 1, 1, 64)

    def score(m: int, n: int) -> float:
        """Attention score between a query at m and a key at n, before softmax."""
        return float((rope(q, offset=m) * rope(k, offset=n)).sum())

    near = abs(score(5, 3) - score(400, 398))
    check("RoPE score depends only on m - n", near < 1e-4, f"delta {near:.1e}")
    check("RoPE distinguishes distances", abs(score(5, 3) - score(10, 4)) > 1e-3)
    check("RoPE has no parameters", sum(p.numel() for p in rope.parameters()) == 0)
    check("RoPE preserves the norm",
          torch.allclose(rope(q, offset=37).norm(), q.norm(), atol=1e-5))

    # 2. A small model, on the real code path.
    cfg = ModelConfig(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=4,
                      d_ff=176, context_length=32)
    model = DecoderLM(cfg).eval()
    ids = torch.randint(0, vocab_size, (2, 16))
    targets = torch.randint(0, vocab_size, (2, 16))
    with torch.no_grad():
        logits, loss, attn = model(ids, targets=targets, need_weights=True)

    check("logit shape", tuple(logits.shape) == (2, 16, vocab_size),
          str(tuple(logits.shape)))
    check("attention returned per layer", len(attn) == cfg.n_layers,
          f"{len(attn)} layers")
    check("attention rows sum to 1",
          torch.allclose(attn[0].sum(-1), torch.ones(2, 4, 16), atol=1e-5))
    check("no weight above the diagonal",
          float(attn[-1].triu(1).abs().max()) == 0.0)

    # 3. An untrained model's first loss should sit close to ln(V).
    expected = math.log(vocab_size)
    check("initial loss near ln(vocab)", abs(float(loss) - expected) < 1.0,
          f"{float(loss):.2f} vs {expected:.2f}")

    # 4. RoPE is not a lookup table, so a longer sequence must simply run.
    with torch.no_grad():
        longer, _, _ = model(torch.randint(0, vocab_size, (1, cfg.context_length * 2)))
    check("runs past the trained context", longer.shape[1] == cfg.context_length * 2,
          f"{longer.shape[1]} tokens")

    return bool(ok)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Decoder-only Transformer self-test.")
    ap.add_argument("--self-test", action="store_true",
                    help="check RoPE, attention, masking and initialisation")
    args = ap.parse_args()
    if args.self_test:
        print("\n  Transformer self-test\n  " + "-" * 76)
        passed = _self_test()
        print(f"\n  overall: {'PASS' if passed else 'FAIL'}\n")
        raise SystemExit(0 if passed else 1)
    ap.print_help()
