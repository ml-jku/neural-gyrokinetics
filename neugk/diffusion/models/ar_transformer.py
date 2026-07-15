"""Autoregressive transformer for discrete VQVAE token prediction."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neugk.models.layers import Film


class DiTCausalBlock(nn.Module):
    """Causal transformer block with DiT-style adaptive LayerNorm conditioning.

    Attribute names (norm1, norm2, self_attn, linear1, linear2, activation)
    mirror ``nn.TransformerEncoderLayer`` so the KV-cache helpers can reuse
    the same internal access pattern.
    """

    def __init__(
        self, dim: int, num_heads: int, ff_dim: int, dropout: float, cond_dim: int
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, dim)
        self.activation = nn.GELU()

        # adaLN: scale1, shift1, gate1, scale2, shift2, gate2
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim),
        )
        # zero-init so modulation starts as identity
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, attn_mask: torch.Tensor = None
    ):
        mod = self.adaLN(cond).unsqueeze(1)  # (B, 1, 6D)
        s1, sh1, g1, s2, sh2, g2 = mod.chunk(6, dim=-1)

        # self-attention with adaLN
        residual = x
        h = self.norm1(x) * (1 + s1) + sh1
        h = self.self_attn(h, h, h, attn_mask=attn_mask)[0]
        x = residual + g1 * h

        # FFN with adaLN
        residual = x
        h = self.norm2(x) * (1 + s2) + sh2
        h = self.linear2(self.activation(self.linear1(h)))
        x = residual + g2 * h

        return x


class ARTransformer(nn.Module):
    """GPT-style causal transformer for next-token prediction of VQ codes.

    Supports three conditioning modes (``conditioning_mode``):

    * ``"cls"`` *(default)* -- plasma parameters projected to a prefix token.
    * ``"film"`` -- per-layer FiLM (scale + shift) modulation.
    * ``"dit"`` -- per-layer adaptive LayerNorm (scale + shift + gate).

    Position ``i`` in the output predicts token ``i`` of the VQ sequence,
    so the loss is simply ``CE(logits, token_ids)``.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        seq_len: int,
        depth: int,
        num_heads: int,
        cond_embed: Optional[nn.Module] = None,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
        conditioning_mode: str = "cls",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dim = dim
        self.label_smoothing = label_smoothing
        self.conditioning_mode = conditioning_mode

        self.token_embed = nn.Embedding(vocab_size, dim)
        # +1 for the prepended prefix (cond token for cls, learned start for film/dit)
        self.pos_embed = nn.Embedding(seq_len + 1, dim)

        self.cond_embed = cond_embed
        cond_dim = cond_embed.cond_dim if cond_embed is not None else None

        # --- build layers per conditioning mode ---
        if conditioning_mode == "cls":
            if cond_embed is not None:
                self.cond_proj = nn.Linear(cond_dim, dim)
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=dim,
                        nhead=num_heads,
                        dim_feedforward=dim * 4,
                        dropout=dropout,
                        batch_first=True,
                        norm_first=True,
                        activation="gelu",
                    )
                    for _ in range(depth)
                ]
            )

        elif conditioning_mode == "film":
            self.start_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=dim,
                        nhead=num_heads,
                        dim_feedforward=dim * 4,
                        dropout=dropout,
                        batch_first=True,
                        norm_first=True,
                        activation="gelu",
                    )
                    for _ in range(depth)
                ]
            )
            self.films = nn.ModuleList([Film(cond_dim, dim) for _ in range(depth)])

        elif conditioning_mode == "dit":
            self.start_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            self.layers = nn.ModuleList(
                [
                    DiTCausalBlock(dim, num_heads, dim * 4, dropout, cond_dim)
                    for _ in range(depth)
                ]
            )

        else:
            raise ValueError(f"Unknown conditioning_mode: {conditioning_mode}")

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # expose for the runner
        self.latent_shape = (seq_len,)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        if self.conditioning_mode == "cls" and self.cond_embed is not None:
            nn.init.normal_(self.cond_proj.weight, std=0.02)
            if self.cond_proj.bias is not None:
                nn.init.zeros_(self.cond_proj.bias)

    # ------------------------------------------------------------------
    # forward: teacher-forced training
    # ------------------------------------------------------------------
    def forward(
        self,
        token_ids: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        tstep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Args:
            token_ids: ``(B, S)`` int64 -- full VQ index sequence.
            condition:  ``(B, n_cond)`` float -- plasma parameters.
            tstep: ignored (accepted for interface compatibility).

        Returns:
            logits: ``(B, S, vocab_size)`` aligned with *token_ids* as targets.
        """
        B, S = token_ids.shape
        x = self.token_embed(token_ids)  # (B, S, D)

        # build prefix
        if self.conditioning_mode == "cls":
            if self.cond_embed is not None and condition is not None:
                c = self.cond_proj(self.cond_embed(condition)).unsqueeze(1)  # (B,1,D)
                x = torch.cat([c, x], dim=1)  # (B, S+1, D)
        else:  # film / dit — learned start token
            x = torch.cat([self.start_token.expand(B, -1, -1), x], dim=1)

        x = x + self.pos_embed(torch.arange(x.shape[1], device=x.device))

        # causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(
            x.shape[1], device=x.device
        )

        if self.conditioning_mode == "cls":
            for layer in self.layers:
                x = layer(x, src_mask=mask, is_causal=True)

        elif self.conditioning_mode == "film":
            cond = self.cond_embed(condition)
            for layer, film in zip(self.layers, self.films):
                x = film(x, cond)
                x = layer(x, src_mask=mask, is_causal=True)

        elif self.conditioning_mode == "dit":
            cond = self.cond_embed(condition)
            for layer in self.layers:
                x = layer(x, cond=cond, attn_mask=mask)

        x = self.ln_f(x)

        # positions [0 .. S-1] predict tokens [0 .. S-1]
        logits = self.head(x[:, :S])  # (B, S, V)
        return logits

    # ------------------------------------------------------------------
    # generate: autoregressive sampling with KV-cache
    # ------------------------------------------------------------------

    def _cached_layer_step(self, layer, x_new, kv_cache):
        """One new-token step through a standard TransformerEncoderLayer.

        Only computes Q/K/V for the new position and appends K/V to the
        running cache, giving O(1) work per layer per step instead of O(L).
        """
        mha = layer.self_attn
        dim = mha.embed_dim
        num_heads = mha.num_heads
        head_dim = dim // num_heads
        B = x_new.shape[0]

        # --- self-attention (pre-norm) ---
        residual = x_new
        x_norm = layer.norm1(x_new)  # (B, 1, D)

        qkv = F.linear(x_norm, mha.in_proj_weight, mha.in_proj_bias)
        q, k, v = qkv.split(dim, dim=-1)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)

        q = q.view(B, 1, num_heads, head_dim).transpose(1, 2)
        k_mh = k.view(B, -1, num_heads, head_dim).transpose(1, 2)
        v_mh = v.view(B, -1, num_heads, head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k_mh, v_mh)
        attn_out = attn_out.transpose(1, 2).reshape(B, 1, dim)
        attn_out = mha.out_proj(attn_out)

        x_new = residual + attn_out

        # --- FFN (pre-norm) ---
        residual = x_new
        x_norm = layer.norm2(x_new)
        x_new = residual + layer.linear2(layer.activation(layer.linear1(x_norm)))

        return x_new, (k, v)

    def _cached_dit_step(self, layer, x_new, kv_cache, cond):
        """One new-token step through a DiTCausalBlock with KV cache."""
        mha = layer.self_attn
        dim = mha.embed_dim
        num_heads = mha.num_heads
        head_dim = dim // num_heads
        B = x_new.shape[0]

        mod = layer.adaLN(cond).unsqueeze(1)  # (B, 1, 6D)
        s1, sh1, g1, s2, sh2, g2 = mod.chunk(6, dim=-1)

        # --- self-attention with adaLN ---
        residual = x_new
        x_norm = layer.norm1(x_new) * (1 + s1) + sh1

        qkv = F.linear(x_norm, mha.in_proj_weight, mha.in_proj_bias)
        q, k, v = qkv.split(dim, dim=-1)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)

        q = q.view(B, 1, num_heads, head_dim).transpose(1, 2)
        k_mh = k.view(B, -1, num_heads, head_dim).transpose(1, 2)
        v_mh = v.view(B, -1, num_heads, head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k_mh, v_mh)
        attn_out = attn_out.transpose(1, 2).reshape(B, 1, dim)
        attn_out = mha.out_proj(attn_out)

        x_new = residual + g1 * attn_out

        # --- FFN with adaLN ---
        residual = x_new
        x_norm = layer.norm2(x_new) * (1 + s2) + sh2
        x_new = residual + g2 * layer.linear2(layer.activation(layer.linear1(x_norm)))

        return x_new, (k, v)

    def _step_all_layers(self, x_new, kv_caches, cond=None):
        """Push one new position through every transformer layer."""
        if self.conditioning_mode == "cls":
            for i, layer in enumerate(self.layers):
                x_new, kv_caches[i] = self._cached_layer_step(
                    layer, x_new, kv_caches[i]
                )
        elif self.conditioning_mode == "film":
            for i, (layer, film) in enumerate(zip(self.layers, self.films)):
                x_new = film(x_new, cond)
                x_new, kv_caches[i] = self._cached_layer_step(
                    layer, x_new, kv_caches[i]
                )
        elif self.conditioning_mode == "dit":
            for i, layer in enumerate(self.layers):
                x_new, kv_caches[i] = self._cached_dit_step(
                    layer, x_new, kv_caches[i], cond
                )
        return x_new

    @torch.no_grad()
    def generate(
        self,
        condition: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        device: torch.device = "cuda",
    ) -> torch.Tensor:
        """Autoregressively sample a full sequence of VQ indices.

        Uses KV-caching so each step is O(S) instead of O(S^2).

        Returns:
            indices: ``(B, seq_len)`` int64.
        """
        B = condition.shape[0] if condition is not None else 1
        num_layers = len(self.layers)
        kv_caches = [None] * num_layers

        # build prefix & conditioning vector
        cond = None
        if self.conditioning_mode == "cls":
            if self.cond_embed is not None and condition is not None:
                prefix = self.cond_proj(self.cond_embed(condition)).unsqueeze(1)
            else:
                prefix = torch.zeros(B, 1, self.dim, device=device)
        else:  # film / dit — learned start token + per-layer conditioning
            prefix = self.start_token.expand(B, -1, -1)
            if self.cond_embed is not None and condition is not None:
                cond = self.cond_embed(condition)

        # run prefix through all layers (populates KV caches)
        x = prefix + self.pos_embed(torch.tensor([0], device=device))
        x = self._step_all_layers(x, kv_caches, cond=cond)

        tokens = []
        for i in range(self.seq_len):
            # logits from the current last position
            logits = self.head(self.ln_f(x.squeeze(1)))  # (B, V)

            if temperature > 0:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)  # (B, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            tokens.append(next_token)

            # prepare next step (skip on last iteration)
            if i < self.seq_len - 1:
                pos = torch.tensor([i + 1], device=device)
                x = self.token_embed(next_token) + self.pos_embed(pos)
                x = self._step_all_layers(x, kv_caches, cond=cond)

        return torch.cat(tokens, dim=1)  # (B, seq_len)
