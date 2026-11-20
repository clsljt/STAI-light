"""STAI (lite): a masked-reconstruction autoencoder whose channel-independent temporal tokens read a compact,
cross-channel spectral memory.

    spectral path   x -> InstanceNorm -> rFFT -> bin tokens [Re, Im, log(1+|X|)]
                      -> K learnable global tokens cross-attend to the bin tokens        => memory (B*C, K, D)
                      -> channel fusion: per global token, attention across channels
    temporal path   x -> RevIN -> per-channel conv embedding                             => h (B*C, L, D)
    fusion          every time step cross-attends to the K memory tokens of its channel (gated residual + FFN)
    decoder         Linear over time (L -> L), Linear D -> 1, RevIN denormalisation

Shapes: B batch, L seq_len, C channels, D d_model, K num_global_tokens, F = L // 2 + 1 frequency bins.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PostAttentionFFN(nn.Module):
    """x + FFN(LayerNorm(x)), hidden size 4 * D."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.ffn(self.norm(x))


class SpectralGlobalMemory(nn.Module):
    """x (B, L, C) -> memory (B*C, K, D): K learnable tokens attend to the F rFFT bins of each channel's window."""

    def __init__(self, seq_len, enc_in, d_model, n_heads, num_global_tokens, dropout):
        super().__init__()
        self.freq_len = seq_len // 2 + 1
        self.input_norm = nn.InstanceNorm1d(enc_in, affine=True)
        self.freq_mlp = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.freq_embedding = nn.Parameter(torch.randn(1, self.freq_len, d_model) * 0.02)
        self.global_tokens = nn.Parameter(torch.randn(1, num_global_tokens, d_model) * 0.02)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.post_attn_ffn = PostAttentionFFN(d_model, dropout)

    def forward(self, x):
        B, L, C = x.shape
        x = self.input_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        spec = torch.fft.rfft(x, dim=1)                                                    # (B, F, C)
        feat = torch.stack([spec.real, spec.imag, torch.log1p(torch.abs(spec))], dim=-1)   # (B, F, C, 3)
        feat = feat.permute(0, 2, 1, 3).contiguous().reshape(B * C, self.freq_len, 3)
        bins = self.norm_kv(self.freq_mlp(feat) + self.freq_embedding)                     # (B*C, F, D)
        tokens = self.global_tokens.expand(B * C, -1, -1)
        updated, _ = self.cross_attention(self.norm_q(tokens), bins, bins)
        return self.post_attn_ffn(tokens + updated)                                        # (B*C, K, D)


class ChannelFusion(nn.Module):
    """For each global token the C channels attend to each other; added back with a scalar gate sigmoid(g)."""

    def __init__(self, d_model, n_heads, dropout, gate_init):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, memory, batch_size, channels):
        _, K, D = memory.shape
        x = memory.reshape(batch_size, channels, K, D).permute(0, 2, 1, 3).reshape(batch_size * K, channels, D)
        normed = self.norm(x)
        attended, _ = self.attn(normed, normed, normed)
        x = x + torch.sigmoid(self.gate) * attended
        return x.reshape(batch_size, K, channels, D).permute(0, 2, 1, 3).reshape(batch_size * channels, K, D)


class MemoryCrossAttention(nn.Module):
    """Temporal tokens read the memory:  fused = h + a * g * Attn(h, m),  out = fused + a * FFN(fused),
    with a = sigmoid(scalar gate) and g = sigmoid(MLP([h, context])) per time step and dimension."""

    def __init__(self, d_model, n_heads, dropout, gate_init):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.token_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.Sigmoid(),
        )
        self.residual_gate = nn.Parameter(torch.tensor(gate_init))
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, h, memory):
        alpha = torch.sigmoid(self.residual_gate)
        memory = self.norm_kv(memory)
        context, _ = self.attn(self.norm_q(h), memory, memory)
        fused = h + alpha * self.token_gate(torch.cat([h, context], dim=-1)) * context
        return fused + alpha * self.ffn(fused)


class RevIN(nn.Module):
    """Reversible instance normalisation over time, per window and channel, with affine parameters."""

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.stdev * self.affine_weight + self.affine_bias
        x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        return x * self.stdev + self.mean


class TokenEmbedding(nn.Module):
    """Point value -> D with a circular Conv1d (kernel 3, no bias)."""

    def __init__(self, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(1, d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class ValueEmbedding(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.value_embedding = TokenEmbedding(d_model)
        # The experiments in the paper were run with a code base whose embedding also built a (never used) 5 -> D
        # time-feature layer here.  Building and discarding it keeps the random initialisation stream, and hence
        # every seed, identical to those runs; it adds no parameters to this model.
        nn.Linear(5, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x))


class Model(nn.Module):
    def __init__(self, seq_len, enc_in, d_model=64, n_heads=4, num_global_tokens=16, dropout=0.1,
                 channel_fusion_gate_init=-3.0, cross_attn_gate_init=-3.0):
        super().__init__()
        self.seq_len = seq_len
        self.global_spectral_attention = SpectralGlobalMemory(seq_len, enc_in, d_model, n_heads, num_global_tokens, dropout)
        self.channel_global_fusion = ChannelFusion(d_model, n_heads, dropout, channel_fusion_gate_init)
        self.global_injector = MemoryCrossAttention(d_model, n_heads, dropout, cross_attn_gate_init)
        self.normalize_layers = nn.ModuleList([RevIN(enc_in)])
        self.embedding = ValueEmbedding(d_model, dropout)
        self.reconstruct_layers = nn.ModuleList([nn.Linear(seq_len, seq_len)])
        self.projection = nn.Linear(d_model, 1)

    def forward(self, x):
        """x: (B, L, C) standardised window with masked steps set to 0 -> reconstruction (B, L, C)."""
        B, L, C = x.shape
        memory = self.channel_global_fusion(self.global_spectral_attention(x), B, C)
        h = self.normalize_layers[0](x, "norm").permute(0, 2, 1).reshape(B * C, L, 1)
        h = self.global_injector(self.embedding(h), memory)
        out = self.reconstruct_layers[0](h.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.projection(out).reshape(B, C, L).permute(0, 2, 1)
        return self.normalize_layers[0](out, "denorm")
