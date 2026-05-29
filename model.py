import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

class RMSNorm(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        self.eps = config.get("norm_eps", 1e-6)
        self.weights = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = out * self.weights.float()
        return out.to(dtype=x.dtype)

class FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config["d_model"]
        ffn_ratio = config["ffn_ratio"]
        self.up_proj = nn.Linear(d_model, 2 * ffn_ratio * d_model, bias=False)
        self.down_proj = nn.Linear(ffn_ratio * d_model, d_model, bias=False)
    
    def forward(self, x):
        x, gate = self.up_proj(x).chunk(2, dim=-1)
        return self.down_proj(x * F.silu(gate))

class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["head_dim"]
        max_seq_len = config["max_seq_len"]
        base = config.get("rope_base", 10000)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
        freq = torch.outer(torch.arange(max_seq_len), inv_freq)
        self.register_buffer("sin", freq.sin()[None, None])
        self.register_buffer("cos", freq.cos()[None, None])

    def forward(self, x, offset=0):
        T = x.size(-2)
        sin = self.sin[:, :, offset:offset + T]
        cos = self.cos[:, :, offset:offset + T]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([cos * x1 - sin * x2, sin * x1 + cos * x2], dim=-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config["d_model"]
        n_heads = config["n_heads"]
        kv_heads = config["kv_heads"]
        head_dim = d_model // n_heads
        
        assert n_heads % kv_heads == 0
        self.n_heads = n_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.group_size = n_heads // kv_heads
        
        self.q_norm = RMSNorm(head_dim, config)
        self.k_norm = RMSNorm(head_dim, config)
        
        self.rope = RotaryEmbedding(config)
        
        self.attn_proj = nn.Linear(d_model, d_model + kv_heads * 2 * head_dim, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, cache, cu_seqlens=None, n_docs=None, max_seqlen=None):
        B, T, _ = x.shape

        qkv = self.attn_proj(x)
        q, k, v = rearrange(qkv, "B T (H C) -> B H T C", C=self.head_dim).split([self.n_heads, self.kv_heads, self.kv_heads], dim=1)

        prev_len = cache[0].shape[2] if cache is not None else 0

        q = self.rope(self.q_norm(q), offset=prev_len)
        k = self.rope(self.k_norm(k), offset=prev_len)

        if cache is not None:
            if cu_seqlens is not None:
                raise ValueError("kv_cache and cu_seqlens are mutually exclusive")
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        if cu_seqlens is not None:
            q = q.permute(0, 2, 1, 3).reshape(-1, self.n_heads, self.head_dim).contiguous()
            k = k.permute(0, 2, 1, 3).reshape(-1, self.kv_heads, self.head_dim).contiguous()
            v = v.permute(0, 2, 1, 3).reshape(-1, self.kv_heads, self.head_dim).contiguous()
            
            actual_cu = cu_seqlens[: n_docs + 1].contiguous()
            actual_max = max_seqlen if max_seqlen is not None else T
            
            attn_out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=actual_cu,
                cu_seqlens_k=actual_cu,
                max_seqlen_q=actual_max,
                max_seqlen_k=actual_max,
                causal=True,
            )
            attn_out = attn_out.view(B, T, self.n_heads, self.head_dim)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            
            attn_out = flash_attn_func(q, k, v, causal=True)

        out = rearrange(attn_out, "B T H C -> B T (H C)")
        return self.output_proj(out), new_cache

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ffn = FeedForwardNetwork(config)
        self.ffn_pre_norm = RMSNorm(config["d_model"], config)
        self.attn = Attention(config)
        self.attn_pre_norm = RMSNorm(config["d_model"], config)
    
    def forward(self, x, cache, cu_seqlens=None, n_docs=None, max_seqlen=None):
        attn_out, new_cache = self.attn(self.attn_pre_norm(x), cache, cu_seqlens, n_docs, max_seqlen)
        x = x + attn_out
        x = x + self.ffn(self.ffn_pre_norm(x))
        return x, new_cache


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Compute derived values
        config["head_dim"] = config["d_model"] // config["n_heads"]
        
        vocab_size = config["vocab_size"]
        d_model = config["d_model"]
        weight_tied = config["weight_tied"]
        n_layers = config["n_layers"]
        
        self.weight_tied = weight_tied
        
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        
        self.blocks = nn.ModuleList([Block(config, i) for i in range(n_layers)])
        
        self.final_norm = RMSNorm(d_model, config)
        
        if not self.weight_tied: self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x, kv_cache=None, cu_seqlens=None, n_docs=None, max_seqlen=None):
        x = self.input_embedding(x)
        
        caches = kv_cache if kv_cache is not None else [None] * len(self.blocks)
        new_kv_cache = []
        for block, cache in zip(self.blocks, caches):
            x, updated_cache = block(x, cache, cu_seqlens, n_docs, max_seqlen)
            new_kv_cache.append(updated_cache)
        
        x = self.final_norm(x)
        logits = F.linear(x, self.input_embedding.weight) if self.weight_tied else self.lm_head(x)
        return logits, new_kv_cache