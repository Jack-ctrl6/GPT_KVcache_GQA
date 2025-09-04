import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from dataclasses import dataclass


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # 较小索引位置对应较低频率
        # 较大的索引位置有较高的频率

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

@dataclass
class MLAConfig:
    dims: int
    q_lora_rank: int

    n_heads: int
    q_head_dims: int  ##   qk_nope_head_dims + qk_rope_head_dim = q_head_dims

    qk_nope_head_dims: int
    qk_rope_head_dims: int

    kv_lora_rank: int
    v_head_dims: int

    max_position_embeddings: int
    rope_theta: float
    attention_dropout: float


class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.q_down_proj = nn.Linear(config.dims, config.q_lora_rank, bias=False)
        self.q_up_proj = nn.Linear(config.q_lora_rank, config.n_heads * config.q_head_dims, bias=False)
        self.q_down_rmsnorm = DeepseekV2RMSNorm(config.q_lora_rank, eps=1e-6)

        self.kv_down_proj = nn.Linear(config.dims, config.kv_lora_rank + config.qk_rope_head_dims, bias=False)
        self.kv_up_proj = nn.Linear(config.kv_lora_rank,
                                    config.n_heads * (config.qk_nope_head_dims + config.v_head_dims), bias=False)
        self.kv_down_rmsnorm = DeepseekV2RMSNorm(config.kv_lora_rank, eps=1e-6)

        # 初始化 rope 的参数
        self.rotary_emb = DeepseekV2RotaryEmbedding(
            self.config.qk_rope_head_dims,
            self.config.max_position_embeddings,
            self.config.rope_theta,
        )

        self.o_proj = nn.Linear(
            config.n_heads * config.v_head_dims,
            config.dims,
            bias=False,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bs, slen, _ = x.shape

        q = self.q_up_proj(self.q_down_rmsnorm(self.q_down_proj(x)))  # shape bs  n-heads slen q-head-dims
        q = q.view(bs, slen, self.config.n_heads, self.config.q_head_dims).permute(0, 2, 1, 3)
        q_rope, q_nope = torch.split(q, [self.config.qk_rope_head_dims, self.config.qk_nope_head_dims], dim=-1)

        compressed_kv = self.kv_down_proj(x)  # shape bs slen kv_lora_rank + qk_rope_head_dims
        compressed_kv, k_rope = torch.split(compressed_kv, [self.config.kv_lora_rank, self.config.qk_rope_head_dims],
                                            dim=-1)

        k_rope = k_rope.view(bs, slen, 1, self.config.qk_rope_head_dims).permute(0, 2, 1, 3)

        kv = self.kv_up_proj(
            self.kv_down_rmsnorm(compressed_kv))  ## shape bs slen n-heads*(qk-nope-head-dims + v-head-dims)
        kv = kv.view(bs, slen, self.config.n_heads, self.config.qk_nope_head_dims + self.config.v_head_dims).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)  ##shape bs n-heads slen, qk-nope-head-dism+v-head-dims
        k_nope, value = torch.split(kv, [self.config.qk_nope_head_dims, self.config.v_head_dims], dim=-1)

        # 3. apply rope
        kv_seqlen = value.shape[-2]

        cos, sin = self.rotary_emb(value, seq_len=kv_seqlen)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin, position_ids=position_ids)

        # 4. combine
        query = torch.cat((q_nope, q_rope), dim=-1)
        # print(k_nope.shape, k_rope.shape)
        # key = torch.cat((k_nope, k_rope), dim=-1)

        key = torch.empty(bs,self.config.n_heads,slen,self.config.q_head_dims,device=k_rope.device)
        key[:,:,:,:self.config.qk_nope_head_dims] = k_nope
        key[:,:,:,self.config.qk_nope_head_dims:] = k_rope

        # 5. compute attention score
        attn_weights = torch.matmul(query, key.transpose(2, 3))
        attn_weights = attn_weights / math.sqrt(self.config.q_head_dims)

        if mask is not None:
            attn_weights = torch.masked_fill(
                attn_weights,
                mask == 0,
                float("-inf"),
            )

        # 6. Softmax and dropout
        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(
            attn_weights, p=self.config.attention_dropout, training=self.training)

        # 7. Compute attention output
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).reshape(bs, slen, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


if __name__ == '__main__':
    config = MLAConfig(
        dims=8,
        q_lora_rank=8,

        n_heads=2,
        q_head_dims=8,  ##   qk_nope_head_dims + qk_rope_head_dim = q_head_dims

        qk_nope_head_dims=4,
        qk_rope_head_dims=4,

        kv_lora_rank=4,
        v_head_dims=4,

        max_position_embeddings=4,
        rope_theta=128000,
        attention_dropout=0.1,
    )
    mla = MLA(config)
    mla.to('cuda')
    x = torch.randn(2, 4, 8).cuda()
    position_ids = torch.arange(
        config.max_position_embeddings,
    ).unsqueeze(0).expand(
        x.size(0), -1
    ).cuda()
    attn_output, attn_weights = mla(x, position_ids=position_ids)
    print(attn_output.shape)
    print(attn_weights.shape)