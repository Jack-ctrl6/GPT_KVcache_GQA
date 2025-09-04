import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
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


from dataclasses import dataclass


@dataclass
class DeepseekConfig:
    hidden_size: int
    num_heads: int
    max_position_embeddings: int
    rope_theta: float
    attention_dropout: float

    q_lora_rank: int
    qk_rope_head_dim: int
    kv_lora_rank: int
    v_head_dim: int
    qk_nope_head_dim: int
    attention_bias: bool

class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.attention_bias = config.attention_bias

        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.q_dowm_proj = nn.Linear(self.hidden_size,self.q_lora_rank,bias=config.attention_bias)
        self.q_dowm_layernorm = RMSNorm(self.q_lora_rank)

        self.q_up_proj = nn.Linear(self.q_lora_rank,self.num_heads*self.q_head_dim,bias=False)


        self.kv_down_proj = nn.Linear(
            self.hidden_size,self.kv_lora_rank+self.qk_rope_head_dim,bias=config.attention_bias
        )
        self.kv_down_layernorm = RMSNorm(self.kv_lora_rank)

        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,self.num_heads*(self.q_head_dim-self.qk_rope_head_dim+self.v_head_dim),bias=False
        )

        self.out_proj = nn.Linear(self.num_heads*self.v_head_dim,self.hidden_size,bias=config.attention_bias)

        self.rotary_emb = DeepseekV2RotaryEmbedding(self.qk_rope_head_dim, self.max_position_embeddings,self.rope_theta)

    def forward(self, hidden_states,attention_mask,position_ids):
        bs,seqlen,_ = hidden_states.shape

        # 1. query proj and split
        q = self.q_up_proj(self.q_dowm_layernorm(self.q_dowm_proj(hidden_states)))
        q = q.view(bs,seqlen,self.num_heads,self.q_head_dim).permute(0,2,1,3)

        q_nope,q_pe = torch.split(q,[self.qk_nope_head_dim,
                                     self.qk_rope_head_dim],dim=-1)

        # 2. key value proj and split
        compressed_kv = self.kv_down_proj(hidden_states)
        compressed_kv,k_pe = torch.split(compressed_kv,[self.kv_lora_rank,
                                                        self.qk_rope_head_dim],dim=-1)
        k_pe = k_pe.view(bs,seqlen,1,self.qk_rope_head_dim).permute(0,2,1,3)
        kv = (
            self.kv_up_proj(self.kv_down_layernorm(compressed_kv))
            .view(bs,seqlen,self.num_heads,self.qk_nope_head_dim+self.v_head_dim).permute(0,2,1,3)
        )
        k_nope,value_states = torch.split(kv,
                                          [self.qk_nope_head_dim,self.v_head_dim],dim=-1)

        # 3. Rope to posiitional-dependent parts
        kv_seq_len = value_states.shape[-2]
        cos,sin = self.rotary_emb(value_states,seq_len = kv_seq_len)
        q_pe,k_pe = apply_rotary_pos_emb(q_pe,k_pe,cos,sin,position_ids)

        # 4. combinee positional-dependeent and independent parts
        querys_states = torch.empty(bs,self.num_heads,seqlen,self.q_head_dim,device=k_pe.device)
        querys_states[:,:,:,:self.qk_nope_head_dim] = q_nope
        querys_states[:,:,:,self.qk_nope_head_dim:] = q_pe

        key_states = torch.empty(bs,self.num_heads,seqlen,self.q_head_dim,device=k_pe.device)
        key_states[:,:,:,:self.qk_nope_head_dim] = k_nope
        key_states[:,:,:,self.qk_nope_head_dim:] = k_pe

        # 5. compute attention scores
        attn_weights = torch.matmul(querys_states,key_states.transpose(2,3))
        attn_weights = attn_weights / math.sqrt(self.q_head_dim)

        if attention_mask is not None:
            attn_weights = torch.masked_fill(attn_weights,attention_mask==0,float('-inf'))

        # 6. softmax and dropout
        attn_weights = F.softmax(attn_weights,dim=-1,dtype=torch.float32).to(querys_states.dtype)
        attn_weights = F.dropout(attn_weights,p=self.attention_dropout,training=self.training)

        # 7. compute attention output
        attn_output = torch.matmul(attn_weights,value_states)
        attn_output = attn_output.permute(0,2,1,3).reshape(bs,seqlen,-1)
        attn_output = self.out_proj(attn_output)

        return attn_output,attn_weights


# 写一个测试函数
def test_mla():
    config = DeepseekConfig(
        hidden_size=7168,
        num_heads=16,
        max_position_embeddings=1024,
        rope_theta=128000,
        attention_dropout=0.1,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        kv_lora_rank=512,

        v_head_dim=128,
        qk_nope_head_dim=128,
        attention_bias=False,
    )

    mla = MLA(config)
    x = torch.randn(2, 1024, 7168)
    position_ids = torch.arange(
        config.max_position_embeddings,
    ).unsqueeze(0).expand(
        x.size(0), -1
    )
    attn_output, attn_weights = mla(x, position_ids=position_ids)
    print(attn_output.shape)
    print(attn_weights.shape)


test_mla()