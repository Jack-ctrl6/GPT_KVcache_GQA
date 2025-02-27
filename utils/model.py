import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


class GroupQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dims = config.dims
        self.n_groups = config.n_groups
        self.n_heads = config.n_heads

        self.head_dims = self.dims // self.n_heads
        self.groups_dims = self.n_heads // self.n_groups

        self.w_q = nn.Linear(self.dims, self.dims)
        self.w_k = nn.Linear(self.dims, self.n_groups * self.head_dims)
        self.w_v = nn.Linear(self.dims, self.n_groups * self.head_dims)

        self.w_o = nn.Linear(self.dims, self.dims)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('attention_mask',
                             torch.tril(torch.ones(config.max_length, config.max_length)))

        self.kv_cache = {}
        self.setting = False

    def kv_cache_setting(self,setting):
        self.setting = setting

    def expand(self, data):
        bs, seqlen = data.shape[0], data.shape[2]
        data = data[:, :, None, :, :].expand(bs, self.n_groups, self.groups_dims, seqlen,
                                             self.head_dims).contiguous().view(bs, -1, seqlen, self.head_dims)
        return data

    def forward(self, x,is_question=True):
        bs, seqlen, dims = x.shape
        if self.setting is False:   ## 不开kv cache
            q = self.w_q(x)
            k = self.w_k(x)
            v = self.w_v(x)
        else:
            if is_question:
                q = self.w_q(x)
                k = self.w_k(x)
                v = self.w_v(x)

            else:
                last_x = x[:,-1:,:]
                q = self.w_q(last_x)
                k = self.w_k(last_x)
                v = self.w_v(last_x)
                if 'q' in self.kv_cache:
                    q = torch.cat((self.kv_cache['q'],q), dim=1)
                if 'k' in self.kv_cache:
                    k = torch.cat((self.kv_cache['k'],k), dim=1)
                if 'v' in self.kv_cache:
                    v = torch.cat((self.kv_cache['v'],v), dim=1)
            self.kv_cache.update({'q': q.detach(), 'k': k.detach(), 'v': v.detach()})


        q = q.view(bs, -1, self.n_heads, self.head_dims).permute(0, 2, 1,
                                                                               3)  ## shape  bs  n_heads  seqlen head_dims
        k = k.view(bs, -1, self.n_groups, self.head_dims).permute(0, 2, 1,
                                                                                3)  ## shape  bs  n_groups seqlen head_dims
        v = v.view(bs, -1, self.n_groups, self.head_dims).permute(0, 2, 1,
                                                                                3)  ## shape  bs  n_groups seqlen head_dims

        k = self.expand(k)
        v = self.expand(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dims)
        scores = self.dropout(self.softmax(scores)) @ v
        scores = scores.permute(0, 2, 1, 3).contiguous().view(bs, seqlen, self.dims)
        output = self.w_o(scores)
        return output

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dims = config.dims
        self.ffn = nn.Sequential(
            nn.Linear(self.dims, self.dims * 4),
            nn.GELU(),
            nn.Linear(self.dims * 4, self.dims),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.ffn(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layernorm1 = nn.LayerNorm(config.dims)
        self.layernorm2 = nn.LayerNorm(config.dims)

        self.atten = GroupQueryAttention(config)
        self.ffn = FFN(config)

    def open_kv_cache(self):
        self.atten.kv_cache_setting(self.config.setting)

    def close_kv_cache(self):
        self.atten.kv_cache_setting(self.config.setting)

    def forward(self, x,is_question=True):
        x = x + self.atten(self.layernorm1(x),is_question)
        out = x + self.ffn(self.layernorm2(x))
        return out

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.dims)
        self.position_embedding = nn.Embedding(config.max_length, config.dims)

        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layers)]
        )
        self.classfier = nn.Linear(config.dims, config.vocab_size, bias=False)
        self.layernorm = nn.LayerNorm(config.dims)

        self.token_embedding.weight = self.classfier.weight
        self.apply(self._init_weights)

        if config.setting:
            self.open_kvcache()
        else:
            self.close_kvcache()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def open_kvcache(self):
        for layer in self.blocks:
            layer.open_kv_cache()
    def close_kvcache(self):
        for layer in self.blocks:
            layer.close_kv_cache()

    def forward(self, idx,targets=None,is_question=True):
        batch, seqlen = idx.shape
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(seqlen, device=idx.device))

        x = token_emb + pos_emb  # shape  (bs seqlen dims)
        # x = self.blocks(x,is_question)  # shape bs seqlen dims
        for block in self.blocks:
            x = block(x, is_question)
        x = self.layernorm(x)
        logits = self.classfier(x)  ## shape (bs seqlen vocab_size)

        if targets is None:
            loss = None
        else:
            bs, seqlen, vocab_size = logits.shape
            logits = logits.view(bs * seqlen, vocab_size)
            targets = targets.view(bs * seqlen)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx):    ##  max_new_tokens 允许最大输出长度
        is_question = True
        for _ in range(self.config.max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_length else idx[:, -self.config.max_length:]   ## 输入截断，输入【问题+当前回答】
            print(idx_cond.shape)
            logits, _ = self.forward(idx_cond,None,is_question)   ## 输入进model
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)  ## 取概率最大的下一个数
            print(idx_next)
            idx = torch.cat([idx, idx_next], dim=1)   ## 问题 回答拼接
            if idx_next == 50256:   ## 如果输出的是eos_token则直接停止
                break
            if idx.size(1) > self.config.max_length:
                print("超出最大长度，生成结束")
                break
            is_question = False
        return idx