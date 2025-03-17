import math 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import LLMConfig
from typing import List,Tuple,Optional
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps  # 防止除零的小常数
        self.weight = nn.Parameter(torch.ones(dim))  # 学习参数，初始全1

    def forward(self, x):
        # 计算x每个元素的均方根归一化，再乘以学习参数
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)

# 预先计算位置编码（旋转位置编码，RoPE）的复数形式

def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1000000.0):

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):

    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: LLMConfig):
        super().__init__()
        # 未指定n_kv_heads 则变为多头注意力
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_heads = args.n_heads
        self.n_group = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads # 每个头的维度

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias = False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias = False)

        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias = False)
        #注意力和残差的dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # 掩码
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal = 1)
        self.register_buffer("mask", mask, persistent = False)

    def forward(self, 
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache = False):
        batch_size, seq_len, dim = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        #调整形状以适应多头计算
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)
        #若有kv缓存 则拼接当前kv
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim = 1)
            xv = torch.cat([past_key_value[1], xv], dim = 1)
        kv = (xk, xv) if use_cache else None
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_group).transpose(1, 2),
            repeat_kv(xv, self.n_group).transpose(1, 2)
        )
        # 计算score
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 加入掩码
        scores += self.mask[:, :, :seq_len, :seq_len]
        # softmax
        scores = F.softmax(scores.float(), dim = -1).type_as(xq)
        scores = self.attn_dropout(scores)
        # 计算输出
        output = scores @ xv
        # 恢复形状并进行输出投影、加残差
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, kv


class FeedForward(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        #两个线性层以及一个SwiGLU辅助线性层
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias = False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias = False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias = False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: LLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads

        #注意力子层
        self.attention = Attention(config)
        self.layer_id = layer_id

        #两个norm 一个注意力前 一个ffn前
        self.attention_norm = RMSNorm(config.dim, eps = config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps = config.norm_eps)

        #ffn
        self.feed_forward = FeedForward(config)
    
    def forward(self, x, pos_cis, past_key_value = None, use_cache = False):
        # 归一化 和 attention
        x_attn, past_kv = self.attention(
            self.attention_norm(x), # preNorm 防止梯度消失/爆炸
            pos_cis, 
            past_key_value = past_key_value,
            use_cache = use_cache
        )
        # 残差
        h = x + x_attn
        # 经过ffn再加残差
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_kv


class Llama(PreTrainedModel):
    config_class = LLMConfig

    def __init__(self, params: LLMConfig = None):
        self.params = params or LLMConfig()
        super().__init__(self.params)

        # 词表大小和层数
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        # 词嵌入层
        self.token_embedding = nn.Embedding(self.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        #多层transformer
        self.layers = nn.ModuleList([TransformerBlock(i, params) for i in range(self.n_layers)])
        self.norm = RMSNorm(params.dim, eps = params.norm_eps)
        # 输出线性层
        self.output = nn.Linear(params.dim, params.vocab_size, bias = False)
        # 共享权重
        self.token_embedding.weight = self.output.weight
        #预先计算位置编码 存储为buffer 不参与训练
        self.register_buffer("pos_cis", precompute_pos_cis(dim = params.dim // params.n_heads, theta = params.rope_theta), persistent = False)
        self.OUT = CausalLMOutputWithPast()
    
    def forward(self,
                input_ids:Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args
                ):
        # 若没有传入kv cache 则置为none
        past_key_values = past_key_values or [None] * len(self.layers)

        start_pos = args.get('start_pos', 0)

        #词嵌入 + dropout  h为在前向传播中向前传递的数据
        h = self.dropout(self.token_embedding(input_ids))
        #根据输入序列长度获得位置编码
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]
        past_kvs = []

        # 逐层传递数据 并收集kv缓存
        for i, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value = past_key_values[i],
                use_cache = use_cache
            )
            past_kvs.append(past_kv)

        # 最后经过归一化和输出线性层得到logits (batch_size, sequence_length, vocab_size)
        logits = self.output(self.norm(h))
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('past_key_values', past_kvs)

        return self.OUT

    @torch.inference_mode()
     # 生成函数：支持流式生成与一次性生成
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
            
        return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

            
     # 内部流式生成函数
    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            # 首次调用或未使用缓存时，传入整个序列
            if first_seq or not use_cache:
                # 调用forawrd
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
            else:
                # 仅传入最后一个token，同时更新start_pos
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                        start_pos=input_ids.shape[1] - 1, **args)
            # 取出最后一步的logits及更新后的KV缓存
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            # 对已经生成的token进行惩罚，防止重复生成
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            # 温度缩放
            logits /= (temperature + 1e-9)
            # 如果设置了top_p采样，则进行核采样处理
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            # 根据采样后的概率分布选取下一个token
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            # 将新token拼接到已有序列上
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            # 生成器返回新生成部分
            yield input_ids[:, start:]
            # 若生成的token为结束符，则停止生成
            if input_ids_next.item() == eos_token_id:
                break




