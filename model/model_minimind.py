# =============================================================================
# MiniMind 配置类（与 HuggingFace PretrainedConfig 对齐，便于加载/保存）
# =============================================================================

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    """模型超参数：决定层数、宽度、注意力头、RoPE、可选 MOE 等。"""
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = True,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        # 把其余 kwargs 交给父类（如 from_pretrained 会用到的字段）
        super().__init__(**kwargs)
        # Dropout 比例；训练时在注意力权重与残差投影等处使用
        self.dropout = dropout
        # 序列开始/结束 token，用于生成与特殊位置
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        # 前馈里用的激活函数名（映射到 ACT2FN）
        self.hidden_act = hidden_act
        # 隐藏维度 d_model，也是 embedding 与每层主干的宽度
        self.hidden_size = hidden_size #d_model
        # FFN 中间层宽度；若为 None 会在 FeedForward 里按 hidden 推导并向上取整到 64
        self.intermediate_size = intermediate_size
        # RoPE 预计算频率表的最大长度上限
        self.max_position_embeddings = max_position_embeddings
        # 查询头的数量
        self.num_attention_heads = num_attention_heads
        # Transformer 堆叠层数
        self.num_hidden_layers = num_hidden_layers
        # KV 头数（GQA：可小于 num_attention_heads，减少 K/V 参数量与缓存）
        self.num_key_value_heads = num_key_value_heads
        # 词表大小
        self.vocab_size = vocab_size
        # RMSNorm 分母稳定项，防止除零
        self.rms_norm_eps = rms_norm_eps
        # RoPE 螺旋频率的底数 θ，越大低频分量越多，长上下文行为随之变化
        self.rope_theta = rope_theta
        # 是否在推理时启用 YaRN 式 RoPE 外推缩放
        self.inference_rope_scaling = inference_rope_scaling
        # 启用时：有效上下文 ≈ factor * original_max_position_embeddings（此处 16 * 2048）
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        # 训练主路径上在条件满足时用 PyTorch SDPA（Flash 内核）加速注意力
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        # 是否用混合专家 FFN 替代单层 FeedForward
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个 token 选 top-k 个路由专家
        self.n_routed_experts = n_routed_experts  # 路由专家总数
        self.n_shared_experts = n_shared_experts  # 始终参与的前馈专家数
        self.scoring_func = scoring_func  # 门控 logits 如何变成路由概率（目前仅 softmax）
        self.aux_loss_alpha = aux_loss_alpha  # 负载均衡辅助损失的系数
        self.seq_aux = seq_aux  # True：按序列聚合辅助损失；False：按 batch 聚合
        self.norm_topk_prob = norm_topk_prob  # top-k 权重是否在专家维度上归一化


# =============================================================================
# MiniMind 主体：RMSNorm、RoPE、Attention、FFN/MoE、Block、Model、CausalLM
# =============================================================================

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

@dataclass
class MiniMindCausalLMOutputWithPast(CausalLMOutputWithPast):
    aux_loss: Optional[torch.FloatTensor] = None


class RMSNorm(torch.nn.Module):
    """Root Mean Square LayerNorm：按最后一维 RMS 缩放，无减均值；带可学习缩放 weight。"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 与层归一化中的 gamma 类似，逐通道可学习
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # rms = sqrt(mean(x^2)+eps)；这里返回 x/rms，即归一化后的向量
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 在 float 上算归一化更稳，再 cast 回原始 dtype，最后乘 weight
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """
    在每一层的attention之前,都要对token进行编码
    预计算 RoPE 所需的 cos/sin 表。
    dim: 每个头的维度；end: 要提前算多少个位置；rope_base: 频率底数。
    rope_scaling 非空时使用 YaRN：对部分频带用 ramp 混合原始频率与缩放频率。
    返回与 LLaMA 类实现一致的两张表，最后一维复制一份以匹配 rotate_half 的拼接方式。
    本质位置编码
    """
    # 成对维度上的逆波长 1/θ^(2i/d)；attn_factor（幅度缩放系数）默认为 1，YaRN 时可缩放注意力 logits
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0: #如果需要外推，则使用YaRN算法
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    # 位置 t=0..end-1 与每个频率做外积，得到 [end, dim//2] 的 angle 网格
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # 沿最后一维拼接相同 cos/sin，使长度变为 dim，与 q/k 的最后一维对齐
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """将 RoPE 相位乘子应用到 q、k：半维旋转等价于乘 e^{iθ}。"""

    def rotate_half(x):
        # 后一半取负与前一半交换，对应复数域乘以 i 的块矩阵形式
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # cos/sin 在指定维上 broadcast 到 head 维
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    GQA：把 KV 头在 head 维上重复 n_rep 次，使 K/V 头数与 Q 头数一致以便矩阵乘。
    等价于 torch.repeat_interleave(x, dim=2, repeats=n_rep)。
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].
        expand(bs, slen, num_key_value_heads, n_rep, head_dim).
        reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """多头自注意力：可选 GQA、RoPE、KV cache、Flash SDPA 或显式因果 mask 路径。
    传统注意力中单独的KV和Q配对太占用显存，GQA将KV头数与Q头数对齐，减少显存占用。
    工程验证4个Q头配1个KV头效果最佳。"""

    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # 未指定 KV 头数时退化为 MHA（KV 头= Q 头）
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        # 每个 KV 头要广播几次才能对齐 Q 
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        
        #加速计算
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape

        # 1. 线性投影得到 Q/K/V，并 reshape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 2. 加入 RoPE 位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # ==========================================================
        # 【管家的核心手术区：完美兼容 HuggingFace 的缓存格式】
        
        # 将 K 和 V 转换为 HF 要求的标准形状: [bs, heads, seq_len, head_dim]
        xk_hf = xk.transpose(1, 2)
        xv_hf = xv.transpose(1, 2)

        # 此时的 seq_len 维度已经变成了 dim=2！在这里拼接历史记忆绝不会报错！
        if past_key_value is not None:
            xk_hf = torch.cat([past_key_value[0], xk_hf], dim=2)
            xv_hf = torch.cat([past_key_value[1], xv_hf], dim=2)
        
        # 存下符合 HF 标准的缓存供下一步使用
        past_kv = (xk_hf, xv_hf) if use_cache else None

        # 为了兼容原作者那个娇贵的 repeat_kv 函数，我们再偷偷转回去
        # 变回 [bs, 总seq_len, heads, head_dim]
        xk = xk_hf.transpose(1, 2)
        xv = xv_hf.transpose(1, 2)
        # ==========================================================

        # 3. 准备 Attention 计算 (再次转置，并展开 KV 头)
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # 4. Attention 核心计算逻辑
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 因果掩码：只用管新输入的 seq_len 即可，不需要管历史长度
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 5. 恢复形状并投影输出
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        
        return output, past_kv


class FeedForward(nn.Module): #FFN层，升维 -> 激活 -> 降维
    # 在每个token位置上做非线性特征变换

    """SwiGLU 风格 FFN：down(act(gate)·up)，与 LLaMA 类结构一致。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None: 
            # 经典 8/3 * d 规则，再对齐到 64 的倍数便于内核
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        # 门控，然后送入SwiGLU激活函数
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # 降维
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # 升维
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # dropout
        self.dropout = nn.Dropout(config.dropout)
        # 激活函数
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # SwiGLU: down( σ(gate·x) ⊙ up·x )
        # x*gate*up(x) -> down -> dropout 
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module): #混合专家门控，负责选择专家
    """路由门控：线性打分 → softmax → top-k，并返回辅助负载损失（训练时）。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config  # 保存完整配置，供调试/扩展使用（不直接参与计算）。
        self.top_k = config.num_experts_per_tok  # 每个 token 最多选择的专家数量（top-k）。
        self.n_routed_experts = config.n_routed_experts  # 参与路由的“可选专家”总数（不含 shared experts）。

        self.scoring_func = config.scoring_func  # 路由分数到概率的映射方式；当前 forward 仅支持 'softmax'。
        self.alpha = config.aux_loss_alpha  # 辅助损失系数：用于负载均衡，抑制路由塌缩到少数专家；alpha<=0 则关闭。
        self.seq_aux = config.seq_aux  # 负载均衡 aux_loss 采用的计算方式：是否按“序列维”聚合统计。

        self.norm_topk_prob = config.norm_topk_prob  # top-k 概率是否再归一化：保证每个 token 选中的 top-k 概率和为 1。
        self.gating_dim = config.hidden_size  # gating 的输入维度：与 Transformer hidden_size 对齐。
        # 每个专家一个权重向量；对 token hidden_states 做点积（线性层）得到该专家的 logits。
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 使用 Kaiming 初始化（fan-in）以获得较稳定的门控/路由 logits 分布。
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # hidden_states: [bsz, seq_len, hidden_size]
        bsz, seq_len, h = hidden_states.shape
        # 展平 token 维：把所有 token 视为 batch 里的样本，一次性算所有 token 的路由 logits。
        hidden_states = hidden_states.view(-1, h)  # [bsz*seq_len, hidden_size]
        # logits: [bsz*seq_len, n_routed_experts]，表示“每个 token 对每个专家”的打分（未归一）。
        logits = F.linear(hidden_states, self.weight, None) 
        if self.scoring_func == 'softmax':
            # scores: 概率分布（对每个 token 的专家维做 softmax）
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # topk_idx/topk_weight 都是每个 token 的路由结果：
        # topk_idx: [bsz*seq_len, top_k] 选中的专家编号
        # topk_weight: [bsz*seq_len, top_k] 对应权重（来自 scores）
        # 从每个 token 的专家概率里，挑出 top_k 个最大概率的专家

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False) 
        # 防御性边界裁剪：确保 topk_idx 一定落在 [0, n_routed_experts-1]，
        # 避免后续 gather/scatter 的 CUDA 越界 device-side assert。
        # 正常情况下 topk 的结果本就不会越界；一旦出现越界，这里能把训练从硬崩中拉回来。
        topk_idx = topk_idx.clamp(0, self.n_routed_experts - 1)
        
        """计算整个batch的辅助损失"""

        if self.top_k > 1 and self.norm_topk_prob:
            # 被选中的 top-k 专家权重再归一：让每个 token 的 top-k 权重和为 1
            #之前的归一化是每个token的专家概率归一化，现在是对top_k个专家概率归一化
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        # 训练时，计算辅助损失，alpha是对aux_loss的惩罚系数
        if self.training and self.alpha > 0.0:
            # 负载均衡（aux_loss）：惩罚路由分布偏离均匀，避免塌缩到少数专家
            scores_for_aux = scores #保存原有的专家概率用于计算辅助损失
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1) # [bsz, seq_len*top_k] 统计这一个 batch 里，每个专家被路由了多少次。

            #序列级别的辅助损失
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # ce: [bsz, n_routed_experts]，表示每个专家被选中的次数。
                # 原实现用 scatter_add 统计计数，在部分 CUDA 情况下会触发 out-of-bounds。
                # 这里改用 bincount（索引安全）计算专家计数。
                ce = torch.zeros(
                    bsz,
                    self.n_routed_experts,
                    device=hidden_states.device,
                    dtype=scores_for_aux.dtype,
                )
                # topk_idx_for_aux_loss: [bsz, seq_len*aux_topk]
                denom = (seq_len * aux_topk) / self.n_routed_experts
                for bi in range(bsz):
                    ce[bi] = torch.bincount(
                        topk_idx_for_aux_loss[bi].to(torch.int64),
                        minlength=self.n_routed_experts,
                    ).to(ce.dtype)
                ce = ce / denom  # 与原 scatter_add 的 div_ 含义一致：得到归一化后的使用率

                # 每个专家的使用率 * 每个专家的平均分数（偏好）= 每个专家的辅助损失
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 另一种统计方式：不按序列维聚合，而是对 batch 内统计整体均值。
                # 避免 F.one_hot 在 CUDA 上走 scatter/gather 路径引发 index 越界。
                flat_idx = topk_idx_for_aux_loss.reshape(-1).to(torch.int64)
                flat_idx = flat_idx.clamp(0, self.n_routed_experts - 1)
                counts = torch.bincount(flat_idx, minlength=self.n_routed_experts).to(scores_for_aux.dtype)
                ce = counts / flat_idx.numel()
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            # eval 或 alpha<=0：不启用 aux_loss，返回 0。
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """混合专家 FFN：门控选专家，训练时按专家分批前向；推理时用 moe_infer 分组计算。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 门控为每个 token 选出 top-k 专家索引与权重，并计算 aux_loss
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])  # [N_tokens, H]
        top_k = self.config.num_experts_per_tok
        flat_topk_idx = topk_idx.reshape(-1)  # [N_tokens*top_k]
        flat_topk_weight = topk_weight.reshape(-1)  # [N_tokens*top_k]

        # 稳定实现：dense experts + gather 选择输出。
        # 这样避免了训练时依赖 index_add / scatter 的汇聚路径，从而规避
        # ScatterGatherKernel 触发 “index out of bounds” 的不稳定问题。
        n_tokens = x.shape[0]
        topk_idx_2d = flat_topk_idx.view(n_tokens, top_k)          # [N_tokens, top_k]
        topk_weight_2d = flat_topk_weight.view(n_tokens, top_k)  # [N_tokens, top_k]
        # 再做一次防御性裁剪，避免由于浮点/未定义状态导致 gather 索引越界。
        n_experts = len(self.experts)
        if topk_idx_2d.dtype != torch.long:
            topk_idx_2d = topk_idx_2d.to(torch.long)
        topk_idx_2d = topk_idx_2d.clamp(0, n_experts - 1)

        # [n_experts, N_tokens, H]
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=0)
        # [N_tokens, n_experts, H]
        expert_outs = expert_outs.permute(1, 0, 2)

        # 稳定实现：不用 F.one_hot（CUDA 上可能走 scatter/gather 内核）。
        # 对每个 expert 用 mask 选择 top-k 中落在该 expert 的 token，并累加权重。
        # topk_idx_2d: [N_tokens, top_k]
        # topk_weight_2d: [N_tokens, top_k]
        y_flat = torch.zeros(
            n_tokens,
            expert_outs.size(-1),
            device=expert_outs.device,
            dtype=expert_outs.dtype,
        )
        for e in range(n_experts):
            mask_e = topk_idx_2d == e  # [N_tokens, top_k]
            # w_e: [N_tokens]，该 token 选择了 expert e 的权重之和
            w_e = (mask_e.to(topk_weight_2d.dtype) * topk_weight_2d).sum(dim=1)
            y_flat = y_flat + w_e.unsqueeze(-1).to(y_flat.dtype) * expert_outs[:, e, :]
        y = y_flat.view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """推理使用 dense experts + gather 选择输出（避免 scatter/index_add 边界问题）。"""
        top_k = self.config.num_experts_per_tok
        n_tokens = x.shape[0]

        topk_idx_2d = flat_expert_indices.view(n_tokens, top_k)           # [N_tokens, top_k]
        topk_weight_2d = flat_expert_weights.view(n_tokens, top_k)      # [N_tokens, top_k]
        n_experts = len(self.experts)
        if topk_idx_2d.dtype != torch.long:
            topk_idx_2d = topk_idx_2d.to(torch.long)
        topk_idx_2d = topk_idx_2d.clamp(0, n_experts - 1)

        # [n_experts, N_tokens, H] -> [N_tokens, n_experts, H]
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=0).permute(1, 0, 2)
        n_experts = expert_outs.size(1)

        # 推理也走 mask 累加，避免 one_hot / scatter/gather 相关不稳定。
        y_flat = torch.zeros(
            n_tokens,
            expert_outs.size(-1),
            device=expert_outs.device,
            dtype=expert_outs.dtype,
        )
        for e in range(n_experts):
            mask_e = topk_idx_2d == e  # [N_tokens, top_k]
            w_e = (mask_e.to(topk_weight_2d.dtype) * topk_weight_2d).sum(dim=1)
            y_flat = y_flat + w_e.unsqueeze(-1).to(y_flat.dtype) * expert_outs[:, e, :]

        return y_flat  # [N_tokens, H]


class MiniMindBlock(nn.Module):
    """单层 Transformer：Pre-LN 注意力子层 + 残差，再接 Pre-LN MLP/MoE + 残差。"""

    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states # 残差
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), 
            position_embeddings, #位置编码
            past_key_value, #历史KV
            use_cache, #是否使用cache
            attention_mask #掩码
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """Embedding + N 个 MiniMindBlock + 最终 RMSNorm；缓存每层 KV；聚合 MoE aux_loss。"""

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # token_id -> token向量表示
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # 堆叠多个MiniMindBlock
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        """预计算RoPE所需的cos/sin表"""
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        # 非持久 buffer：不写入 checkpoint，随 config 在 __init__ 重建
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        # 兼容部分 HF 包装类型：若 past_key_values 带 layers 属性则按无缓存处理
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        # 已缓存前缀长度，用于从 freqs 表 correct 截取当前步 RoPE
        #计算当前步RoPE的起始位置，原有的token已经编码过，以缓存的KV形式存在
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids)) #随机dropout，embedding层
        #位置编码：从freqs_cos和freqs_sin中截取当前步RoPE所需的cos/sin表
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        """每一层都用“当前的 hidden_states + 这一层自己的过去 KV + 当前 token 的位置编码”
        算出新的 hidden_states，并返回这一层新的 KV cache（供下一次生成用）"""

        """进入block层"""
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        """进入norm层"""
        hidden_states = self.norm(hidden_states)

        # 只对 MoE 层累加辅助损失；无 MoE 时为 0
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin): #模型标准化
    """因果语言模型：主干 + lm_head；与 embed 共享词表权重（tie weights）。"""
    """MiniMindForCausalLM 负责“把 token id 走完 Transformer，
    然后用 lm_head 产生词表 logits，并在训练时用 labels 计算下一词交叉熵 loss”，
    同时把 MoE 的 aux_loss 一起挂在输出上。"""

    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # 输入 embedding 与输出投影共用一个矩阵，减少参数并常带来更好收敛

        # str         ->      token_id    ->  hidden_states
        #      tokenizer_encoder ->      embedding 

        # hidden_states -> logits     ->    token_id      ->   str
        #            lm_head     GenerationMixin     tokenizer_decoder

        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        # 只算最后 logits_to_keep 个位置（例如生成时只关心最新 token），减少算力
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # 标准 CLM：预测下一 token，logits 与 labels 错一位对齐
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        return MiniMindCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            aux_loss=aux_loss,
        )
