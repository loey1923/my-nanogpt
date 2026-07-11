import torch
import torch.nn as nn
import torch.nn.functional as F


# ====== Masked Multi-Head Attention ======
class MultiHeadAttention(nn.Module):
    """
    多头因果自注意力
    """
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0, "嵌入维度应当可被注意力头数整除"
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3*n_embd)
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, input):
        """
        input 为多头注意力层输入，形状为(B, T, C)
        输出为注意力分数， 形状为(B, T, C)
        """
        B, T, _ = input.shape # C == n_embd
        qkv = self.c_attn(input) # (B, T, 3*n_embd)
        q, k, v = qkv.split(self.n_embd, dim = -1) # (B, T, n_embd)
        q = q.view(B, T, self.n_head, -1).transpose(1, 2) # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)

        attn_score = q @ k.transpose(-2, -1) / (self.head_size ** 0.5) # (B, n_head, T, T)
        attn_score = attn_score.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf')) # (B, n_head, T, T)
        attn_weight = F.softmax(attn_score, dim = -1)
        attn = attn_weight @ v # (B, n_head, T, head_size)

        attn = attn.transpose(1, 2).contiguous().view(B, T, -1) # (B, T, n_embd)
        output = self.c_proj(attn)

        return output


# ====== FFN ======
class FFN(nn.Module):
    """
    一层Feed Forward 神经网络
    包含一个线性层（维度扩展到四倍），一个GeLU激活函数， 一个线性层（恢复原维度）
    """
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4*n_embd)
        self.c_proj = nn.Linear(4*n_embd, n_embd)

    def forward(self, input):
        x = self.c_fc(input)
        x = F.gelu(x, approximate = 'tanh') # 使用tanh近似版本
        output = self.c_proj(x)
        return output
    

# ====== TransformerBlock ======
class TransformerBlock(nn.Module):
    """
    一个TransformerBlock(decoder only)
    流程包括：LN -> Multi-Head Attention -> 残差连接 -> LN -> FFN -> 残差连接
    """
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = FFN(n_embd)

    def forward(self, input):
        x = self.attn(self.ln_1(input)) + input # 残差连接应当跨过整个子层(norm + attn/ffn)
        output = self.mlp(self.ln_2(x)) + x
        return output


# ====== GPT Model ======
class GPT(nn.Module):
    """
    一个小GPT模型

    输入：(B, T)
    输出: logits,(B, T, vocab_size)与loss, 标量
    """
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            h = nn.ModuleList([TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd)
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias = False)
        self.transformer.wte.weight = self.lm_head.weight # weight tying

    def forward(self, idx, targets = None):
        """
        输入idx, 形状(B, T)
        输出logits，形状(B, T, vocab_size)与loss
        """
        B, T = idx.shape
        position = torch.arange(T, device = idx.device)
        x = self.transformer.wpe(position) + self.transformer.wte(idx) # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(-1))
        else:
            loss = None

        return logits, loss