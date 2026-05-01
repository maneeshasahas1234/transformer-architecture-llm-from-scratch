import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import GPTConfig

class Embeddings(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.seq_len = cfg.seq_len

    def forward(self, x):
        B, T = x.shape
        tok = self.tok_emb(x)

        # Position embeddings
        pos = torch.arange(T, device=x.device)
        pos = self.pos_emb(pos)

        # Add together (broadcasting: pos goes from [T,d] to [1,T,d])
        out = self.dropout(tok + pos)   # [B, T, d_model]
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        assert cfg.d_model % cfg.n_heads == 0

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.seq_len = cfg.seq_len

        self.W_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.out_dropout = nn.Dropout(cfg.dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg.seq_len, cfg.seq_len), diagonal=1).bool()
        )

    def forward(self, x):
        B, T, C = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            self.mask[:T, :T],
            torch.finfo(scores.dtype).min
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = weights @ V
        out = out.transpose(1, 2).reshape(B, T, C)

        out = self.out_proj(out)
        out = self.out_dropout(out)

        return out



class FFN(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            # W1: expand d_model → d_ff (4× expansion)
            nn.Linear(cfg.d_model, cfg.d_ff),
            # GELU activation
            nn.GELU(),
            # W2: compress d_ff → d_model
            nn.Linear(cfg.d_ff, cfg.d_model),
            # Dropout
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        # x: [B, T, d_model]
        return self.net(x)   # [B, T, d_model]


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ffn  = FFN(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x   # [B, T, d_model]


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.embed      = Embeddings(cfg)
        self.blocks     = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.out_head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)


    def forward(self, x):
        x = self.embed(x)       # [B, T] → [B, T, d_model]
        x = self.blocks(x)      # [B, T, d_model]
        x = self.final_norm(x)  # [B, T, d_model]
        return self.out_head(x)

