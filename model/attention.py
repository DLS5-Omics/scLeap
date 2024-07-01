import math

import torch
from torch import nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, bias=None, mask_attn=None):
        attn = torch.matmul(q / self.scale, k.transpose(-1, -2))
        
        if bias is not None:
            attn += bias

        if mask_attn is not None:
            attn += mask_attn
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_key, n_head, dropout):
        super(MultiHeadAttention, self).__init__()
        if d_model % n_head != 0:
            raise ValueError(
                "The hidden size is not a multiple of the number of attention heads"
            )
        self.n_head = n_head
        self.d_k = d_key // n_head
        self.fc_query = nn.Linear(d_model, d_key, bias=False)
        self.fc_key = nn.Linear(d_model, d_key, bias=False)
        self.fc_value = nn.Linear(d_model, d_key, bias=False)

        self.attention = ScaledDotProductAttention(
            scale=self.d_k**0.5, dropout=dropout
        )
        self.fc_out = nn.Linear(d_key, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        """
        x has shape (*, L, C)
        return shape (*, nhead, L, C/nhead)
        """
        new_shape = x.shape[:-1] + (self.n_head, -1)
        x = x.view(*new_shape)
        return x.transpose(-3, -2)

    def forward(self, x, bias=None, mask_attn=None):
        q = self.transpose_for_scores(self.fc_query(x))
        k = self.transpose_for_scores(self.fc_key(x))
        v = self.transpose_for_scores(self.fc_value(x))

        x, attn_weight = self.attention(q, k, v, bias=bias, mask_attn=mask_attn)
        x = x.transpose(-3, -2)
        x = x.reshape(*x.shape[:-2], -1)
        x = self.dropout(self.fc_out(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_key, n_head, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model, d_key=d_key, n_head=n_head, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout
        )

    def forward(self, x, bias, mask_attn):
        x = x + self.attn(self.norm1(x), bias, mask_attn = mask_attn)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layer, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(**kwargs) for _ in range(n_layer)]
        )

    def forward(self, x, mask, mask_attn_embed):
        bias = torch.zeros((x.shape[0], x.shape[1], x.shape[1]), device=x.device)
        bias[mask.unsqueeze(1).expand_as(bias)] = -10000
        bias = bias.unsqueeze(1)
        mask_attn_embed = mask_attn_embed.to(dtype=x.dtype)

        embs = []
        
        for module in self.layers:
            x = module(x, bias, mask_attn_embed)
            embs.append(x[:, 0])
        return x, embs


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_key, n_head, dim_feedforward, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model, d_key=d_key, n_head=n_head, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout
        )

    def forward(self, src, tgt, bias):
        x = torch.cat([src, tgt], dim=-2)
        x = x + self.attn(self.norm1(x), bias=bias)
        x = x + self.ffn(self.norm2(x))
        return x[..., : src.shape[-2], :], x[..., -tgt.shape[-2] :, :]


class TransformerDecoder(nn.Module):
    def __init__(self, n_layer, **kwargs):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(**kwargs) for _ in range(n_layer)]
        )

    def forward(self, src, tgt, mask):
        L = src.shape[1] + tgt.shape[1]
        bias = torch.zeros((src.shape[0], L, src.shape[-2]), device=src.device)
        bias[mask.unsqueeze(1).expand_as(bias)] = -10000
        bias1 = torch.ones((src.shape[-2], tgt.shape[-2]), device=src.device) * -10000
        bias2 = (1.0 - torch.eye(tgt.shape[-2], device=src.device)) * -10000
        bias12 = torch.cat([bias1, bias2], dim=0)
        bias = torch.cat([bias, bias12[None].expand(bias.shape[0], -1, -1)], dim=-1)
        bias = bias.unsqueeze(1)

        embs = []
        for module in self.layers:
            src, tgt = module(src, tgt, bias)
            embs.append(src[:, 0])
        return src, tgt, embs
