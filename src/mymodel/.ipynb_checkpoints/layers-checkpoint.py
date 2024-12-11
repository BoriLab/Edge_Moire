# %%
import torch
import torch.nn as nn
import random
import math


def gaussian_attention(distances, shift, width):
    width = width.clamp(min=5e-1)
    return torch.exp(-((distances - shift) ** 2) / width)


def laplacian_attention(distances, shift, width):
    width = width.clamp(min=5e-1)
    return torch.exp(-torch.abs(distances - shift) / width)


def cauchy_attention(distances, shift, width):
    width = width.clamp(min=5e-1)
    return 1 / (1 + ((distances - shift) / width) ** 2)


def sigmoid_attention(distances, shift, width):
    width = width.clamp(min=5e-1)
    return 1 / (1 + torch.exp((-distances + shift) / width))


def triangle_attention(distances, shift, width):
    width = width.clamp(min=5e-1)
    return torch.clamp(1 - torch.abs(distances - shift) / width, min=0)


def get_moire_focus(attention_type):
    if attention_type == "gaussian":
        return gaussian_attention
    elif attention_type == "laplacian":
        return laplacian_attention
    elif attention_type == "cauchy":
        return cauchy_attention
    elif attention_type == "sigmoid":
        return sigmoid_attention
    elif attention_type == "triangle":
        return triangle_attention
    else:
        raise ValueError("Invalid attention type")


class GaussianNoise(nn.Module):
    def __init__(self, std=0.01):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            dropout_mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
            return x * dropout_mask
        return x


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            GaussianNoise(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.ffn(x)


class MoireAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        focus,
        initial_shifts,
        initial_widths,
    ):
        super(MoireAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        assert (
            self.head_dim * num_heads == output_dim
        ), "output_dim must be divisible by num_heads"
        self.focus = focus
        self.shifts = nn.Parameter(
            torch.tensor(initial_shifts, dtype=torch.float).view(1, num_heads, 1, 1)
        )
        self.widths = nn.Parameter(
            torch.tensor(initial_widths, dtype=torch.float).view(1, num_heads, 1, 1)
        )
        self.self_loop_W = nn.Parameter(
            torch.tensor(
                [1 / self.head_dim + random.uniform(0, 1) for _ in range(num_heads)],
                dtype=torch.float,
            ).view(1, num_heads, 1, 1),
            requires_grad=False,
        )
        self.qkv_proj = nn.Linear(input_dim, 3 * output_dim)
        self.scale2 = math.sqrt(self.head_dim)

    def forward(self, x, adj, mask):
        batch_size, num_nodes, _ = x.size()
        qkv = (
            self.qkv_proj(x)
            .view(batch_size, num_nodes, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        Q, K, V = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale2
        #1
        # moire_adj = self.focus(adj.unsqueeze(1), self.shifts, self.widths).clamp(
        #     min=1e-6
        # )
        adjusted_scores = scores #+ torch.log(moire_adj)# +~
        I = torch.eye(num_nodes, device=x.device).unsqueeze(0).unsqueeze(0)
        adjusted_scores.add_(I * self.self_loop_W)
        mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)
        adjusted_scores.masked_fill_(~mask_2d.unsqueeze(1), -1e6)
        attention_weights = torch.softmax(adjusted_scores, dim=-1)
        return (
            torch.matmul(attention_weights, V)
            .transpose(1, 2)
            .reshape(batch_size, num_nodes, -1)
        )


class MoireLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        moire_attention_func,
        initial_shift,
        initial_width,
        dropout,
    ):
        super(MoireLayer, self).__init__()
        self.attention = MoireAttention(
            input_dim,
            output_dim,
            num_heads,
            moire_attention_func,
            [initial_shift + random.uniform(-0.5, 0.5) for _ in range(num_heads)],
            [initial_width + random.uniform(-0.5, 0.5) for _ in range(num_heads)],
        )
        self.ffn = FFN(output_dim, output_dim, output_dim, dropout)
        self.projection_for_residual = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj, mask):
        h = self.attention(x, adj, mask)
        h.mul_(mask.unsqueeze(-1))
        h = self.ffn(h)
        h.mul_(mask.unsqueeze(-1))
        x_proj = self.projection_for_residual(x)
        h = h * 0.5 + x_proj * 0.5
        return h
