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
        initial_shifts,
        initial_widths,
        focus,
        edge_attr_dim,  # 추가된 부분
    ):
        super(MoireAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        assert (
            self.head_dim * num_heads == output_dim
        ), "output_dim must be divisible by num_heads"
        self.focus = focus
        self.shifts = nn.Parameter(
            torch.tensor(initial_shifts, dtype=torch.float, requires_grad=True).view(1, num_heads, 1, 1)
        )
        self.widths = nn.Parameter(
            torch.tensor(initial_widths, dtype=torch.float, requires_grad=True).view(1, num_heads, 1, 1)
        )
        self.self_loop_W = nn.Parameter(
            torch.tensor(
                [1 / self.head_dim + random.uniform(0, 1) for _ in range(num_heads)],
                dtype=torch.float,
            ).view(1, num_heads, 1, 1),
            requires_grad=False,
        )
        self.qkv_proj = nn.Linear(input_dim, 3 * output_dim)
        self.edge_ffn = FFN(edge_attr_dim, edge_attr_dim, edge_attr_dim)  # 엣지 속성을 위한 FFN 추가 (2번 적용으로 변경)
        self.scale2 = math.sqrt(self.head_dim)

    def forward(self, x, adj, edge_index, edge_attr, mask):
        batch_size, num_nodes, _ = x.size()
        qkv = (
            self.qkv_proj(x)
            .view(batch_size, num_nodes, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        Q, K, V = qkv[0], qkv[1], qkv[2]  # (batch_size, num_heads, num_nodes, head_dim)
    
        # 엣지 속성 처리
        edge_attr = self.edge_ffn(edge_attr)
        edge_attr = self.edge_ffn(edge_attr)  # (batch_size, num_edges, edge_attr_dim)
    
        # edge_attr의 크기를 (batch_size, num_heads, num_edges)로 맞춤
        # edge_attr_dim != num_heads인 경우에 대비하여 조정
        edge_attr_dim = edge_attr.size(-1)
        if edge_attr_dim == self.num_heads:
            # 크기를 (batch_size, num_heads, num_edges)로 변환
            edge_attr = edge_attr.permute(0, 2, 1)
        elif edge_attr_dim == 1:
            # num_heads 차원으로 확장
            edge_attr = edge_attr.transpose(1, 2)  # (batch_size, 1, num_edges)
            edge_attr = edge_attr.expand(-1, self.num_heads, -1)
        else:
            # edge_attr를 num_heads 차원으로 매핑하기 위해 선형 레이어 사용
            edge_attr = edge_attr.transpose(1, 2)  # (batch_size, edge_attr_dim, num_edges)
            edge_attr = self.edge_mapping(edge_attr)  # (batch_size, num_heads, num_edges)
    
        # 어텐션 스코어 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale2  # (batch_size, num_heads, num_nodes, num_nodes)
        moire_adj = self.focus(adj.unsqueeze(1), self.shifts, self.widths).clamp(min=1e-6)
        adjusted_scores = scores + torch.log(moire_adj)
    
        # 고급 인덱싱을 위한 인덱스 준비
        batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1, 1)
        head_indices = torch.arange(self.num_heads, device=x.device).view(1, -1, 1)
        batch_indices = batch_indices.expand(-1, self.num_heads, edge_index.size(-1))
        head_indices = head_indices.expand(batch_size, -1, edge_index.size(-1))
    
        edge_index_u = edge_index[:, 0, :].unsqueeze(1).expand(-1, self.num_heads, -1)
        edge_index_v = edge_index[:, 1, :].unsqueeze(1).expand(-1, self.num_heads, -1)
    
        # 고급 인덱싱을 사용하여 adjusted_scores를 업데이트
        adjusted_scores[batch_indices, head_indices, edge_index_u, edge_index_v] += edge_attr
    
        # 이후 코드는 동일하게 유지
        I = torch.eye(num_nodes, device=x.device).unsqueeze(0)
        adjusted_scores.add_(I * self.self_loop_W)
        if mask is not None:
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
        shift_min,
        shift_max,
        dropout,
        focus,
        edge_attr_dim,  # 추가된 부분
    ):
        super(MoireLayer, self).__init__()
        shifts = [
            shift_min + random.uniform(0, 1) * (shift_max - shift_min)
            for _ in range(num_heads)
        ]
        widths = [1.3**shift for shift in shifts]
        self.attention = MoireAttention(
            input_dim,
            output_dim,
            num_heads,
            shifts,
            widths,
            focus,
            edge_attr_dim,  # 추가된 부분
        )

        self.ffn = FFN(output_dim, output_dim, output_dim, dropout)
        self.projection_for_residual = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj, edge_index, edge_attr, mask):
        h = self.attention(x, adj, edge_index, edge_attr, mask)
        if mask is not None:
            h.mul_(mask.unsqueeze(-1))
        h = self.ffn(h)
        if mask is not None:
            h.mul_(mask.unsqueeze(-1))
        x_proj = self.projection_for_residual(x)
        h = h * 0.5 + x_proj * 0.5
        return h
