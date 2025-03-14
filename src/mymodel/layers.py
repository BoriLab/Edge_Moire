import torch
import torch.nn as nn
import random
import math

################################################################################
# 1. 모아레 어텐션에서 사용할 집중(focus) 함수들
################################################################################

def gaussian_attention(distances, shift, width):
    # width가 0이 되지 않도록 clamp
    width = width.clamp(min=5e-1)
    return torch.exp(-((distances - shift) ** 2) / (width ** 2))

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

################################################################################
# 2. 노이즈, 드롭아웃, FFN 모듈
################################################################################

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

################################################################################
# 3. Edge Wave Superposition Attention 레이어 (valid_edge_mask 추가)
################################################################################

class EdgeWaveAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        initial_shifts,
        initial_widths,
        focus,         # 예: get_moire_focus("gaussian")
        edge_attr_dim, # 엣지 속성 차원
    ):
        super(EdgeWaveAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        assert self.head_dim * num_heads == output_dim, "output_dim must be divisible by num_heads"

        # 기본 shift, width 파라미터 (shape: (1, num_heads, 1, 1))
        self.focus = focus
        self.shifts = nn.Parameter(
            torch.tensor(initial_shifts, dtype=torch.float).view(1, num_heads, 1, 1)
        )
        self.widths = nn.Parameter(
            torch.tensor(initial_widths, dtype=torch.float).view(1, num_heads, 1, 1)
        )

        self.self_loop_W = nn.Parameter(
            torch.tensor([1 / self.head_dim + random.uniform(0, 1) for _ in range(num_heads)],
                         dtype=torch.float).view(1, num_heads, 1, 1),
            requires_grad=False,
        )

        self.qkv_proj = nn.Linear(input_dim, 3 * output_dim)

        self.edge_ffn = FFN(edge_attr_dim, edge_attr_dim, edge_attr_dim)
        self.delta_shift_mlp = nn.Linear(edge_attr_dim, num_heads)
        self.delta_width_mlp = nn.Linear(edge_attr_dim, num_heads)

        self.scale2 = math.sqrt(self.head_dim)

        self.edge_mapping = nn.Linear(edge_attr_dim, num_heads)

    def forward(self, x, adj, edge_index, edge_attr, mask, valid_edge_mask):
        batch_size, num_nodes, _ = x.size()

        # Q, K, V 계산
        qkv = (
            self.qkv_proj(x)
            .view(batch_size, num_nodes, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        Q, K, V = qkv[0], qkv[1], qkv[2]  # (B, H, num_nodes, head_dim)

        # 엣지 속성 FFN 두 번 적용
        edge_features = self.edge_ffn(edge_attr)
        edge_features = self.edge_ffn(edge_features)  # (B, num_edges, edge_attr_dim)

        # edge_mapping 적용: (B, num_edges, edge_attr_dim) -> (B, num_edges, num_heads)
        edge_features = self.edge_mapping(edge_features)  # (B, num_edges, num_heads)
        edge_features = edge_features.permute(0, 2, 1)      # (B, H, num_edges)

        # delta shift, delta width 계산: (B, num_edges, edge_attr_dim) -> (B, H, num_edges)
        delta_shift = self.delta_shift_mlp(edge_attr).permute(0, 2, 1)
        delta_width = self.delta_width_mlp(edge_attr).permute(0, 2, 1)

        # 기본 어텐션 score 계산: (B, H, num_nodes, num_nodes)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale2

        # 기본 모아레 포커스 값 계산: (B, H, num_nodes, num_nodes)
        baseline_focus = self.focus(adj.unsqueeze(1), self.shifts, self.widths).clamp(min=1e-6)
        # print("adj shape:", adj.shape)                         # [B, num_nodes, num_nodes]
        # print("self.shifts shape:", self.shifts.shape)         # [1, H, 1, 1]
        # print("self.widths shape:", self.widths.shape)         # [1, H, 1, 1]

        # edge_index: (B, 2, num_edges) -> (B, H, num_edges)
        edge_index_u = edge_index[:, 0, :].unsqueeze(1).expand(batch_size, self.num_heads, -1)
        edge_index_v = edge_index[:, 1, :].unsqueeze(1).expand(batch_size, self.num_heads, -1)
        # print("edge_index shape:", edge_index.shape)
        # print("edge_index_u shape:", edge_index_u.shape)
        # print("edge_index_v shape:", edge_index_v.shape)

        # adj 확장: (B, H, num_nodes, num_nodes)
        adj_expanded = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        temp = torch.gather(adj_expanded, 2, edge_index_u.unsqueeze(-1).expand(-1, -1, -1, num_nodes))
        edge_distances = torch.gather(temp, 3, edge_index_v.unsqueeze(-1))  # (B, H, num_edges, 1)
        edge_distances = edge_distances.squeeze(-1)  # (B, H, num_edges)
        # print("edge_distances shape:", edge_distances.shape)  # Expected: [B, H, num_edges]

        # valid_edge_mask 처리: 전달된 valid_edge_mask가 2D 또는 3D일 수 있음
        if valid_edge_mask.dim() == 2:
            valid_edge_mask = valid_edge_mask.unsqueeze(1).expand(batch_size, self.num_heads, -1)
        else:
            valid_edge_mask = valid_edge_mask[:, 0, :].unsqueeze(1).expand(batch_size, self.num_heads, -1)

        # modulated shift 및 width 계산: [B, H, num_edges]
        modulated_shifts = self.shifts.view(1, self.num_heads, 1).expand(batch_size, self.num_heads, edge_index_u.size(-1))
        modulated_widths = self.widths.view(1, self.num_heads, 1).expand(batch_size, self.num_heads, edge_index_u.size(-1))
        # print("modulated_shifts shape (after view & expand):", modulated_shifts.shape)
        # print("delta_shift shape:", delta_shift.shape)
        # print("modulated_widths shape (after view & expand):", modulated_widths.shape)
        # print("delta_width shape:", delta_width.shape)
        
        modulated_shifts = modulated_shifts + delta_shift
        modulated_widths = modulated_widths + delta_width
        # print("modulated_shifts shape (after addition):", modulated_shifts.shape)
        # print("modulated_widths shape (after addition):", modulated_widths.shape)
        
        # 모든 텐서를 [B, H, num_edges, 1]로 맞추기
        modulated_shifts = modulated_shifts.unsqueeze(-1)
        modulated_widths = modulated_widths.unsqueeze(-1)
        edge_distances = edge_distances.unsqueeze(-1)
        # print("modulated_shifts after unsqueeze:", modulated_shifts.shape)
        # print("modulated_widths after unsqueeze:", modulated_widths.shape)
        # print("edge_distances after unsqueeze:", edge_distances.shape)
        
        modulated_focus = self.focus(edge_distances, modulated_shifts, modulated_widths)
        modulated_focus = modulated_focus.squeeze(-1).clamp(min=1e-6)  # (B, H, num_edges)
        # 적용: invalid edge의 log(modulated_focus)를 0으로 설정
        modulated_log = torch.log(modulated_focus)
        modulated_log = modulated_log.masked_fill(torch.logical_not(valid_edge_mask), 0)
        
        adjusted_scores = scores + torch.log(baseline_focus)
        batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1, 1).expand(batch_size, self.num_heads, edge_index_u.size(-1))
        head_indices = torch.arange(self.num_heads, device=x.device).view(1, -1, 1).expand(batch_size, -1, edge_index_u.size(-1))
        adjusted_scores[batch_indices, head_indices, edge_index_u, edge_index_v] = (
            scores[batch_indices, head_indices, edge_index_u, edge_index_v] + modulated_log
        )
        
        I = torch.eye(num_nodes, device=x.device).unsqueeze(0)
        adjusted_scores = adjusted_scores + I.unsqueeze(1) * self.self_loop_W
        
        if mask is not None:
            mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)
            adjusted_scores.masked_fill_(~mask_2d.unsqueeze(1), -1e6)
        
        attention_weights = torch.softmax(adjusted_scores, dim=-1)
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).reshape(batch_size, num_nodes, -1)
        return out

################################################################################
# 4. 모아레 레이어 (Edge Wave Superposition 적용, valid_edge_mask 추가)
################################################################################

class MoireLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads,
        shift_min,
        shift_max,
        dropout,
        focus,         # 예: get_moire_focus("gaussian")
        edge_attr_dim, # 엣지 속성 차원
    ):
        super(MoireLayer, self).__init__()

        shifts = [
            shift_min + random.uniform(0, 1) * (shift_max - shift_min)
            for _ in range(num_heads)
        ]
        widths = [1.3 ** shift for shift in shifts]

        self.attention = EdgeWaveAttention(
            input_dim,
            output_dim,
            num_heads,
            shifts,
            widths,
            focus,
            edge_attr_dim,
        )

        self.ffn = FFN(output_dim, output_dim, output_dim, dropout)
        self.projection_for_residual = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj, edge_index, edge_attr, mask, valid_edge_mask):
        h = self.attention(x, adj, edge_index, edge_attr, mask, valid_edge_mask)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        h = self.ffn(h)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        x_proj = self.projection_for_residual(x)
        h = h * 0.5 + x_proj * 0.5
        return h
