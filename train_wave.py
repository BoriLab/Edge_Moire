import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from data.qm9.qm9 import QM9Dataset
from data.pcqm.pcqm import PCQM4Mv2Dataset
from src.mymodel.layers import MoireLayer, get_moire_focus  # 수정된 Edge Wave Superposition 모듈 포함
from utils.exp import Aliquot, set_device, set_verbose

# Configuration settings

CONFIG = {
    "MODEL": "EdgeWaveSuperposition",
    "DATASET": "QM9",
    "DEPTH": 5 ,# [3, 5, 8, 13, 21]
    "MLP_DIM": 256,
    "HEADS": 32,
    "FOCUS": "gaussian",
    "DROPOUT": 0.1,
    "BATCH_SIZE": 512,
    "LEARNING_RATE": 5e-4,
    "WEIGHT_DECAY": 1e-2,
    "T_MAX": 200,
    "ETA_MIN": 1e-7,
    "DEVICE": "cuda",
    "SCALE_MIN": 0.6,
    "SCALE_MAX": 4.0,
    "WIDTH_BASE": 1.15,
    "VERBOSE": True,
}
# Device 설정
set_device(CONFIG["DEVICE"])

# 데이터셋 로드 (QM9 또는 PCQM4Mv2 선택)
dataset = None
if CONFIG["DATASET"] == "QM9":
    dataset = QM9Dataset(path="/home/bori9691/2025/Edge_Moire/data/qm9/1_qm9_data_eV_edges.pkl")
    criterion = nn.L1Loss()
    dataset.unsqueeze_target()
elif CONFIG["DATASET"] == "PCQM4Mv2":
    dataset = PCQM4Mv2Dataset(path="../../pcqm4mv2_data.pkl")
    criterion = nn.L1Loss()
    dataset.unsqueeze_target()

dataset.float()
dataset.batch_size = CONFIG["BATCH_SIZE"]

# MyModel 클래스 정의 (Edge Wave Superposition 모델, valid_edge_mask 인자 추가)
class MyModel(nn.Module):
    def __init__(self, config, dataset):
        super(MyModel, self).__init__()
        dims = config["MLP_DIM"]
        # 입력 임베딩: 노드 feature 차원은 dataset.node_feat_size
        self.input = nn.Sequential(
            nn.Linear(dataset.node_feat_size, dims),
            nn.ReLU(),
            nn.Linear(dims, dims),
        )
        # 여러 MoireLayer (Edge Wave Superposition 적용된 MoireLayer)
        self.layers = nn.ModuleList(
            [
                MoireLayer(
                    input_dim=dims,
                    output_dim=dims,
                    num_heads=config["HEADS"],
                    shift_min=config["SCALE_MIN"],
                    shift_max=config["SCALE_MAX"],
                    dropout=config["DROPOUT"],
                    focus=get_moire_focus(config["FOCUS"]),
                    edge_attr_dim=dataset.edge_attr_dim,  # 전처리된 edge_attr 차원
                )
                for _ in range(config["DEPTH"])
            ]
        )
        # 출력 임베딩: 예측 크기는 dataset.prediction_size
        self.output = nn.Sequential(
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, dataset.prediction_size),
        )
        # Residual을 위한 projection
        self.projection_for_residual = nn.Linear(dataset.node_feat_size, dims)

    def forward(self, x, adj, edge_index, edge_attr, mask, valid_edge_mask):
        # x: (batch_size, num_nodes, node_feat_size)
        x = self.input(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        for layer in self.layers:
            # 각 MoireLayer에 valid_edge_mask 인자를 함께 전달합니다.
            x = layer(x, adj, edge_index, edge_attr, mask, valid_edge_mask)
        if mask is not None:
            # 집계: max pooling over nodes
            x, _ = x.max(dim=1)
        x = self.output(x)
        # print(x.shape)
        return x

# 모델, 옵티마이저, 스케줄러, 손실함수 초기화
model = MyModel(CONFIG, dataset)
if CONFIG["DEVICE"] == "cuda":
    model = nn.DataParallel(model)
optimizer = optim.AdamW(
    model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"]
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CONFIG["T_MAX"], eta_min=CONFIG["ETA_MIN"]
)
criterion = nn.L1Loss()

# Aliquot 초기화 후 학습 실행
aliquot = Aliquot(
    model=model,
    dataset=dataset,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
)

aliquot(
    wandb_project="edge_wave_superposition_eV",
    wandb_config=CONFIG,
    num_epochs=10000,
    patience=50,
)
