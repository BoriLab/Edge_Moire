import argparse
from data.qm9.qm9 import QM9Dataset
from data.pcqm.pcqm import PCQM4Mv2Dataset
from src.mymodel.layers import MoireLayer, get_moire_focus
from utils.exp import Aliquot, set_device, set_verbose
from torch.optim.lr_scheduler import LambdaLR
from torch import optim
import torch.nn as nn
import torch

# Configuration settings
CONFIG = {
    "MODEL": "Moire",
    "DATASET": "QM9",
    "DEPTH": 7,  # [3, 5, 8, 13, 21]
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
    "SCALE_MAX": 3.0,
    "WIDTH_BASE": 1.15,
    "VERBOSE": True,
}


set_device(CONFIG["DEVICE"])
dataset = None
match CONFIG["DATASET"]:
    case "QM9":
        dataset = QM9Dataset(path="/home/bioscience/dev/SJ_paper/Mn_moire2/data/qm9/qm9_data_with_edges.pkl")
        criterion = nn.L1Loss()
        dataset.unsqueeze_target()
    case "PCQM4Mv2":
        dataset = PCQM4Mv2Dataset(path="../../pcqm4mv2_data.pkl")
        criterion = nn.L1Loss()
        dataset.unsqueeze_target()
dataset.float()
dataset.batch_size = CONFIG["BATCH_SIZE"]

# 데이터셋에서 edge_attr_dim을 설정해야 합니다.

# MyModel 클래스 수정

class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        dims = config["MLP_DIM"]
        self.input = nn.Sequential(
            nn.Linear(dataset.node_feat_size, dims),
            nn.Linear(dims, dims),
        )
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
                    edge_attr_dim=dataset.edge_attr_dim,  # edge_attr_dim 추가
                )
                for _ in range(config["DEPTH"])
            ]
        )
        self.output = nn.Sequential(
            nn.Linear(dims, dims),
            nn.Linear(dims, dataset.prediction_size),
        )


    def forward(self, x, adj, edge_index, edge_attr,mask):
        x = self.input(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, adj, edge_index, edge_attr, mask)
        if mask is not None:
            x, _ = x.max(dim=1)
        x = self.output(x)

        return x

# Initialize model, optimizer, scheduler, and criterion
model = MyModel(CONFIG)
if CONFIG["DEVICE"] == "cuda":
    model = nn.DataParallel(model)
optimizer = optim.AdamW(
    model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"]
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CONFIG["T_MAX"], eta_min=CONFIG["ETA_MIN"]
)
criterion = nn.L1Loss()

# Initialize Aliquot for training
aliquot = Aliquot(
    model=model,
    dataset=dataset,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
)(wandb_project="moire_edge_FFN", wandb_config=CONFIG, num_epochs=1000, patience=20)

# Training loop
# aliquot(wandb_project="moire_edge", wandb_config=CONFIG, num_epochs=1000, patience=20)
