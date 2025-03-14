# import torch
# from torch.utils.data import DataLoader, Dataset
# import random
# import torch.nn.functional as F
# from torch.utils.data.sampler import SubsetRandomSampler
# import scipy.sparse as sp
# import torch.nn.functional as F



# def collate_batch(batch):
#     # batch: list of tuples (node_features, adj, edge_index, edge_attr, targets)
#     node_features, adj_matrices, edge_indices, edge_attrs, targets = zip(*batch)
    
#     # 각 샘플별 최대 노드 수 계산 (노드 수와 edge_index의 최대 인덱스+1 중 큰 값)
#     per_sample_max = []
#     valid_edge_masks = []  # 각 샘플마다 유효한 edge에 대한 마스크 (True/False)
#     for nf, ei in zip(node_features, edge_indices):
#         if ei.numel() > 0:
#             sample_max = max(nf.size(0), int(ei.max().item()) + 1)
#             valid_mask = (ei >= 0)
#         else:
#             sample_max = nf.size(0)
#             valid_mask = torch.ones_like(ei, dtype=torch.bool)
#         per_sample_max.append(sample_max)
#         valid_edge_masks.append(valid_mask)
#         print(f"Sample: nf nodes = {nf.size(0)}, ei max index = {ei.max().item() if ei.numel()>0 else 'N/A'}, sample_max = {sample_max}")
    
#     max_nodes = max(per_sample_max)
#     print("Batch max_nodes:", max_nodes)
    
#     # 노드 마스크 생성
#     node_masks = [
#         torch.cat([torch.ones(nf.size(0)), torch.zeros(max_nodes - nf.size(0))])
#         for nf in node_features
#     ]
    
#     # 노드 특징 패딩
#     padded_node_features = [
#         F.pad(nf, (0, 0, 0, max_nodes - nf.size(0))) for nf in node_features
#     ]
#     # 인접 행렬 패딩
#     padded_adj_matrices = [
#         F.pad(adj, (0, max_nodes - adj.size(1), 0, max_nodes - adj.size(0))) for adj in adj_matrices
#     ]
#     # 엣지 인덱스 및 엣지 속성 패딩 (패딩 값 0으로, 그리고 별도로 valid_edge_mask 저장)
#     padded_edge_indices = []
#     padded_edge_attrs = []
#     padded_valid_masks = []
#     max_edges = max([ei.size(1) for ei in edge_indices])
#     print("Batch max_edges:", max_edges)
#     for ei, ea, vm in zip(edge_indices, edge_attrs, valid_edge_masks):
#         pad_size = max_edges - ei.size(1)
#         if pad_size > 0:
#             pad_ei = torch.zeros((2, pad_size), dtype=torch.long)  # 패딩 값을 0으로 채움
#             ei = torch.cat([ei, pad_ei], dim=1)
#             pad_ea = torch.zeros((pad_size, ea.size(1)))
#             ea = torch.cat([ea, pad_ea], dim=0)
#             pad_vm = torch.zeros((2, pad_size), dtype=torch.bool)
#             vm = torch.cat([vm, pad_vm], dim=1)
#         padded_edge_indices.append(ei)
#         padded_edge_attrs.append(ea)
#         padded_valid_masks.append(vm)
#     node_features = torch.stack(padded_node_features)
#     adj_matrices = torch.stack(padded_adj_matrices)
#     edge_indices = torch.stack(padded_edge_indices)
#     edge_attrs = torch.stack(padded_edge_attrs)
#     node_masks = torch.stack(node_masks).bool()
#     targets = torch.stack(targets)
#     # 패딩된 edge의 유효성 마스크를 함께 반환 (추후 모델에서 -1 인덱스를 무시하는 대신 사용)
#     return (node_features, adj_matrices, edge_indices, edge_attrs, node_masks, targets, padded_valid_masks)

# class MyDataset(Dataset):
#     def __init__(
#         self,
#         node_features,
#         adj_matrices,
#         edge_indices,
#         edge_attrs,
#         targets,
#         evaluation_size=0.025,
#         test_size=0.025,
#         batch_size=32,
#         seed=42
#     ):
#         self.node_features = node_features
#         self.adj_matrices = adj_matrices
#         self.edge_indices = edge_indices
#         self.edge_attrs = edge_attrs
#         self.targets = targets
#         self.batch_size = batch_size
#         self.seed = seed

#         for i in range(len(self.node_features)):
#             if sp.issparse(self.node_features[i]):
#                 self.node_features[i] = self.node_features[i].toarray()
#             self.node_features[i] = torch.tensor(self.node_features[i], dtype=torch.float)
#             if sp.issparse(self.adj_matrices[i]):
#                 self.adj_matrices[i] = self.adj_matrices[i].toarray()
#             self.adj_matrices[i] = torch.tensor(self.adj_matrices[i], dtype=torch.float)
#             self.edge_indices[i] = torch.tensor(self.edge_indices[i], dtype=torch.long)
#             self.edge_attrs[i] = torch.tensor(self.edge_attrs[i], dtype=torch.float)
#             self.targets[i] = torch.tensor(self.targets[i], dtype=torch.float)

#         random.seed(self.seed)
#         indices = list(range(len(self.node_features)))
#         random.shuffle(indices)
#         if evaluation_size < 1:
#             evaluation_size = int(evaluation_size * len(indices))
#         if test_size < 1:
#             test_size = int(test_size * len(indices))
#         self.indices = {
#             "train": indices[test_size + evaluation_size:],
#             "eval": indices[:evaluation_size],
#             "test": indices[evaluation_size: test_size + evaluation_size],
#         }

#         self.node_feat_size = self.node_features[0].shape[1]
#         self.prediction_size = self.targets[0].shape[0] if self.targets[0].dim() > 0 else 1

#     def float(self):
#         for i in range(len(self.node_features)):
#             self.node_features[i] = self.node_features[i].float()
#             self.adj_matrices[i] = self.adj_matrices[i].float()
#             self.edge_attrs[i] = self.edge_attrs[i].float()
#             self.targets[i] = self.targets[i].float()

#     def unsqueeze_target(self):
#         for i in range(len(self.targets)):
#             if self.targets[i].dim() == 0:
#                 self.targets[i] = self.targets[i].unsqueeze(-1)

#     def __len__(self):
#         return len(self.node_features)

#     def __getitem__(self, idx):
#         return (
#             self.node_features[idx],
#             self.adj_matrices[idx],
#             self.edge_indices[idx],
#             self.edge_attrs[idx],
#             self.targets[idx],
#         )

#     def get_dataloader(self, split="train"):
#         if split == "train":
#             sampler = SubsetRandomSampler(self.indices[split])
#             shuffle = False
#         else:
#             sampler = self.indices[split]
#             shuffle = False
#         return DataLoader(
#             self,
#             batch_size=self.batch_size,
#             sampler=sampler,
#             collate_fn=collate_batch,
#             shuffle=shuffle,
#         )

#     def train(self):
#         return self.get_dataloader(split="train")

#     def eval(self):
#         return self.get_dataloader(split="eval")

#     def test(self):
#         return self.get_dataloader(split="test")
import torch
from torch.utils.data import DataLoader, Dataset
import random
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import scipy.sparse as sp
import torch.nn.functional as F

def collate_batch(batch):
    # batch: list of tuples (node_features, adj, edge_index, edge_attr, targets, mask, valid_edge_mask)
    node_features, adj_matrices, edge_indices, edge_attrs, masks,valid_edge_masks,targets = zip(*batch)
    
    # 각 샘플별 최대 노드 수 계산 (노드 수와 edge_index의 최대 인덱스+1 중 큰 값)
    per_sample_max = []
    for nf, ei in zip(node_features, edge_indices):
        if ei.numel() > 0:
            sample_max = max(nf.size(0), int(ei.max().item()) + 1)
        else:
            sample_max = nf.size(0)
        per_sample_max.append(sample_max)
        # print(f"Sample: nf nodes = {nf.size(0)}, ei max index = {ei.max().item() if ei.numel()>0 else 'N/A'}, sample_max = {sample_max}")
    
    max_nodes = max(per_sample_max)
    # print("Batch max_nodes:", max_nodes)
    
    # 노드 마스크 생성 (이미 masks가 존재하면 사용)
    padded_masks = [F.pad(mask, (0, max_nodes - mask.size(0))) for mask in masks]
    
    # 노드 특징 패딩
    padded_node_features = [F.pad(nf, (0, 0, 0, max_nodes - nf.size(0))) for nf in node_features]
    # 인접 행렬 패딩
    padded_adj_matrices = [F.pad(adj, (0, max_nodes - adj.size(1), 0, max_nodes - adj.size(0))) for adj in adj_matrices]
    
    # 엣지 인덱스 및 엣지 속성, valid_edge_mask 패딩 (패딩 값 0으로)
    padded_edge_indices = []
    padded_edge_attrs = []
    padded_valid_masks = []
    max_edges = max([ei.size(1) for ei in edge_indices])
    # print("Batch max_edges:", max_edges)
    for ei, ea, vm in zip(edge_indices, edge_attrs, valid_edge_masks):
        pad_size = max_edges - ei.size(1)
        if pad_size > 0:
            pad_ei = torch.zeros((2, pad_size), dtype=torch.long)  # 0으로 패딩
            ei = torch.cat([ei, pad_ei], dim=1)
            pad_ea = torch.zeros((pad_size, ea.size(1)))
            ea = torch.cat([ea, pad_ea], dim=0)
            pad_vm = torch.zeros((2, pad_size), dtype=torch.bool)
            vm = torch.cat([vm, pad_vm], dim=1)
        padded_edge_indices.append(ei)
        padded_edge_attrs.append(ea)
        padded_valid_masks.append(vm)
    
    node_features = torch.stack(padded_node_features)
    adj_matrices = torch.stack(padded_adj_matrices)
    edge_indices = torch.stack(padded_edge_indices)
    edge_attrs = torch.stack(padded_edge_attrs)
    masks = torch.stack(padded_masks).bool()
    padded_valid_masks = torch.stack(padded_valid_masks)
    targets = torch.stack(targets)
    
    # print("targetdataset",targets.shape)
    # 여기서 valid_edge_masks 리스트를 하나의 텐서로 스택 (shape: [batch_size, 2, max_edges])
    
    return (node_features, adj_matrices, edge_indices, edge_attrs, masks, padded_valid_masks, targets)

class MyDataset(Dataset):
    def __init__(
        self,
        node_features,
        adj_matrices,
        edge_indices,
        edge_attrs,
        targets,
        evaluation_size=0.025,
        test_size=0.025,
        batch_size=32,
        seed=42
    ):
        self.node_features = node_features
        self.adj_matrices = adj_matrices
        self.edge_indices = edge_indices
        self.edge_attrs = edge_attrs
        self.targets = targets
        self.batch_size = batch_size
        self.seed = seed

        # 데이터 전처리: 각 항목을 텐서로 변환
        for i in range(len(self.node_features)):
            if sp.issparse(self.node_features[i]):
                self.node_features[i] = self.node_features[i].toarray()
            self.node_features[i] = torch.tensor(self.node_features[i], dtype=torch.float)
            if sp.issparse(self.adj_matrices[i]):
                self.adj_matrices[i] = self.adj_matrices[i].toarray()
            self.adj_matrices[i] = torch.tensor(self.adj_matrices[i], dtype=torch.float)
            self.edge_indices[i] = torch.tensor(self.edge_indices[i], dtype=torch.long)
            self.edge_attrs[i] = torch.tensor(self.edge_attrs[i], dtype=torch.float)
            self.targets[i] = torch.tensor(self.targets[i], dtype=torch.float)

        # 각 샘플에 대해 노드 마스크: 모든 노드는 유효하다고 가정
        self.masks = [torch.ones(self.node_features[i].size(0)) for i in range(len(self.node_features))]
        # 각 샘플에 대해 valid_edge_mask: edge_indices가 0 이상의 값이면 True
        self.valid_edge_masks = [(self.edge_indices[i] >= 0) for i in range(len(self.edge_indices))]

        random.seed(self.seed)
        indices = list(range(len(self.node_features)))
        random.shuffle(indices)
        if evaluation_size < 1:
            evaluation_size = int(evaluation_size * len(indices))
        if test_size < 1:
            test_size = int(test_size * len(indices))
        self.indices = {
            "train": indices[test_size + evaluation_size:],
            "eval": indices[:evaluation_size],
            "test": indices[evaluation_size: test_size + evaluation_size],
        }

        self.node_feat_size = self.node_features[0].shape[1]
        # target이 스칼라인 경우(0-dim) unsqueeze해서 [1] shape로 만듭니다.
        if self.targets[0].dim() == 0:
            self.prediction_size = 1
        else:
            self.prediction_size = self.targets[0].shape[0]

    def float(self):
        for i in range(len(self.node_features)):
            self.node_features[i] = self.node_features[i].float()
            self.adj_matrices[i] = self.adj_matrices[i].float()
            self.edge_attrs[i] = self.edge_attrs[i].float()
            self.targets[i] = self.targets[i].float()

    def unsqueeze_target(self):
        for i in range(len(self.targets)):
            if self.targets[i].dim() == 0:
                self.targets[i] = self.targets[i].unsqueeze(0)

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        # target 반환 전에 target의 차원이 0-dim이면 [1]로 변경
        target = self.targets[idx]
        if target.dim() == 0:
            target = target.unsqueeze(0)
        return (
            self.node_features[idx],
            self.adj_matrices[idx],
            self.edge_indices[idx],
            self.edge_attrs[idx],
            self.masks[idx],
            self.valid_edge_masks[idx],
            target,
        )

    def get_dataloader(self, split="train"):
        if split == "train":
            sampler = SubsetRandomSampler(self.indices[split])
            shuffle = False
        else:
            sampler = self.indices[split]
            shuffle = False
        return DataLoader(
            self,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=collate_batch,
            shuffle=shuffle,
        )

    def train(self):
        return self.get_dataloader(split="train")

    def eval(self):
        return self.get_dataloader(split="eval")

    def test(self):
        return self.get_dataloader(split="test")
