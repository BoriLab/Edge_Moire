

import os
import random
import numpy as np
import torch
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import Dataset
import pandas as pd

##############################################################################
# 시드 고정 함수
##############################################################################
def set_seed(seed=42):
    """
    Python, NumPy, PyTorch, (및 CUDA)가 지원되는 경우 CUDA까지 동일한 랜덤 시드 설정.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


##############################################################################
# QM9 데이터 전처리 함수들
##############################################################################
def convert_to_float(value):
    try:
        return float(value.replace("*^", "e"))
    except ValueError:
        return float("nan")

def read_xyz(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        num_atoms = int(lines[0].strip())
        comment = lines[1].strip().split()
        gdb_info = {
            "tag": comment[0],
            "index": int(comment[1]),
            "A": convert_to_float(comment[2]),
            "B": convert_to_float(comment[3]),
            "C": convert_to_float(comment[4]),
            "mu": convert_to_float(comment[5]),
            "alpha": convert_to_float(comment[6]),
            "homo": convert_to_float(comment[7]),
            "lumo": convert_to_float(comment[8]),
            "gap": convert_to_float(comment[9]),
            "r2": convert_to_float(comment[10]),
            "zpve": convert_to_float(comment[11]),
            "U0": convert_to_float(comment[12]),
            "U": convert_to_float(comment[13]),
            "H": convert_to_float(comment[14]),
            "G": convert_to_float(comment[15]),
            "Cv": convert_to_float(comment[16]),
        }
        data = []
        for line in lines[2 : 2 + num_atoms]:
            parts = line.split()
            atom = parts[0]
            x, y, z = map(convert_to_float, parts[1:4])
            data.append([atom, x, y, z])
        df = pd.DataFrame(data, columns=["Atom", "X", "Y", "Z"])
        return num_atoms, gdb_info, df

def make_adjacency_by_distance(coor, threshold=1.5):
    n = coor.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coor[i] - coor[j])
            if d < threshold:
                adj[i, j] = d
                adj[j, i] = d
    return adj

def extract_edge_info(coor, thresholds=[1.5, 2.0, 2.5]):
    edge_index = []
    edge_attr = []
    n = coor.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(coor[i] - coor[j])
            if any(distance < threshold for threshold in thresholds):
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_attr.append([1 if distance < t else 0 for t in thresholds])
                edge_attr.append([1 if distance < t else 0 for t in thresholds])
    edge_index = np.array(edge_index).T
    edge_attr = np.array(edge_attr)
    return edge_index, edge_attr

def one_hot_encode_nodes(nodes):
    flattened_nodes = [atom for molecule in nodes for atom in molecule]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(flattened_nodes)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    idx = 0
    one_hot_nodes = []
    for molecule in nodes:
        one_hot_nodes.append(onehot_encoded[idx : idx + len(molecule)])
        idx += len(molecule)
    return one_hot_nodes

def read_xyz_directory(directory_path):
    filename, nodes, coords, adjs, edge_indices, edge_attrs, targets = [], [], [], [], [], [], []
    for file in os.listdir(directory_path):
        if file.endswith(".xyz"):
            num_atoms, gdb_info, df = read_xyz(os.path.join(directory_path, file))
            filename.append(file)
            nodes.append(df["Atom"].values)
            coords.append(df[["X", "Y", "Z"]].values)
            adjs.append(make_adjacency_by_distance(df[["X", "Y", "Z"]].values))
            edge_index, edge_attr = extract_edge_info(df[["X", "Y", "Z"]].values)
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)
            # 기존 Hartree 단위의 gap 값을 eV 단위로 변환하여 저장 (1 Hartree ≈ 27.2114 eV)
            targets.append(gdb_info["gap"] * 27.2114)
    one_hot_nodes = one_hot_encode_nodes(nodes)
    return filename, nodes, one_hot_nodes, coords, adjs, edge_indices, edge_attrs, targets

def save_qm9_data_to_pickle(data, file_path="1_qm9_data_eV_edges.pkl"):
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


##############################################################################
# Dataset 클래스
##############################################################################
class QM9Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["filename"])

    def __getitem__(self, idx):
        return {
            "filename": self.data["filename"][idx],
            "nodes": self.data["nodes"][idx],
            "one_hot_nodes": self.data["one_hot_nodes"][idx],
            "coords": self.data["coords"][idx],
            "adjs": self.data["adjs"][idx],
            "edge_index": self.data["edge_indices"][idx],
            "edge_attr": self.data["edge_attrs"][idx],
            "targets": self.data["targets"][idx],
        }


##############################################################################
# 전체 데이터를 Dataset 객체로 로드 및 피클 저장 (DataLoader 없이)
##############################################################################
def load_and_save_qm9_dataset(directory_path, pkl_path="1_qm9_data_eV_edges.pkl", seed=42):
    # 시드 고정
    set_seed(seed)

    filename, nodes, one_hot_nodes, coords, adjs, edge_indices, edge_attrs, targets = read_xyz_directory(directory_path)
    data = {
        "filename": filename,
        "nodes": nodes,
        "one_hot_nodes": one_hot_nodes,
        "coords": coords,
        "adjs": adjs,
        "edge_indices": edge_indices,
        "edge_attrs": edge_attrs,
        "targets": targets,
    }
    
    # 데이터 피클 파일로 저장
    save_qm9_data_to_pickle(data, pkl_path)
    
    # Dataset 객체 생성
    dataset = QM9Dataset(data)
    
    return dataset

# 사용 예시
if __name__ == "__main__":
    directory_path = "/home/bori9691/2025/dataset/qm9/raw"
    dataset = load_and_save_qm9_dataset(directory_path, seed=42)
    
    # Dataset에서 첫 샘플을 출력하여 확인
    print(dataset[0])

