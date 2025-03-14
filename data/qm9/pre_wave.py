import os
import random
import numpy as np
import torch
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import Dataset
import pandas as pd
from rdkit import Chem

##############################################################################
# 시드 고정 함수
##############################################################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##############################################################################
# 헬퍼 함수: 각도 및 dihedral 계산
##############################################################################
def compute_angle(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)

def compute_dihedral(a, b, c, d):
    b0 = a - b
    b1 = c - b
    b2 = d - c
    b1 = b1 / (np.linalg.norm(b1) + 1e-8)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    angle = np.arctan2(y, x)
    return np.degrees(angle)

def get_neighbors(coor, i, threshold=1.5):
    neighbors = []
    n = coor.shape[0]
    for k in range(n):
        if k != i:
            d = np.linalg.norm(coor[i] - coor[k])
            if d < threshold:
                neighbors.append(k)
    return neighbors

def compute_bond_angle(coor, i, j):
    angles = []
    neighbors_i = get_neighbors(coor, i)
    if j in neighbors_i:
        neighbors_i.remove(j)
    for k in neighbors_i:
        v1 = coor[j] - coor[i]
        v2 = coor[k] - coor[i]
        angles.append(compute_angle(v1, v2))
    neighbors_j = get_neighbors(coor, j)
    if i in neighbors_j:
        neighbors_j.remove(i)
    for l in neighbors_j:
        v1 = coor[i] - coor[j]
        v2 = coor[l] - coor[j]
        angles.append(compute_angle(v1, v2))
    if angles:
        return np.mean(angles)
    else:
        return 0.0

def compute_dihedral_angle(coor, i, j):
    dihedrals = []
    neighbors_i = get_neighbors(coor, i)
    if j in neighbors_i:
        neighbors_i.remove(j)
    neighbors_j = get_neighbors(coor, j)
    if i in neighbors_j:
        neighbors_j.remove(i)
    for k in neighbors_i:
        for l in neighbors_j:
            dihedrals.append(compute_dihedral(coor[k], coor[i], coor[j], coor[l]))
    if dihedrals:
        return np.mean(dihedrals)
    else:
        return 0.0

##############################################################################
# 화학적 특성 관련 함수
##############################################################################
# Covalent radii (Å)
covalent_radii = {
    'H': 0.31,
    'C': 0.76,
    'N': 0.71,
    'O': 0.66,
    'F': 0.57,
    'P': 1.07,
    'S': 1.05,
    'Cl': 1.02,
}

def get_bond_order(atom1, atom2, distance, factor=1.2, tol=0.1):
    """
    각 원자의 covalent radius 합에 factor를 곱한 값과 실제 거리를 비교하여,
    tol 이하이면 double bond (2), 그렇지 않으면 single bond (1)로 추정하는 간단한 휴리스틱.
    """
    r1 = covalent_radii.get(atom1, 0.7)
    r2 = covalent_radii.get(atom2, 0.7)
    d_expected = (r1 + r2) * factor
    if distance < d_expected * (1 - tol):
        return 2
    else:
        return 1

##############################################################################
# RDKit 기반 Molecule 생성 (수정됨: 최대 허용 결합 수를 자체적으로 추적)
##############################################################################
def build_rdkit_mol(df, factor=1.0, tol=0.1):
    """
    df: DataFrame with columns ["Atom", "X", "Y", "Z"]
    factor, tol: 각 원자의 covalent radius 합에 기반한 결합 추가를 위한 파라미터.
    각 원자 쌍에 대해 거리가 (r1 + r2) * factor 이하일 때만 결합을 추가하며,
    현재 결합 수와 추가될 결합 수를 자체적으로 추적하여 최대 허용 결합 수를 초과하지 않도록 합니다.
    """
    mol = Chem.RWMol()
    atoms = df["Atom"].values
    coords = df[["X", "Y", "Z"]].values
    n = len(atoms)
    
    # 원자별 최대 허용 결합 수 (일반적인 값; 필요시 조정)
    max_valence = {
        'H': 1,
        'C': 4,
        'N': 3,
        'O': 2,
        'F': 1,
        'P': 5,
        'S': 2,
        'Cl': 1,
    }
    
    # 각 원자의 현재 결합 수를 추적하는 리스트 (초기값 0)
    current_valences = [0] * n
    
    # 원자 추가
    for i in range(n):
        atom = Chem.Atom(atoms[i])
        mol.AddAtom(atom)
    
    # 좌표 기반으로 결합 추가: 각 원자 쌍에 대해 covalent radii 기반으로 결정
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            r1 = covalent_radii.get(atoms[i], 0.7)
            r2 = covalent_radii.get(atoms[j], 0.7)
            d_expected = (r1 + r2) * factor
            if d < d_expected:
                allowed_i = max_valence.get(atoms[i], 4)
                allowed_j = max_valence.get(atoms[j], 4)
                # 예상 추가 결합 수 (get_bond_order를 두 번 호출해도 되지만, 여기서는 동일 값으로 가정)
                additional = get_bond_order(atoms[i], atoms[j], d, factor, tol)
                # 현재 추적된 결합 수와 추가될 결합 수의 합이 허용 범위 내인지 체크
                if current_valences[i] + additional <= allowed_i and current_valences[j] + additional <= allowed_j:
                    if additional == 1:
                        bond_type = Chem.rdchem.BondType.SINGLE
                    elif additional == 2:
                        bond_type = Chem.rdchem.BondType.DOUBLE
                    elif additional >= 3:
                        bond_type = Chem.rdchem.BondType.TRIPLE
                    else:
                        bond_type = Chem.rdchem.BondType.SINGLE
                    mol.AddBond(i, j, bond_type)
                    current_valences[i] += additional
                    current_valences[j] += additional
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        print("Sanitization error:", e)
    return mol




##############################################################################
# RDKit 기반 Bond Type 매핑 함수
##############################################################################
def map_bond_type(bond):
    """
    RDKit bond 객체의 BondType을 정수로 매핑합니다.
    매핑: SINGLE -> 1, DOUBLE -> 2, TRIPLE -> 3, AROMATIC -> 4, 그 외 -> 0
    """
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        return 1
    elif bt == Chem.rdchem.BondType.DOUBLE:
        return 2
    elif bt == Chem.rdchem.BondType.TRIPLE:
        return 3
    elif bt == Chem.rdchem.BondType.AROMATIC:
        return 4
    else:
        return 0

##############################################################################
# RDKit 기반 엣지 특성 추출 함수 (Bond Type: SINGLE, DOUBLE, TRIPLE, AROMATIC)
##############################################################################
def extract_edge_info(coor, atoms, mol):
    """
    RDKit Molecule(mol) 객체를 활용하여, 각 원자 쌍 (i, j)에 대해 결합이 존재하면 아래 특성을 추출:
      1. Bond Type: RDKit bond.GetBondType() 기반 (1: single, 2: double, 3: triple, 4: aromatic)
      2. Bond Length: 원자 간 유클리드 거리
      3. Bond Angle: (i, j) 결합에 대한 평균 결합 각도 (주변 이웃 기반)
      4. Dihedral Angle: 가능한 dihedral 각도의 평균 (주변 이웃 기반)
    
    결합이 존재하지 않는 (i, j) 쌍은 생략합니다.
    
    반환:
      edge_index: shape [2, num_edges] (양방향)
      edge_attr: shape [num_edges, 4]
    """
    edge_index = []
    edge_attr = []
    n = coor.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is None:
                continue
            bond_type = map_bond_type(bond)
            bond_length = np.linalg.norm(coor[i] - coor[j])
            bond_angle = compute_bond_angle(coor, i, j)
            dihedral_angle = compute_dihedral_angle(coor, i, j)
            feature_vec = [bond_type, bond_length, bond_angle, dihedral_angle]
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append(feature_vec)
            edge_attr.append(feature_vec)
    if len(edge_index) > 0:
        edge_index = np.array(edge_index).T
        edge_attr = np.array(edge_attr)
    else:
        edge_index = np.empty((2, 0))
        edge_attr = np.empty((0, 4))
    return edge_index, edge_attr

##############################################################################
# 기타 전처리 함수들: read_xyz, make_adjacency_by_distance, one_hot_encode_nodes
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
        for line in lines[2: 2 + num_atoms]:
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
        for j in range(i+1, n):
            d = np.linalg.norm(coor[i] - coor[j])
            if d < threshold:
                adj[i, j] = d
                adj[j, i] = d
    return adj

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
        one_hot_nodes.append(onehot_encoded[idx: idx + len(molecule)])
        idx += len(molecule)
    return one_hot_nodes

def read_xyz_directory(directory_path):
    filename, nodes, coords, adjs, edge_indices, edge_attrs, targets = [], [], [], [], [], [], []
    for file in os.listdir(directory_path):
        if file.endswith(".xyz"):
            num_atoms, gdb_info, df = read_xyz(os.path.join(directory_path, file))
            filename.append(file)
            atoms = df["Atom"].values
            nodes.append(atoms)
            coords_arr = df[["X", "Y", "Z"]].values
            coords.append(coords_arr)
            adjs.append(make_adjacency_by_distance(coords_arr))
            mol = build_rdkit_mol(df)
            edge_index, edge_attr = extract_edge_info(coords_arr, atoms, mol)
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)
            targets.append(gdb_info["gap"])  # Hartree -> eV 변환
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
# 전체 데이터를 Dataset 객체로 로드 및 피클 저장
##############################################################################
def load_and_save_qm9_dataset(directory_path, pkl_path="1_qm9_data_eV_edges.pkl", seed=42):
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
    save_qm9_data_to_pickle(data, pkl_path)
    dataset = QM9Dataset(data)
    return dataset

# 사용 예시
if __name__ == "__main__":
    directory_path = "/home/bori9691/2025/dataset/qm9/raw"
    dataset = load_and_save_qm9_dataset(directory_path, seed=42)
    print(dataset[0])
