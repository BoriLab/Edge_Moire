# %% Import necessary libraries
import os
import pandas as pd
import numpy as np
from IPython.display import display


def convert_to_float(value):
    try:
        return float(value.replace("*^", "e"))
    except ValueError:
        return float("nan")


def read_xyz(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

        # Number of atoms
        num_atoms = int(lines[0].strip())

        # Comment line (contains additional information about the molecule)
        comment = lines[1].strip().split()
        gdb_info = {
            "tag": comment[0],  # "gdb9"; string constant to ease extraction via grep
            "index": int(
                comment[1]
            ),  # Consecutive, 1-based integer identifier of molecule
            "A": convert_to_float(comment[2]),  # Rotational constant A (GHz)
            "B": convert_to_float(comment[3]),  # Rotational constant B (GHz)
            "C": convert_to_float(comment[4]),  # Rotational constant C (GHz)
            "mu": convert_to_float(comment[5]),  # Dipole moment (Debye)
            "alpha": convert_to_float(comment[6]),  # Isotropic polarizability (Bohr^3)
            "homo": convert_to_float(
                comment[7]
            ),  # Energy of Highest occupied molecular orbital (HOMO) (Hartree)
            "lumo": convert_to_float(
                comment[8]
            ),  # Energy of Lowest occupied molecular orbital (LUMO) (Hartree)
            "gap": convert_to_float(
                comment[9]
            ),  # Gap, difference between LUMO and HOMO (Hartree)
            "r2": convert_to_float(comment[10]),  # Electronic spatial extent (Bohr^2)
            "zpve": convert_to_float(
                comment[11]
            ),  # Zero point vibrational energy (Hartree)
            "U0": convert_to_float(comment[12]),  # Internal energy at 0 K (Hartree)
            "U": convert_to_float(comment[13]),  # Internal energy at 298.15 K (Hartree)
            "H": convert_to_float(comment[14]),  # Enthalpy at 298.15 K (Hartree)
            "G": convert_to_float(comment[15]),  # Free energy at 298.15 K (Hartree)
            "Cv": convert_to_float(
                comment[16]
            ),  # Heat capacity at 298.15 K (cal/(mol K))
        }

        # Atomic coordinates
        data = []
        for line in lines[2 : 2 + num_atoms]:
            parts = line.split()
            atom = parts[0]
            x, y, z = map(convert_to_float, parts[1:4])
            data.append([atom, x, y, z])

        # Create a DataFrame
        df = pd.DataFrame(data, columns=["Atom", "X", "Y", "Z"])

        return num_atoms, gdb_info, df


filename = []
nodes = []  # node features: atomic symbol
coords = []  # coordinates: x, y, z
adjs = []  # adjacency matrix: distance between atoms
targets = []  # target: gap


def make_adjacency_by_distance(coor):
    n = coor.shape[0]
    adj = np.zeros((n, n))
    for j in range(n):
        for k in range(j + 1, n):
            d = np.linalg.norm(coor[j] - coor[k])
            adj[j, k] = d
            adj[k, j] = d
    return adj


def read_xyz_directory(directory_path):
    for file in os.listdir(directory_path):
        if file.endswith(".xyz"):
            num_atoms, gdb_info, df = read_xyz(os.path.join(directory_path, file))
            filename.append(file)
            nodes.append(df["Atom"].values)
            # One-hot encoding of atomic symbols
            coords.append(df[["X", "Y", "Z"]].values)
            adjs.append(make_adjacency_by_distance(df[["X", "Y", "Z"]].values))
            targets.append(gdb_info["gap"])


directory_path = "/Users/saankim/datasets/qm9/raw"
results = read_xyz_directory(directory_path)
print(f"Number of files: {len(filename)}")
print(f"Number of nodes: {len(nodes)}")
print(f"Number of coordinates: {len(coords)}")
print(f"Number of adjacency matrices: {len(adjs)}")
print(f"Number of targets: {len(targets)}")

# %%
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def one_hot_encode_nodes(nodes):
    # Flatten the list of lists
    flattened_nodes = [atom for molecule in nodes for atom in molecule]

    # Label encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(flattened_nodes)

    # One-hot encoding
    onehot_encoder = OneHotEncoder(sparse=False)  # Set sparse=False to prevent sparse array
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    # Reshape the one-hot encoded array back to original molecules shape
    idx = 0
    one_hot_nodes = []
    for molecule in nodes:
        one_hot_nodes.append(onehot_encoded[idx : idx + len(molecule)])
        idx += len(molecule)

    return one_hot_nodes


one_hot_nodes = one_hot_encode_nodes(nodes)


# %%
# print first element of each list
# display(filename[0])
display(nodes[0])
display(one_hot_nodes[2])
# display(coords[0])
# display(adjs[0])
# display(targets[0])

# %%
# check dimensions of each data point in the list
# len(node) == len(coord) == len(adj)

for i in range(len(filename)):
    if not (
        len(nodes[i]) == len(coords[i]) == len(adjs[i]) == one_hot_nodes[i].shape[0]
    ):
        print(f"Error: {filename[i]}")
        break

# %%
# save files into a pickle file
import pickle

with open("qm9_data.pkl", "wb") as file:
    pickle.dump(
        {
            "filename": filename,
            "nodes": nodes,
            "one_hot_nodes": one_hot_nodes,
            "coords": coords,
            "adjs": adjs,
            "targets": targets,
        },
        file,
    )


# %%
# Split the data into train, validation, and test sets
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
            "targets": self.data["targets"][idx],
        }


def collate_fn(batch):
    return {
        "filename": [data["filename"] for data in batch],
        "nodes": [data["nodes"] for data in batch],
        "one_hot_nodes": [data["one_hot_nodes"] for data in batch],
        "coords": [data["coords"] for data in batch],
        "adjs": [data["adjs"] for data in batch],
        "targets": torch.tensor([data["targets"] for data in batch]),
    }


with open("qm9_data.pkl", "rb") as file:
    qm9_data = pickle.load(file)

# Split the data into train, validation, and test sets
train_data, test_data = train_test_split(
    qm9_data, test_size=0.2, random_state=42, shuffle=True
)
train_data, val_data = train_test_split(
    train_data, test_size=0.2, random_state=42, shuffle=True
)

# Create Dataset and DataLoader objects
train_dataset = QM9Dataset(train_data)
val_dataset = QM9Dataset(val_data)
test_dataset = QM9Dataset(test_data)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
