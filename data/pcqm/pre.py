# %% Import necessary libraries
import os
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from ogb.lsc import PCQM4Mv2Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
from scipy.spatial.distance import pdist, squareform
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Suppress RDKit warnings and informational messages
RDLogger.DisableLog("rdApp.*")

# Load the dataset to get target values
ROOT = "./data/pcqm/"  # Replace with your dataset root directory
dataset = PCQM4Mv2Dataset(root=ROOT, only_smiles=True)

# Load the SDF file containing 3D structures
sdf_path = "./data/pcqm/pcqm4m-v2-train.sdf"  # Replace with your SDF file path
suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
print("DONE: Load the dataset and SDF file")

# Function to process a batch of molecules
def process_batch(batch_indices):
    batch_mol_ids = []
    batch_nodes = []
    batch_coords = []
    batch_adjs = []
    batch_targets = []

    for idx in batch_indices:
        mol = suppl[idx]
        if mol is None:
            continue  # Skip if molecule cannot be read

        # Get the SMILES string and target value from the dataset
        smiles, target = dataset[idx]

        # Get atom symbols
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        batch_nodes.append(atoms)

        # Get 3D coordinates
        conformer = mol.GetConformer()
        coords_array = conformer.GetPositions()
        batch_coords.append(coords_array)

        # Compute adjacency matrix based on distances
        adj = squareform(pdist(coords_array))
        batch_adjs.append(adj)

        # Append target value and molecule index
        batch_targets.append(target)
        batch_mol_ids.append(idx)

    return batch_mol_ids, batch_nodes, batch_coords, batch_adjs, batch_targets


# Function to one-hot encode a batch of atom symbols
def one_hot_encode_batch(nodes_batch):
    # Flatten the list of atom symbols in the batch
    flattened_nodes = [atom for molecule in nodes_batch for atom in molecule]

    # Label encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(flattened_nodes)

    # One-hot encoding
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    # Reshape back to the original molecules
    idx = 0
    one_hot_nodes_batch = []
    for molecule in nodes_batch:
        n_atoms = len(molecule)
        one_hot_nodes_batch.append(onehot_encoded[idx : idx + n_atoms])
        idx += n_atoms

    return one_hot_nodes_batch


# Define batch size based on your system's memory capacity
batch_size = 300000  # Adjust this value as needed
num_molecules = len(suppl)
num_batches = (num_molecules + batch_size - 1) // batch_size  # Ceiling division

# Prepare to save data incrementally
data_dir = "processed_batches"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

for batch_num in range(num_batches):
    start_idx = batch_num * batch_size
    end_idx = min((batch_num + 1) * batch_size, num_molecules)
    batch_indices = range(start_idx, end_idx)

    print(
        f"Processing batch {batch_num + 1}/{num_batches}: molecules {start_idx} to {end_idx - 1}"
    )

    # Process the batch
    batch_results = process_batch(batch_indices)

    # Unpack batch results
    batch_mol_ids, batch_nodes, batch_coords, batch_adjs, batch_targets = batch_results

    # One-hot encode the nodes in the batch
    one_hot_nodes_batch = one_hot_encode_batch(batch_nodes)

    # Check dimensions for consistency within the batch
    for i in range(len(batch_mol_ids)):
        if not (
            len(batch_nodes[i])
            == len(batch_coords[i])
            == len(batch_adjs[i])
            == one_hot_nodes_batch[i].shape[0]
        ):
            print(
                f"Error at molecule index {batch_mol_ids[i]} in batch {batch_num + 1}"
            )
            break

    # Save the batch data into a pickle file
    batch_file = os.path.join(data_dir, f"batch_{batch_num + 1}.pkl")
    with open(batch_file, "wb") as file:
        pickle.dump(
            {
                "mol_ids": batch_mol_ids,
                "nodes": batch_nodes,
                "one_hot_nodes": one_hot_nodes_batch,
                "coords": batch_coords,
                "adjs": batch_adjs,
                "targets": batch_targets,
            },
            file,
        )

    # Optionally, clear variables to free up memory
    del (
        batch_mol_ids,
        batch_nodes,
        batch_coords,
        batch_adjs,
        batch_targets,
        one_hot_nodes_batch,
    )

print("Processing complete. All batches have been saved.")

# %%
import os
import pickle

data_dir = "processed_batches"
batch_files = sorted(
    [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pkl")]
)

all_mol_ids = []
all_nodes = []
all_one_hot_nodes = []
all_coords = []
all_adjs = []
all_targets = []

for batch_file in batch_files:
    with open(batch_file, "rb") as file:
        batch_data = pickle.load(file)
        all_mol_ids.extend(batch_data["mol_ids"])
        all_nodes.extend(batch_data["nodes"])
        all_one_hot_nodes.extend(batch_data["one_hot_nodes"])
        all_coords.extend(batch_data["coords"])
        all_adjs.extend(batch_data["adjs"])
        all_targets.extend(batch_data["targets"])

# Regenerate one-hot encoding to ensure consistency
all_one_hot_nodes = one_hot_encode_batch(all_nodes)

# Save the combined data into a single pickle file
combined_file = "pcqm4m_v2_processed.pkl"
with open(combined_file, "wb") as file:
    pickle.dump(
        {
            "mol_ids": all_mol_ids,
            "nodes": all_nodes,
            "one_hot_nodes": all_one_hot_nodes,
            "coords": all_coords,
            "adjs": all_adjs,
            "targets": all_targets,
        },
        file,
    )

print(f"Total molecules processed: {len(all_mol_ids)}")
