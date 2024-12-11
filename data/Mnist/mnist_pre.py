# %%
import torch
import pickle
from torch_geometric.datasets import MNISTSuperpixels

# Load the MNIST Superpixels dataset from PyG
dataset = MNISTSuperpixels(root="./mnist_superpixels")
dataset = dataset[:100]
# %%
# Create lists to hold the data
node_features = []
adjs = []
targets = []

# Process the dataset and extract the necessary components
for data in dataset:
    node_features.append(data.x)
    adjs.append(
        torch.sparse_coo_tensor(
            data.edge_index,
            torch.ones(data.edge_index.shape[1]),
            size=(data.num_nodes, data.num_nodes),
        ).to_dense()
    )
    targets.append(data.y)

# Save the data to a pickle file
data_to_save = {"node_feature": node_features, "adjs": adjs, "targets": targets}

# Save the data as a pickle file
with open("mnist_superpixels_data.pkl", "wb") as f:
    pickle.dump(data_to_save, f)

print("MNIST Superpixels dataset has been saved to 'mnist_superpixels_data.pkl'.")
