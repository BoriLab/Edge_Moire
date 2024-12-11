# %%
from utils.dataset import MyDataset
import pickle


# %%
class QM9Dataset(MyDataset):
    def __init__(
        self, path="./dataset/qm9/qm9_data.pkl", evaluation_size=0.1, test_size=0.1, batch_size=128
    ):
        data = pickle.load(open(path, "rb"))

        super().__init__(
            data["one_hot_nodes"],
            data["adjs"],
            data["targets"],
            evaluation_size=evaluation_size,
            test_size=test_size,
            batch_size=batch_size
        )
