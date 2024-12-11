# %%
from utils.dataset import MyDataset
import pickle

# %%
class PCQM4Mv2Dataset(MyDataset):
    def __init__(
        self,
        path="./dataset/pcqm/pcqm4mv2_data.pkl",
        evaluation_size=0.1,
        test_size=0.1,
        batch_size=128,
    ):
        self.data = pickle.load(open(path, "rb"))

        # drop certain indices
        drop_indices = [480108, 512973]
        for drop in drop_indices:
            del self.data["adjs"][drop]
            del self.data["one_hot_nodes"][drop]
            del self.data["targets"][drop]

        super().__init__(
            self.data["one_hot_nodes"],
            self.data["adjs"],
            self.data["targets"],
            evaluation_size=evaluation_size,
            test_size=test_size,
            batch_size=batch_size,
        )
