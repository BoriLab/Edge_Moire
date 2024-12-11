# %%
from utils.dataset import MyDataset
import pickle


# %%
class MnistDataset(MyDataset):
    def __init__(
        self,
        path="./dataset/mnist/mnist_small.pkl",
        evaluation_size=0.1,
        test_size=0.1,
        batch_size=128,
    ):
        self.data = pickle.load(open(path, "rb"))
        super().__init__(
            self.data["node_feature"],
            self.data["adjs"],
            self.data["targets"],
            evaluation_size=evaluation_size,
            test_size=test_size,
            batch_size=batch_size,
        )
