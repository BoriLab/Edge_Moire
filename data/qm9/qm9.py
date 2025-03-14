from utils.dataset import MyDataset
import pickle

class QM9Dataset(MyDataset):
    def __init__(
        self,
        path="/home/bioscience/dev/SJ_paper/Mn_moire2/data/qm9/qm9_data_with_edges.pkl",
        evaluation_size=0.1,
        test_size=0.1,
        batch_size=128,
        seed=42
    ):
        data = pickle.load(open(path, "rb"))
        self.data = data  # 이 줄을 추가합니다.

        super().__init__(
            data["one_hot_nodes"],
            data["adjs"],
            data["edge_indices"],
            data["edge_attrs"],
            data["targets"],
            evaluation_size=evaluation_size,
            test_size=test_size,
            batch_size=batch_size,
            seed=seed
        )

        # edge_attr_dim 설정
        self.edge_attr_dim = data["edge_attrs"][0].shape[1]
        print("Asasas")
        print(self.edge_attr_dim)