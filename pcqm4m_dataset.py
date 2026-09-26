import torch

from torch_geometric.data import Data
from torch_geometric.data import Dataset

from ogb.lsc import PCQM4Mv2Dataset


class PCQM4MPyGDataset(Dataset):
    """
    PyG wrapper for an official PCQM4Mv2 split.

    A preloaded PCQM4Mv2Dataset can be supplied through ``dataset`` so
    that train and validation wrappers share one underlying dataset
    instead of loading two complete copies into CPU memory.
    """

    def __init__(
        self,
        root="data/pcqm4m",
        split="train",
        dataset=None,
    ):
        super().__init__()

        if dataset is None:
            dataset = PCQM4Mv2Dataset(root=root)

        self.dataset = dataset

        split_idx = self.dataset.get_idx_split()

        if split not in split_idx:
            raise ValueError(
                f"Unknown split: {split}. "
                f"Available splits: {list(split_idx.keys())}"
            )

        self.indices = split_idx[split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        if torch.is_tensor(real_idx):
            real_idx = int(real_idx.item())
        else:
            real_idx = int(real_idx)

        graph, y = self.dataset[real_idx]

        x = torch.from_numpy(
            graph["node_feat"]
        ).long()

        edge_index = torch.from_numpy(
            graph["edge_index"]
        ).long()

        edge_attr = torch.from_numpy(
            graph["edge_feat"]
        ).long()

        y = torch.tensor(
            [y],
            dtype=torch.float,
        )

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=x.size(0),
        )
