from torch_geometric.data import Batch, Data, Dataset  # , InMemoryDataset
import torch
from collections import defaultdict
import copy

class GEOMLabelDataset(Dataset):

    def get(self, idx: int):
        data = self.data[idx].clone()
        # print(data)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def len(self) -> int:
        return len(self.data)

    def __init__(self, data=None, label=None, transform=None):
        super().__init__()
        self.data = data
        self.label = label
        self.transform = transform
        self.atom_types = self._atom_types()
        self.edge_types = self._edge_types()

    def __getitem__(self, idx):

        data = self.data[idx].clone()
        label = self.label[idx]
        # print(data)
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.data)

    def _atom_types(self):
        """All atom types."""
        atom_types = set()
        for graph in self.data:
            atom_types.update(graph.atom_type.tolist())
        return sorted(atom_types)

    def _edge_types(self):
        """All edge types."""
        edge_types = set()
        for graph in self.data:
            edge_types.update(graph.edge_type.tolist())
        return sorted(edge_types)

