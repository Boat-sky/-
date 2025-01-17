# sampled_dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.sparse as sp


class SampledDataset(Dataset):
    def __init__(self, original_dataset, adj_matrix, n_users, n_items, sample_ratio=0.1):
        """
        Initialize the sampled dataset.
        :param original_dataset: The original dataset object.
        :param adj_matrix: The full adjacency matrix (scipy sparse matrix).
        :param n_users: Number of users.
        :param n_items: Number of items.
        :param sample_ratio: Fraction of nodes or edges to sample.
        """
        self.original_dataset = original_dataset
        self.adj_matrix = adj_matrix
        self.n_users = n_users
        self.n_items = n_items
        self.sample_ratio = sample_ratio

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        """
        Dynamically sample a subgraph and fetch the corresponding data.
        :param index: Index of the data sample to fetch.
        :return: A dictionary containing the sampled adjacency matrix and original data.
        """
        # Fetch original data
        data = self.original_dataset[index]

        # Sample a subset of nodes
        num_nodes = self.n_users + self.n_items
        sample_size = int(num_nodes * self.sample_ratio)
        sampled_nodes = np.random.choice(num_nodes, size=sample_size, replace=False)

        # Create a subgraph based on sampled nodes
        rows, cols = self.adj_matrix.nonzero()
        mask = np.isin(rows, sampled_nodes) & np.isin(cols, sampled_nodes)
        sub_rows = rows[mask]
        sub_cols = cols[mask]
        sub_data = self.adj_matrix.data[mask]
        sampled_adj = sp.csr_matrix((sub_data, (sub_rows, sub_cols)), shape=self.adj_matrix.shape)

        # Convert to torch sparse tensor
        sampled_adj_tensor = self.csr_to_torch_sparse(sampled_adj)

        # Include sampled adjacency matrix in the returned data
        data['sub_adj'] = sampled_adj_tensor
        return data

    @staticmethod
    def csr_to_torch_sparse(csr_mat):
        """
        Convert a scipy sparse matrix to a PyTorch sparse tensor.
        :param csr_mat: Scipy sparse matrix in CSR format.
        :return: PyTorch sparse tensor.
        """
        coo = csr_mat.tocoo()
        indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        shape = coo.shape
        return torch.sparse_coo_tensor(indices, values, torch.Size(shape)).coalesce()
