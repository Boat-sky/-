import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import GeneralModel
import scipy.sparse as sp
import numpy as np
import networkx as nx
import multiprocessing as mp
import random
from .Params import args
from torch import sparse_coo_tensor 

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class GFormer(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['latdim', 'gcn_layer', 'pnn_layer', 'head', 'keepRate', 'reRate', 'addRate', 'sub']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--latdim', default=32, type=int, help='embedding size')
        #parser.add_argument('--latdim', default=2, type=int, help='embedding size')
        parser.add_argument('--gcn_layer', default=2, type=int, help='number of gcn layers')
        parser.add_argument('--pnn_layer', default=1, type=int, help='number of graph transformer layers')
        parser.add_argument('--head', default=4, type=int, help='number of heads in attention')
        parser.add_argument('--keepRate', default=0.9, type=float, help='ratio of nodes to keep')
        parser.add_argument('--keepRate2', default=0.7, type=float, help='ratio of nodes to keep')
        parser.add_argument('--reRate', default=0.8, type=float, help='ratio of nodes to keep')
        parser.add_argument('--addRate', default=0.01,type=float, help='ratio of nodes to keep')
        parser.add_argument('--sub', default=0.1, type=float, help='sub maxtrix')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.latdim = args.latdim
        self.gcn_layer = args.gcn_layer
        self.pnn_layer = args.pnn_layer
        self.keepRate = args.keepRate
        self.reRate = args.reRate
        self.addRate = args.addRate
        self.sub = args.sub
        self.head = args.head
        self._define_params()
        self.n_users = corpus.n_users  # Pass n_users from corpus
        self.n_items = corpus.n_items  # Pass n_items from corpus
        self.adj_matrix = self._build_adj_matrix(corpus.data_df['train'])  # Build from train data
        #self.apply(self.init_weights)
        

    def _define_params(self):
        self.uEmbeds = nn.Parameter(torch.empty(self.user_num, self.latdim))
        self.iEmbeds = nn.Parameter(torch.empty(self.item_num, self.latdim))
        self.gcnLayers = nn.Sequential(*[GCNLayer(self.latdim) for i in range(self.gcn_layer)])
        self.pnnLayers = nn.Sequential(*[PNNLayer() for i in range(self.pnn_layer)])
        self.gcnLayer = GCNLayer(self.latdim)
        self.gtLayers = GTLayer(self.latdim, self.head).cuda()
        self.bn = nn.BatchNorm1d(self.latdim)

    def _build_adj_matrix(self, train_df):
        """
        Build the adjacency matrix from training data.
        Args:
            train_df (pd.DataFrame): Training dataset with columns ['user_id', 'item_id'].
        Returns:
            sp.csr_matrix: Adjacency matrix in scipy sparse format.
        """
        rows = train_df['user_id'].values
        cols = train_df['item_id'].values + self.n_users  # Offset item indices
        data = np.ones_like(rows, dtype=np.float32)
        adj_matrix = sp.csr_matrix((data, (rows, cols)),
                                   shape=(self.n_users + self.n_items, self.n_users + self.n_items))
        return adj_matrix

    def getEgoEmbeds(self):
        return t.cat([self.uEmbeds, self.iEmbeds], axis=0)
    
    def generate_sub_adj(self):
        """
        Generate or retrieve the sub adjacency matrix.
        Returns:
            torch.sparse.FloatTensor: Sub adjacency matrix.
        """
        rows, cols = self.adj_matrix.nonzero()  # Use the adjacency matrix stored in BaseReader
        edge_count = len(rows)
        keep_count = int(edge_count * 0.1)  # Keep 10% of edges
        keep_indices = np.random.choice(edge_count, size=keep_count, replace=False)
        sub_rows = rows[keep_indices]
        sub_cols = cols[keep_indices]
        sub_data = np.ones_like(sub_rows)

        # Create a sparse adjacency matrix for the sub graph
        sub_adj_matrix = sp.csr_matrix((sub_data, (sub_rows, sub_cols)), shape=self.adj_matrix.shape)
        return self.sp_mat_to_sp_tensor(sub_adj_matrix)

    def generate_cmp_adj(self):
        """
        Generate or retrieve the comparison adjacency matrix.
        Returns:
            torch.sparse.FloatTensor: Comparison adjacency matrix with additional edges.
        """
        rows, cols = self.adj_matrix.nonzero()
        additional_edges = int(len(rows) * 0.1)  # Add 10% more edges
        add_rows = np.random.choice(self.n_users, size=additional_edges)
        add_cols = np.random.choice(self.n_items, size=additional_edges) + self.n_users

        cmp_rows = np.concatenate([rows, add_rows])
        cmp_cols = np.concatenate([cols, add_cols])
        cmp_data = np.ones_like(cmp_rows)

        # Create a sparse adjacency matrix for the comparison graph
        cmp_adj_matrix = sp.csr_matrix((cmp_data, (cmp_rows, cmp_cols)), shape=self.adj_matrix.shape)
        return self.sp_mat_to_sp_tensor(cmp_adj_matrix)

    def generate_encoder_adj(self):
        """
        Generate or retrieve the normalized adjacency matrix for the encoder.
        Returns:
            torch.sparse.FloatTensor: Normalized adjacency matrix for the encoder.
        """
        rowsum = np.array(self.adj_matrix.sum(1)).flatten()  # Compute row sums
        rowsum[rowsum == 0] = 1  # Replace zeros to prevent division by zero
        d_inv = np.power(rowsum, -0.5)  # Compute inverse square root
        d_mat_inv = sp.diags(d_inv)  # Create diagonal matrix

        # Normalize the adjacency matrix
        norm_adj_tmp = d_mat_inv.dot(self.adj_matrix)
        encoder_adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return self.sp_mat_to_sp_tensor(encoder_adj_matrix)

    def sp_mat_to_sp_tensor(self, sp_mat):
        """
        Convert a scipy sparse matrix to a PyTorch sparse tensor.
        Args:
            sp_mat (scipy.sparse.csr_matrix): Input sparse matrix.
        Returns:
            torch.sparse.FloatTensor: PyTorch sparse tensor.
        """
        coo = sp_mat.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        values = torch.from_numpy(coo.data)
        shape = torch.Size(coo.shape)
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()
    
    def forward(self, handler, is_test, sub, cmp, encoderAdj, decoderAdj=None):
        embeds = torch.cat([self.uEmbeds, self.iEmbeds], axis=0)
        embedsLst = [embeds]
        emb, _ = self.gtLayers(cmp, embeds)
        cList = [embeds, args.gtw * emb]
        emb, _ = self.gtLayers(sub, embeds)
        subList = [embeds, args.gtw * emb]

        for i, gcn in enumerate(self.gcnLayers):
            embeds = gcn(encoderAdj, embedsLst[-1])
            embeds2 = gcn(sub, embedsLst[-1])
            embeds3 = gcn(cmp, embedsLst[-1])
            subList.append(embeds2)
            embedsLst.append(embeds)
            cList.append(embeds3)

        if not is_test:
            for i, pnn in enumerate(self.pnnLayers):
                embeds = pnn(handler, embedsLst[-1])
                embedsLst.append(embeds)

        if decoderAdj is not None:
            embeds, _ = self.gtLayers(decoderAdj, embedsLst[-1])
            embedsLst.append(embeds)

        # Adjust shapes in embedsLst
        base_shape = embedsLst[0].shape
        for i, tensor in enumerate(embedsLst):
            if tensor.shape != base_shape:
                #print(f"Adjusting Tensor {i} in embedsLst from {tensor.shape} to {base_shape}")
                if tensor.shape[0] < base_shape[0]:  # Padding
                    padding = (0, 0, 0, base_shape[0] - tensor.shape[0])
                    embedsLst[i] = F.pad(tensor, padding, "constant", 0)
                elif tensor.shape[0] > base_shape[0]:  # Truncation
                    embedsLst[i] = tensor[:base_shape[0]]

        # Adjust shapes in cList
        base_shape = cList[0].shape
        for i, tensor in enumerate(cList):
            if tensor.shape != base_shape:
                #print(f"Adjusting Tensor {i} in cList from {tensor.shape} to {base_shape}")
                if tensor.shape[0] < base_shape[0]:  # Padding
                    padding = (0, 0, 0, base_shape[0] - tensor.shape[0])
                    cList[i] = F.pad(tensor, padding, "constant", 0)
                elif tensor.shape[0] > base_shape[0]:  # Truncation
                    cList[i] = tensor[:base_shape[0]]

        # Adjust shapes in subList
        base_shape = subList[0].shape
        for i, tensor in enumerate(subList):
            if tensor.shape != base_shape:
                #print(f"Adjusting Tensor {i} in subList from {tensor.shape} to {base_shape}")
                if tensor.shape[0] < base_shape[0]:  # Padding
                    padding = (0, 0, 0, base_shape[0] - tensor.shape[0])
                    subList[i] = F.pad(tensor, padding, "constant", 0)
                elif tensor.shape[0] > base_shape[0]:  # Truncation
                    subList[i] = tensor[:base_shape[0]]

        embeds = sum(embedsLst)
        cList = sum(cList)
        subList = sum(subList)

        return {
        'prediction': embeds[:self.n_users],  # Use the first part as predictions
        'item_embeddings': embeds[self.n_users:],  # Item embeddings
        'cList': cList,
        'subList': subList
    }


GFormer.GFormer = GFormer

class GCNLayer(nn.Module):
    def __init__(self, latdim):
        super(GCNLayer, self).__init__()
        self.latdim = latdim

    
    """
    def batched_sparse_mm(self, adj, embeds, batch_size):
        device = embeds.device
        indices = adj._indices()
        values = adj._values()

        adj = adj.to(dtype=torch.float32, device=device)
        embeds = embeds.to(dtype=torch.float32, device=device)
        indices = indices.to(dtype=torch.float32, device=device)
        values = values.to(dtype=torch.float32, device=device)

        #print(f"Initial adj dtype: {adj.dtype}, device: {adj.device}")
        #print(f"Initial indices dtype: {indices.dtype}, device: {indices.device}")
        #print(f"Initial values dtype: {values.dtype}, device: {values.device}")
        #print(f"Initial embeds dtype: {embeds.dtype}, device: {embeds.device}")

        result = []
        for i in range(0, adj.shape[0], batch_size):
            # Identify indices that belong to the current batch
            batch_mask = (indices[0] >= i) & (indices[0] < i + batch_size)
            batch_indices = indices[:, batch_mask]
            batch_values = values[batch_mask]

            # Adjust row indices for the batch
            batch_indices[0] -= i

            #print(f"Batch {i}-{i + batch_size}:")
            #print(f"  batch_indices dtype: {batch_indices.dtype}, shape: {batch_indices.shape}")
            #print(f"  batch_values dtype: {batch_values.dtype}, shape: {batch_values.shape}")

            # Create sparse tensor for the batch
            sub_adj = torch.sparse_coo_tensor(
                batch_indices,
                batch_values,
                size=(batch_size, adj.shape[1]),
                device=device,
                dtype=torch.float32
            ).coalesce()

            # Adjust dimensions of sub_adj and embeds if necessary
            if sub_adj.shape[1] != embeds.shape[0]:
                min_dim = min(sub_adj.shape[1], embeds.shape[0])

                # Filter out columns in sub_adj that exceed min_dim
                valid_mask = sub_adj._indices()[1] < min_dim
                new_indices = sub_adj._indices()[:, valid_mask]
                new_values = sub_adj._values()[valid_mask]
                sub_adj = torch.sparse_coo_tensor(
                    new_indices,
                    new_values,
                    size=(sub_adj.shape[0], min_dim),
                    device=device,
                    dtype=torch.float32
                )

                # Truncate embeds to min_dim
                embeds = embeds[:min_dim, :].to(dtype=torch.float32, device=device)

            #print(f"  sub_adj dtype: {sub_adj.dtype}, device: {sub_adj.device}")
            #print(f"  embeds dtype: {embeds.dtype}, device: {embeds.device}")

            sub_result = torch.sparse.mm(sub_adj.to(torch.float32), embeds.to(torch.float32))
            result.append(sub_result)

        return torch.cat(result, dim=0).to(embeds.dtype)
    """
    
    def batched_sparse_mm(self, adj, embeds, batch_size):
        device = embeds.device

        # Ensure inputs are float32
        adj = adj.to(dtype=torch.float32, device=device)
        embeds = embeds.to(dtype=torch.float32, device=device)

        indices = adj._indices().to(dtype=torch.float32, device=device)
        values = adj._values().to(dtype=torch.float32, device=device)

        # Debugging: Check data types
        assert adj.dtype == torch.float32, f"Expected adj dtype float32, got {adj.dtype}"
        assert embeds.dtype == torch.float32, f"Expected embeds dtype float32, got {embeds.dtype}"
        assert indices.dtype == torch.float32, f"Expected indices dtype float32, got {indices.dtype}"
        assert values.dtype == torch.float32, f"Expected values dtype float32, got {values.dtype}"

        result = []
        for i in range(0, adj.shape[0], batch_size):
            # Identify indices that belong to the current batch
            batch_mask = (indices[0] >= i) & (indices[0] < i + batch_size)
            batch_indices = indices[:, batch_mask]
            batch_values = values[batch_mask]

            # Adjust row indices for the batch
            batch_indices[0] -= i

            # Create sparse tensor for the batch
            sub_adj = torch.sparse_coo_tensor(
                batch_indices,
                batch_values,
                size=(batch_size, adj.shape[1]),
                device=device,
                dtype=torch.float32
            ).coalesce()

            # Adjust dimensions of sub_adj and embeds if necessary
            if sub_adj.shape[1] != embeds.shape[0]:
                min_dim = min(sub_adj.shape[1], embeds.shape[0])

                # Filter out columns in sub_adj that exceed min_dim
                valid_mask = sub_adj._indices()[1] < min_dim
                new_indices = sub_adj._indices()[:, valid_mask]
                new_values = sub_adj._values()[valid_mask]
                sub_adj = torch.sparse_coo_tensor(
                    new_indices,
                    new_values,
                    size=(sub_adj.shape[0], min_dim),
                    device=device,
                    dtype=torch.float32
                )

                # Truncate embeds to min_dim
                embeds = embeds[:min_dim, :].to(dtype=torch.float32, device=device)

            # Sparse matrix multiplication
            sub_result = torch.sparse.mm(sub_adj, embeds)
            result.append(sub_result)

        return torch.cat(result, dim=0).to(embeds.dtype)


    def forward(self, adj, embeds, batch_size=1024):
        device = embeds.device

        # Ensure `adj` is sparse
        if not adj.is_sparse:
            adj = adj.to_sparse()

        return self.batched_sparse_mm(adj.to(device), embeds, batch_size)



class PNNLayer(nn.Module):
    def __init__(self):
        super(PNNLayer, self).__init__()
        self.linear_out_position = nn.Linear(args.latdim, 1)
        self.linear_out = nn.Linear(args.latdim, args.latdim)
        self.linear_hidden = nn.Linear(2 * args.latdim, args.latdim)
        self.act = nn.ReLU()

    def _generate_anchor_set(self, embeds):
        """
        Generate the anchor_set_id dynamically based on the input embeddings.
        For simplicity, we select a subset of indices randomly as anchors.

        Args:
            embeds (torch.Tensor): Input embeddings of shape [num_users + num_items, latdim].

        Returns:
            torch.Tensor: Indices representing the anchor set.
        """
        num_anchors = min(embeds.size(0), 256)  # Select up to 256 anchors (adjustable)
        anchor_set_id = torch.randperm(embeds.size(0))[:num_anchors]  # Randomly select anchors
        return anchor_set_id

    def forward(self, handler, embeds):
        t.cuda.empty_cache()

        # Generate dynamic anchor set
        anchor_set_id = self._generate_anchor_set(embeds)  # Shape: [num_anchors]

        # Simulate distances (or calculate them appropriately)
        dists_array = torch.randn(len(anchor_set_id), embeds.size(0)).abs().to(embeds.device)  # Shape: [num_anchors, num_embeddings]

        # Get embeddings for the anchor set
        set_ids_emb = embeds[anchor_set_id]  # Shape: [num_anchors, latdim]

        # Reshape for broadcasting
        set_ids_reshape = set_ids_emb.repeat(dists_array.shape[1], 1).reshape(-1, len(set_ids_emb), args.latdim)  # Shape: [num_samples, num_anchors, latdim]
        dists_array_emb = dists_array.T.unsqueeze(2)  # Shape: [num_samples, num_anchors, 1]

        # Element-wise product
        messages = set_ids_reshape * dists_array_emb  # Shape: [num_samples, num_anchors, latdim]

        # Generate self features and align dimensions
        self_feature = embeds.unsqueeze(0).expand(messages.size(0), -1, -1)  # Shape: [num_samples, num_embeddings, latdim]

        # Adjust self_feature to match the number of anchors
        if self_feature.size(1) != messages.size(1):
            self_feature = self_feature[:, :messages.size(1), :]  # Truncate to match anchors
            # Alternatively, if expanding is intended:
            # self_feature = self_feature[:, :1, :].expand(-1, messages.size(1), -1)

        # Concatenate along the last dimension
        messages = torch.cat((messages, self_feature), dim=-1)  # Shape: [num_samples, num_anchors, 2 * latdim]

        # Pass through a linear layer
        messages = self.linear_hidden(messages).squeeze()  # Shape: [num_samples, latdim]

        # Aggregate results
        outposition1 = t.mean(messages, dim=1)  # Shape: [num_samples]

        return outposition1

class GTLayer(nn.Module):
    def __init__(self, latdim, head):
        super(GTLayer, self).__init__()
        self.latdim = latdim
        self.head = head
        self.qTrans = nn.Parameter(init(torch.empty(latdim, latdim)))
        self.kTrans = nn.Parameter(init(torch.empty(latdim, latdim)))
        self.vTrans = nn.Parameter(init(torch.empty(latdim, latdim)))


    def makeNoise(self, scores):
        noise = t.rand(scores.shape).cuda()
        noise = -t.log(-t.log(noise))
        return scores + 0.01*noise

    def adjust_latdim(self, total_elements, head):
        for latdim in range(4, total_elements, 4):  # Iterate over potential latdims
            if total_elements % (head * (latdim // head)) == 0:
                return latdim
        raise ValueError("Cannot find a valid args.latdim for the given tensor size.")
    
    def forward(self, adj, embeds, flag=False):
        if adj._nnz() == 0:  # Check if sparse matrix has no non-zero elements
            raise ValueError("Adjacency matrix is empty. Check data preprocessing.")


        device = embeds.device
        indices = adj._indices()
        rows, cols = indices[0, :].to(device), indices[1, :].to(device)

        if rows.numel() == 0 or cols.numel() == 0:
            print("Warning: Adjacency matrix has no edges. Skipping computation.")
            return torch.zeros([adj.shape[0], self.latdim], device=device), None

        rowEmbeds = embeds[rows]
        colEmbeds = embeds[cols]

        qEmbeds = (rowEmbeds @ self.qTrans).view([-1, self.head, self.latdim // self.head])
        kEmbeds = (colEmbeds @ self.kTrans).view([-1, self.head, self.latdim // self.head])
        vEmbeds = (colEmbeds @ self.vTrans).view([-1, self.head, self.latdim // self.head])

        att = t.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
        att = t.clamp(att, -10.0, 10.0)
        expAtt = t.exp(att)

        if expAtt.numel() == 0:
            print("Warning: Attention scores are empty. Skipping computation.")
            return torch.zeros([adj.shape[0], self.latdim], device=device), None

        #print(f"Shape of rows: {rows.shape}")      # Expected: [num_nonzero_elements]
        #print(f"Shape of expAtt: {expAtt.shape}")  # Expected: [num_nonzero_elements, num_heads]
        #print(f"Shape of tem: {tem.shape}")        # Expected: [adj.shape[0], num_heads]

        tem = t.zeros([adj.shape[0], args.head], device=device)
        attNorm = (tem.index_add_(0, rows, expAtt))[rows]
        
        att = expAtt / (attNorm + 1e-8)
        resEmbeds = t.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, self.latdim])
        tem = t.zeros([adj.shape[0], self.latdim], device=device, dtype=resEmbeds.dtype)
        resEmbeds = tem.index_add_(0, rows, resEmbeds)
        return resEmbeds, att

class LocalGraph(nn.Module):

    def __init__(self, gtLayer):
        super(LocalGraph, self).__init__()
        self.gt_layer = gtLayer
        self.sft = t.nn.Softmax(0)
        self.device = "cuda:0"
        self.num_users = self.n_users
        self.num_items = args.item
        self.pnn = PNNLayer().cuda()

    def makeNoise(self, scores):
        noise = t.rand(scores.shape).cuda()
        noise = -t.log(-t.log(noise))
        return scores + noise

    def sp_mat_to_sp_tensor(self, sp_mat):
        coo = sp_mat.tocoo().astype(np.float32)
        indices = t.from_numpy(np.asarray([coo.row, coo.col]))
        return t.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()

    def merge_dicts(self, dicts):
        result = {}
        for dictionary in dicts:
            result.update(dictionary)
        return result

    def single_source_shortest_path_length_range(self, graph, node_range, cutoff):  # 最短路径算法
        dists_dict = {}
        for node in node_range:
            dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
        return dists_dict

    def all_pairs_shortest_path_length_parallel(self, graph, cutoff=None, num_workers=1):
        nodes = list(graph.nodes)
        random.shuffle(nodes)
        if len(nodes) < 50:
            num_workers = int(num_workers / 4)
        elif len(nodes) < 400:
            num_workers = int(num_workers / 2)
        num_workers = 1  # windows
        pool = mp.Pool(processes=num_workers)
        results = self.single_source_shortest_path_length_range(graph, nodes, cutoff)

        output = [p.get() for p in results]
        dists_dict = self.merge_dicts(output)
        pool.close()
        pool.join()
        return dists_dict

    def precompute_dist_data(self, edge_index, num_nodes, approximate=0):
        '''
            Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
            :return:
            '''
        graph = nx.Graph()
        graph.add_edges_from(edge_index)

        n = num_nodes
        dists_dict = self.all_pairs_shortest_path_length_parallel(graph,
                                                                  cutoff=approximate if approximate > 0 else None)
        dists_array = np.zeros((n, n), dtype=np.int8)

        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist != -1:
                    dists_array[node_i, node_j] = 1 / (dist + 1)
        return dists_array

    def forward(self, adj, embeds, handler):

        embeds = self.pnn(handler, embeds)
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        tmp_rows = np.random.choice(rows.cpu(), size=[int(len(rows) * args.addRate)])
        tmp_cols = np.random.choice(cols.cpu(), size=[int(len(cols) * args.addRate)])

        add_cols = t.tensor(tmp_cols).to(self.device)
        add_rows = t.tensor(tmp_rows).to(self.device)

        newRows = t.cat([add_rows, add_cols, t.arange(self.n_users + args.item).cuda(), rows])
        newCols = t.cat([add_cols, add_rows, t.arange(self.n_users + args.item).cuda(), cols])

        #ratings_keep = np.array(t.ones_like(t.tensor(newRows.cpu())))
        ratings_keep = np.array(t.ones_like(newRows.cpu().clone().detach()))
        adj_mat = sp.csr_matrix((ratings_keep, (newRows.cpu(), newCols.cpu())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        add_adj = self.sp_mat_to_sp_tensor(adj_mat).to(self.device)

        embeds_l2, atten = self.gt_layer(add_adj, embeds)
        att_edge = t.sum(atten, dim=-1)

        return att_edge, add_adj


class RandomMaskSubgraphs(nn.Module):
    def __init__(self, num_users, num_items):
        super(RandomMaskSubgraphs, self).__init__()
        self.flag = False
        self.num_users = num_users
        self.num_items = num_items
        self.device = "cuda:0"
        self.sft = t.nn.Softmax(1)

    def normalizeAdj(self, adj):
        degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

    def sp_mat_to_sp_tensor(self, sp_mat):
        coo = sp_mat.tocoo().astype(np.float32)
        indices = t.from_numpy(np.asarray([coo.row, coo.col]))
        return t.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()

    def create_sub_adj(self, adj, att_edge, flag):
        users_up = adj._indices()[0, :]
        items_up = adj._indices()[1, :]
        if flag:
            att_edge = (np.array(att_edge.detach().cpu() + 0.001))
        else:
            att_f = att_edge
            att_f[att_f > 3] = 3
            att_edge = 1.0 / (np.exp(np.array(att_f.detach().cpu() + 1E-8)))  # 基于mlp可以去除
        att_f = att_edge / att_edge.sum()
        keep_index = np.random.choice(np.arange(len(users_up.cpu())), int(len(users_up.cpu()) * args.sub),
                                      replace=False, p=att_f)

        keep_index.sort()

        drop_edges = []
        i = 0
        j = 0
        while i < len(users_up):
            if j == len(keep_index):
                drop_edges.append(True)
                i += 1
                continue
            if i == keep_index[j]:
                drop_edges.append(False)
                j += 1
            else:
                drop_edges.append(True)
            i += 1

        rows = users_up[keep_index]
        cols = items_up[keep_index]
        rows = t.cat([t.arange(self.n_users + args.item).cuda(), rows])
        cols = t.cat([t.arange(self.n_users + args.item).cuda(), cols])

        #ratings_keep = np.array(t.ones_like(t.tensor(rows.cpu())))
        ratings_keep = np.array(t.ones_like(rows.cpu().clone().detach()))
        adj_mat = sp.csr_matrix((ratings_keep, (rows.cpu(), cols.cpu())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        encoderAdj = self.sp_mat_to_sp_tensor(adj_matrix).to(self.device)
        return encoderAdj

    def forward(self, adj, att_edge):
        users_up = adj._indices()[0, :]
        items_up = adj._indices()[1, :]

        att_f = att_edge
        att_f[att_f > 3] = 3
        att_f = 1.0 / (np.exp(np.array(att_f.detach().cpu() + 1E-8)))
        att_f1 = att_f / att_f.sum()

        keep_index = np.random.choice(np.arange(len(users_up.cpu())), int(len(users_up.cpu()) * args.keepRate),
                                          replace=False, p=att_f1)
        keep_index.sort()
        rows = users_up[keep_index]
        cols = items_up[keep_index]
        rows = t.cat([t.arange(self.n_users + args.item).cuda(), rows])
        cols = t.cat([t.arange(self.n_users + args.item).cuda(), cols])
        drop_edges = []
        i, j = 0, 0

        while i < len(users_up):
            if j == len(keep_index):
                drop_edges.append(True)
                i += 1
                continue
            if i == keep_index[j]:
                drop_edges.append(False)
                j += 1
            else:
                drop_edges.append(True)
            i += 1

        #ratings_keep = np.array(t.ones_like(t.tensor(rows.cpu())))
        ratings_keep = np.array(t.ones_like(rows.cpu().clone().detach()))
        adj_mat = sp.csr_matrix((ratings_keep, (rows.cpu(), cols.cpu())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        encoderAdj = self.sp_mat_to_sp_tensor(adj_matrix).to(self.device)


        drop_row_ids = users_up[drop_edges]
        drop_col_ids = items_up[drop_edges]

        ext_rows = np.random.choice(rows.cpu(), size=[int(len(drop_row_ids) * args.ext)])
        ext_cols = np.random.choice(cols.cpu(), size=[int(len(drop_col_ids) * args.ext)])

        ext_cols = t.tensor(ext_cols).to(self.device)
        ext_rows = t.tensor(ext_rows).to(self.device)
        #
        tmp_rows = t.cat([ext_rows, drop_row_ids])
        tmp_cols = t.cat([ext_cols, drop_col_ids])

        new_rows = np.random.choice(tmp_rows.cpu(), size=[int(adj._values().shape[0] * args.reRate)])
        new_cols = np.random.choice(tmp_cols.cpu(), size=[int(adj._values().shape[0] * args.reRate)])

        new_rows = t.tensor(new_rows).to(self.device)
        new_cols = t.tensor(new_cols).to(self.device)

        newRows = t.cat([new_rows, new_cols, t.arange(self.n_users + args.item).cuda(), rows])
        newCols = t.cat([new_cols, new_rows, t.arange(self.n_users + args.item).cuda(), cols])

        hashVal = newRows * (self.n_users + args.item) + newCols
        hashVal = t.unique(hashVal)
        newCols = hashVal % (self.n_users + args.item)
        newRows = ((hashVal - newCols) / (self.n_users + args.item)).long()

        #decoderAdj = t.sparse.FloatTensor(t.stack([newRows, newCols], dim=0), t.ones_like(newRows).cuda().float(),
        #                                  adj.shape)
        decoderAdj = t.sparse_coo_tensor(
            t.stack([newRows, newCols], dim=0), t.ones_like(newRows).cuda().float(),
                                          adj.shape
)


        sub = self.create_sub_adj(adj, att_edge, True)
        cmp = self.create_sub_adj(adj, att_edge, False)

        return encoderAdj, decoderAdj, sub, cmp



