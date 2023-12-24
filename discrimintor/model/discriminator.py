import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from discrimintor import utils, layers
from discrimintor.utils import GaussianFourierProjection


class SDE(torch.nn.Module):

    def __init__(self, config):
        super(SDE, self).__init__()
        self.config = config
        self.anneal_power = self.config.train.anneal_power
        self.hidden_dim = self.config.model.hidden_dim
        self.order = self.config.model.order
        self.noise_type = self.config.model.noise_type

        # self.gpus = gpus
        # self.device = config.train.device
        # torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')            
        # self.config = config

        self.node_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.edge_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.input_mlp = layers.MultiLayerPerceptron(1, [self.hidden_dim, self.hidden_dim],
                                                     activation=self.config.model.mlp_act)
        # self.output_mlp = layers.MultiLayerPerceptron(2 * self.hidden_dim, \
        #                                               [self.hidden_dim, self.hidden_dim // 2, 1],
        #                                               activation=self.config.model.mlp_act)
        self.output_mlp = torch.nn.Linear(2 * self.hidden_dim, 1)
        self.model = layers.GraphIsomorphismNetwork(hidden_dim=self.hidden_dim, \
                                                    num_convs=self.config.model.num_convs, \
                                                    activation=self.config.model.gnn_act, \
                                                    readout="sum", short_cut=self.config.model.short_cut, \
                                                    concat_hidden=self.config.model.concat_hidden)
        self.sigmoid = nn.Sigmoid()
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_noise_level)), dtype=torch.float32)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False)  # (num_noise_level)

        # time step embedding for continous SDE
        assert self.config.scheme.time_continuous
        if self.config.scheme.time_continuous:
            self.t_embed = nn.Sequential(GaussianFourierProjection(embed_dim=self.hidden_dim),
                                         nn.Linear(self.hidden_dim, self.hidden_dim))
            self.dense1 = nn.Linear(self.hidden_dim, 1)

    # 计算扰动核的方差
    def marginal_prob_std(self, t, sigma, device='cuda'):
        """Compute standard deviation of $p_{0t}(x(t) | x(0))$.

        Args:
          t: A vector of time steps.
          sigma: The $\sigma$ in our SDE.

        Returns:
          The standard deviation.
        """
        # t = torch.tensor(t, device=device)
        return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))

    @torch.no_grad()
    # extend the edge on the fly, second order: angle, third order: dihedral
    def extend_graph(self, data: Data, order=3):

        def binarize(x):
            return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

        def get_higher_order_adj_matrix(adj, order):
            """
            Args:
                adj:        (N, N)
                type_mat:   (N, N)
            """
            adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                        binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

            for i in range(2, order + 1):
                adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
            order_mat = torch.zeros_like(adj)

            for i in range(1, order + 1):
                order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

            return order_mat

        num_types = len(utils.BOND_TYPES)

        N = data.num_nodes
        adj = to_dense_adj(data.edge_index).squeeze(0)
        adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

        type_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).squeeze(0)  # (N, N)
        type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order))
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder

        new_edge_index, new_edge_type = dense_to_sparse(type_new)
        _, edge_order = dense_to_sparse(adj_order)

        data.bond_edge_index = data.edge_index  # Save original edges
        data.edge_index, data.edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N)  # modify data
        edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N)  # modify data
        data.is_bond = (data.edge_type < num_types)
        assert (data.edge_index == edge_index_1).all()

        return data

    @torch.no_grad()
    def convert_score_d(self, score_d, pos, edge_index, edge_length):
        dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])  # (num_edge, 3)
        score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0)

        return score_pos

    # decouple the noise generator
    def noise_generator(self, data, noise_type):
        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]
        if noise_type == 'symmetry':
            num_nodes = scatter_add(torch.ones(data.num_nodes, dtype=torch.long, device=self.Device),
                                    node2graph)  # (num_graph)
            num_cum_nodes = num_nodes.cumsum(0)  # (num_graph)
            '''
            节点偏移和边偏移是为了将不同的图中的节点和边映射到一个连续的编号空间中，方便计算无向边的唯一标识符。
            例如，如果有两个图，第一个图有3个节点，第二个图有4个节点，那么节点偏移就是[0, 3]，表示第一个图的节点编号为0, 1, 2，第二个图的节点编号为3, 4, 5, 6。
            同理，如果第一个图有6条边，第二个图有8条边，那么边偏移就是[0, 6]，表示第一个图的边编号为0, 1, …, 5，第二个图的边编号为6, 7, …, 13。
            '''
            node_offset = num_cum_nodes - num_nodes  # (num_graph)
            edge_offset = node_offset[edge2graph]  # (num_edge)

            num_nodes_square = num_nodes ** 2  # (num_graph)
            num_nodes_square_cumsum = num_nodes_square.cumsum(-1)  # (num_graph)
            edge_start = num_nodes_square_cumsum - num_nodes_square  # (num_graph)
            edge_start = edge_start[edge2graph]

            all_len = num_nodes_square_cumsum[-1]

            node_index = data.edge_index.t() - edge_offset.unsqueeze(-1)
            # node_in, node_out = node_index.t()
            node_large = node_index.max(dim=-1)[0]
            node_small = node_index.min(dim=-1)[0]
            undirected_edge_id = node_large * (node_large + 1) + node_small + edge_start
            symm_noise = torch.Tensor(all_len.detach().cpu().numpy(), device='cpu').normal_()
            symm_noise = symm_noise.to(self.Device)
            d_noise = symm_noise[undirected_edge_id].unsqueeze(-1)  # (num_edge, 1)
        elif noise_type == 'rand':
            d = data.edge_length
            d_noise = torch.randn_like(d)
        return d_noise

    # 计算边的长度
    @torch.no_grad()
    def get_distance(self, data: Data):
        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1)  # (num_edge, 1)
        data.edge_length = d
        return data

    def forward(self, data):
        """
        Input:
            data: torch geometric batched data object
        Output:
            loss
        """
        # a workaround to get the current device, we assume all tensors in a model are on the same device.
        self.device = self.sigmas.device
        data = self.extend_graph(data, self.order)
        data = self.get_distance(data)

        assert data.edge_index.size(1) == data.edge_length.size(0)
        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]

        # sample noise level
        noise_level = torch.randint(0, self.sigmas.size(0), (data.num_graphs,), device=self.device)  # (num_graph)

        noise_level = torch.zeros_like(noise_level)
        used_sigmas = self.sigmas[noise_level]  # (num_graph)
        used_sigmas = used_sigmas[edge2graph].unsqueeze(-1)  # (num_edge, 1)

        # perturb
        d = data.edge_length  # (num_edge, 1)

        if self.noise_type == 'symmetry':
            num_nodes = scatter_add(torch.ones(data.num_nodes, dtype=torch.long, device=self.device),
                                    node2graph)  # (num_graph)
            num_cum_nodes = num_nodes.cumsum(0)  # (num_graph)
            node_offset = num_cum_nodes - num_nodes  # (num_graph)
            edge_offset = node_offset[edge2graph]  # (num_edge)

            num_nodes_square = num_nodes ** 2  # (num_graph)
            num_nodes_square_cumsum = num_nodes_square.cumsum(-1)  # (num_graph)
            edge_start = num_nodes_square_cumsum - num_nodes_square  # (num_graph)
            edge_start = edge_start[edge2graph]

            all_len = num_nodes_square_cumsum[-1]

            node_index = data.edge_index.t() - edge_offset.unsqueeze(-1)
            # node_in, node_out = node_index.t()
            node_large = node_index.max(dim=-1)[0]
            node_small = node_index.min(dim=-1)[0]
            undirected_edge_id = node_large * (node_large + 1) + node_small + edge_start

            symm_noise = torch.zeros(all_len, device=self.device, dtype=torch.float).normal_()
            d_noise = symm_noise[undirected_edge_id].unsqueeze(-1)  # (num_edge, 1)

        elif self.noise_type == 'rand':
            d_noise = torch.randn_like(d)
        else:
            raise NotImplementedError('noise type must in [distance_symm, distance_rand]')
        assert d_noise.shape == d.shape
        perturbed_d = d + d_noise * used_sigmas
        # tmp = torch.nonzero(torch.abs(used_sigmas - 10) < 0.5)
        # if tmp.numel() > 0:
        #     print('有接近于10 的数字')
        # perturbed_d = torch.clamp(perturbed_d, min=0.1, max=float('inf'))    # distances must be greater than 0

        # get target, origin_d minus perturbed_d
        target = -1 / (used_sigmas ** 2) * (perturbed_d - d)  # (num_edge, 1)

        # estimate scores
        node_attr = self.node_emb(data.atom_type)  # (num_node, hidden)
        edge_attr = self.edge_emb(data.edge_type)  # (num_edge, hidden)
        d_emb = self.input_mlp(perturbed_d)  # (num_edge, hidden)
        edge_attr = d_emb * edge_attr  # (num_edge, hidden)

        output = self.model(data, node_attr, edge_attr)
        h_row, h_col = output["node_feature"][data.edge_index[0]], output["node_feature"][
            data.edge_index[1]]  # (num_edge, hidden)

        distance_feature = torch.cat([h_row * h_col, edge_attr], dim=-1)  # (num_edge, 2 * hidden)
        scores = self.output_mlp(distance_feature)  # (num_edge, 1)
        scores = scores * (1. / used_sigmas)  # f_theta_sigma(x) =  f_theta(x) / sigma, (num_edge, 1)

        # target = target.view(-1)  # (num_edge)
        # scores = scores.view(-1)  # (num_edge)
        # loss = 0.5 * ((scores - target) ** 2) * (used_sigmas.squeeze(-1) ** self.anneal_power)  # (num_edge)
        # loss = scatter_add(loss, edge2graph)  # (num_graph)
        # return loss
        scores = scores.view(-1)

        scores = scatter_add(scores, edge2graph)  # (num_graph)


        # 使用 torch.bincount 统计每个图的边数
        edge_num_of_graphs = torch.bincount(edge2graph)
        scores = scores / edge_num_of_graphs
        # tmp = torch.nonzero(torch.abs(used_sigmas - 10) < 0.000001)
        # if tmp.numel() > 0:
        #     print('有接近于10 的数字')
        #     # print(scores[tmp])
        return scores

    # @torch.no_grad()
    def judge(self, data, d, sigma):
        '''
        Args:
            data: the molecule_batched data, which provides edge_index and the information of other features
            d: input the reconsturcted data from sample process, e.g.torch.size[22234,1]
            t: input the SDE equation time with batch_size, e.g.torch.size[128]

        Returns:
            scores shape=(num_edges)
        '''
        self.Device = self.sigmas.device
        self.device = self.sigmas.device
        # data = self.extend_graph(data, self.order)  # 扩展图
        # data = self.get_distance(data)  # 计算距离
        # assert data.edge_index.size(1) == data.edge_length.size(0)
        #
        # self.device = self.sigmas.device
        # data = self.extend_graph(data, self.order)
        # data = self.get_distance(data)

        # assert data.edge_index.size(1) == data.edge_length.size(0)
        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]

        # sample noise level
        # noise_level = torch.randint(0, self.sigmas.size(0), (data.num_graphs,), device=self.device)  # (num_graph)
        # used_sigmas = self.sigmas[noise_level]  # (num_graph)
        # used_sigmas = used_sigmas[edge2graph].unsqueeze(-1)  # (num_edge, 1)
        used_sigmas = sigma
        # perturb
        # d = data.edge_length  # (num_edge, 1)

        if self.noise_type == 'symmetry':
            num_nodes = scatter_add(torch.ones(data.num_nodes, dtype=torch.long, device=self.device),
                                    node2graph)  # (num_graph)
            num_cum_nodes = num_nodes.cumsum(0)  # (num_graph)
            node_offset = num_cum_nodes - num_nodes  # (num_graph)
            edge_offset = node_offset[edge2graph]  # (num_edge)

            num_nodes_square = num_nodes ** 2  # (num_graph)
            num_nodes_square_cumsum = num_nodes_square.cumsum(-1)  # (num_graph)
            edge_start = num_nodes_square_cumsum - num_nodes_square  # (num_graph)
            edge_start = edge_start[edge2graph]

            all_len = num_nodes_square_cumsum[-1]

            node_index = data.edge_index.t() - edge_offset.unsqueeze(-1)
            # node_in, node_out = node_index.t()
            node_large = node_index.max(dim=-1)[0]
            node_small = node_index.min(dim=-1)[0]
            undirected_edge_id = node_large * (node_large + 1) + node_small + edge_start

            symm_noise = torch.zeros(all_len, device=self.device, dtype=torch.float).normal_()
            d_noise = symm_noise[undirected_edge_id].unsqueeze(-1)  # (num_edge, 1)

        elif self.noise_type == 'rand':
            d_noise = torch.randn_like(d)
        else:
            raise NotImplementedError('noise type must in [distance_symm, distance_rand]')
        assert d_noise.shape == d.shape
        perturbed_d = d + d_noise * used_sigmas
        # perturbed_d = torch.clamp(perturbed_d, min=0.1, max=float('inf'))    # distances must be greater than 0

        # get target, origin_d minus perturbed_d
        target = -1 / (used_sigmas ** 2) * (perturbed_d - d)  # (num_edge, 1)

        # estimate scores
        node_attr = self.node_emb(data.atom_type)  # (num_node, hidden)
        edge_attr = self.edge_emb(data.edge_type)  # (num_edge, hidden)
        d_emb = self.input_mlp(perturbed_d)  # (num_edge, hidden)
        edge_attr = d_emb * edge_attr  # (num_edge, hidden)

        output = self.model(data, node_attr, edge_attr)
        h_row, h_col = output["node_feature"][data.edge_index[0]], output["node_feature"][
            data.edge_index[1]]  # (num_edge, hidden)

        distance_feature = torch.cat([h_row * h_col, edge_attr], dim=-1)  # (num_edge, 2 * hidden)
        scores = self.output_mlp(distance_feature)  # (num_edge, 1)
        scores = scores * (1. / used_sigmas)  # f_theta_sigma(x) =  f_theta(x) / sigma, (num_edge, 1)

        # target = target.view(-1)  # (num_edge)
        # scores = scores.view(-1)  # (num_edge)
        # loss = 0.5 * ((scores - target) ** 2) * (used_sigmas.squeeze(-1) ** self.anneal_power)  # (num_edge)
        # loss = scatter_add(loss, edge2graph)  # (num_graph)
        # return loss
        scores = scores.view(-1)

        scores = scatter_add(scores, edge2graph)  # (num_graph)


        # 使用 torch.bincount 统计每个图的边数
        edge_num_of_graphs = torch.bincount(edge2graph)
        scores = scores / edge_num_of_graphs
        scores = self.sigmoid(scores)
        return scores
