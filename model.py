import torch
import torch.nn as nn
from torch.nn.functional import normalize
import torch_geometric
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_sparse import matmul, spmm
from torch_geometric.utils import degree

class LightGCNConv(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class LightGCN(MessagePassing):
    def __init__(self, num_users, num_items, hidden_dim, num_layers):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.users_emb = nn.Embedding(self.num_users, self.hidden_dim)
        self.items_emb = nn.Embedding(self.num_items, self.hidden_dim)

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

        self.convs = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.convs.append(LightGCNConv())

    def forward(self, edge_index):
        x0 = torch.cat([self.users_emb.weight, self.items_emb.weight], dim=0)
        xs = [x0]
        xi = x0

        for conv in self.convs:
            xi = conv.forward(xi, edge_index)
            xs.append(xi)

        xs = torch.stack(xs, dim=1)
        x_final = torch.mean(xs, dim=1)
        users_emb, items_emb = torch.split(x_final, [self.num_users, self.num_items], dim=0)

        return users_emb, self.users_emb.weight, items_emb, self.items_emb.weight