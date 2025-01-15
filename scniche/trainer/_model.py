# The source code from CMGEC (https://github.com/wangemm/CMGEC-TMM-2021) was reused in this project.
# We obtained formal written permission from the original authors to reuse and adapt their code.
# This project is licensed under the GPL-3.0 License.


import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import dgl
from dgl.nn.pytorch import GraphConv


class MGAE(nn.Module):
    def __init__(self, input_size, hidden_size_v, hidden_size, views, batch_size):
        super(MGAE, self).__init__()
        self.views = views

        # dynamically create GCN layers for each view
        self.layers_per_view = nn.ModuleList(
            [self._create_view_layers(input_size[i], hidden_size_v) for i in range(views)])

        # create layers for multi-view fusion
        self.layer_m = self._create_multi_view_layers(hidden_size_v[-1], hidden_size)

        self.featfusion = FeatureFusion(views=self.views, size=hidden_size_v[-1])
        self.gfn = GFN(input_size=batch_size, hidden_size=int(batch_size / 2))
        self.decoder = InnerProductDecoder(activation=lambda x: x)

    def _create_view_layers(self, in_dim, hidden_dims_v):
        layers = []
        if len(hidden_dims_v) >= 2:
            layers.append(GraphConv(in_feats=in_dim, out_feats=hidden_dims_v[0], activation=F.relu))
            for j in range(1, len(hidden_dims_v)):
                activation = F.relu if j != len(hidden_dims_v) - 1 else lambda x: x
                layers.append(
                    GraphConv(in_feats=hidden_dims_v[j - 1], out_feats=hidden_dims_v[j], activation=activation))
        else:
            layers.append(GraphConv(in_feats=in_dim, out_feats=hidden_dims_v[0], activation=lambda x: x))
        return nn.ModuleList(layers)

    def _create_multi_view_layers(self, last_hidden_v, hidden_dims):
        layers = [GraphConv(in_feats=last_hidden_v, out_feats=hidden_dims[0], activation=F.relu)]
        for i in range(1, len(hidden_dims)):
            activation = F.relu if i != len(hidden_dims) - 1 else lambda x: x
            layers.append(GraphConv(in_feats=hidden_dims[i - 1], out_feats=hidden_dims[i], activation=activation))
        return nn.ModuleList(layers)

    def consensus_graph(self, graphs, device):
        adj = graphs[0].ndata['adj']
        for graph in graphs[1:]:
            adj = torch.add(adj, graph.ndata['adj'])

        adj = adj.to_dense()
        fused_adj = self.gfn(adj)

        # # normalization
        fused_adj = torch.clamp(fused_adj, 0, 1)
        fused_adj = torch.round(fused_adj + 0.1)

        # build symmetric adjacency matrix
        adj_np = fused_adj.detach().cpu().numpy()
        adj_np += adj_np.T
        adj_sparse = sp.csr_matrix(adj_np)
        g = dgl.from_scipy(adj_sparse, device=device)

        return fused_adj, g

    def forward(self, graphs, data, device):
        feats = [self._apply_gcn_layers(layer, graphs[i], data[i]) for i, layer in enumerate(self.layers_per_view)]
        feat_fusion = self.featfusion(*feats)
        adj_r, g = self.consensus_graph(graphs, device)

        for conv in self.layer_m:
            feat_fusion = conv(dgl.add_self_loop(g), feat_fusion)

        adj_rec = {i: self.decoder(feat_fusion) for i in range(self.views)}

        return adj_r, adj_rec, feat_fusion

    def _apply_gcn_layers(self, layers, graph, features):
        for conv in layers:
            features = conv(graph, features)
        return features


class FeatureFusion(nn.Module):
    def __init__(self, views, activation=torch.relu, dropout=0.1, size=64):
        super(FeatureFusion, self).__init__()
        self.views = views
        self.dropout = dropout
        self.activation = activation
        self.weights = nn.ParameterList([Parameter(torch.FloatTensor(size, size)) for _ in range(self.views)])
        for weight in self.weights:
            torch.nn.init.xavier_uniform_(weight)

    def forward(self, *features):
        z = sum(F.dropout(f, self.dropout) @ w for f, w in zip(features, self.weights))
        z = F.softmax(z, dim=1)
        z = self.activation(z)
        return z


class InnerProductDecoder(nn.Module):
    def __init__(self, activation=torch.sigmoid, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        adj = self.activation(torch.mm(z, z.t()))
        return adj


class GFN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GFN, self).__init__()
        self.gfn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        out = self.gfn(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, latent_dim=120):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        self.discriminator = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.discriminator(z)
        return out

