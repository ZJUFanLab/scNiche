import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import dgl
from dgl.nn.pytorch import GraphConv


class GAE(nn.Module):
    def __init__(self, input_size, hidden_size_v, hidden_size, views, batch_size):
        super(GAE, self).__init__()
        self.views = views

        # layers: view 1
        if len(hidden_size_v) >= 2:
            layers1 = [GraphConv(in_feats=input_size[0], out_feats=hidden_size_v[0], activation=F.relu)]
            for j in range(1, len(hidden_size_v)):
                if j != len(hidden_size_v) - 1:
                    layers1.append(
                        GraphConv(in_feats=hidden_size_v[j - 1], out_feats=hidden_size_v[j], activation=F.relu)
                    )
                else:
                    layers1.append(
                        GraphConv(in_feats=hidden_size_v[j - 1], out_feats=hidden_size_v[j], activation=lambda x: x)
                    )
        else:
            layers1 = [GraphConv(in_feats=input_size[0], out_feats=hidden_size_v[0], activation=lambda x: x)]

        # layers: view 2
        if len(hidden_size_v) >= 2:
            layers2 = [GraphConv(in_feats=input_size[1], out_feats=hidden_size_v[0], activation=F.relu)]
            for j in range(1, len(hidden_size_v)):
                if j != len(hidden_size_v) - 1:
                    layers2.append(
                        GraphConv(in_feats=hidden_size_v[j - 1], out_feats=hidden_size_v[j], activation=F.relu)
                    )
                else:
                    layers2.append(
                        GraphConv(in_feats=hidden_size_v[j - 1], out_feats=hidden_size_v[j], activation=lambda x: x)
                    )
        else:
            layers2 = [GraphConv(in_feats=input_size[1], out_feats=hidden_size_v[0], activation=lambda x: x)]

        # layers: view 3
        if len(hidden_size_v) >= 2:
            layers3 = [GraphConv(in_feats=input_size[2], out_feats=hidden_size_v[0], activation=F.relu)]
            for j in range(1, len(hidden_size_v)):
                if j != len(hidden_size_v) - 1:
                    layers3.append(
                        GraphConv(in_feats=hidden_size_v[j - 1], out_feats=hidden_size_v[j], activation=F.relu)
                    )
                else:
                    layers3.append(
                        GraphConv(in_feats=hidden_size_v[j - 1], out_feats=hidden_size_v[j], activation=lambda x: x)
                    )
        else:
            layers3 = [GraphConv(in_feats=input_size[2], out_feats=hidden_size_v[0], activation=lambda x: x)]

        # layers: view multi
        if len(hidden_size) >= 2:
            layers_m = [GraphConv(in_feats=int(hidden_size_v[-1]), out_feats=hidden_size[0], activation=F.relu)]
            for i in range(1, len(hidden_size)):
                if i != len(hidden_size) - 1:
                    layers_m.append(
                        GraphConv(in_feats=hidden_size[i - 1], out_feats=hidden_size[i], activation=F.relu)
                    )
                else:
                    layers_m.append(
                        GraphConv(in_feats=hidden_size[i - 1], out_feats=hidden_size[i], activation=lambda x: x)
                    )
        else:
            layers_m = [
                GraphConv(in_feats=int(hidden_size_v[-1] * views), out_feats=hidden_size[0], activation=lambda x: x)]

        self.layer_m = nn.ModuleList(layers_m)
        self.layer1 = nn.ModuleList(layers1)
        self.layer2 = nn.ModuleList(layers2)
        self.layer3 = nn.ModuleList(layers3)
        self.featfusion = FeatureFusion(size=hidden_size_v[-1])
        self.gfn = GFN(input_size=batch_size, hidden_size=int(batch_size / 2))
        self.decoder = InnerProductDecoder(activation=lambda x: x)

    def consensus_graph(self, g1, g2, g3, device):
        adjin = torch.add(g1.ndata['adj'], g2.ndata['adj'])
        adjin = torch.add(adjin, g3.ndata['adj'])
        adjin = adjin.to_dense()
        adj_r = self.gfn(adjin)

        # # normalization
        adj_p = torch.clamp(adj_r, 0, 1)
        adj_p = torch.round(adj_p + 0.1)

        # build symmetric adjacency matrix
        adj_pn = adj_p.detach().cpu().numpy()
        adj_pn += adj_pn.T
        adj_pn = sp.csr_matrix(adj_pn)
        # g = dgl.from_scipy(adj_pn, device = 'cuda:0')
        g = dgl.from_scipy(adj_pn, device=device)

        return adj_p, g

    def forward(self, g1, g2, g3, data1, data2, data3, device):
        feat1 = data1
        feat2 = data2
        feat3 = data3

        for conv in self.layer1:
            feat1 = conv(g1, feat1)
        for conv in self.layer2:
            feat2 = conv(g2, feat2)
        for conv in self.layer3:
            feat3 = conv(g3, feat3)

        feat_fusion = self.featfusion(feat1, feat2, feat3)

        adj_r, g = self.consensus_graph(g1, g2, g3, device)

        for conv in self.layer_m:
            feat_fusion = conv(dgl.add_self_loop(g), feat_fusion)

        adj_rec = {}
        for i in range(self.views):
            adj_rec[i] = self.decoder(feat_fusion)

        return adj_r, adj_rec, feat_fusion


class FeatureFusion(nn.Module):
    def __init__(self, activation=torch.relu, dropout=0.1, size=64):
        super(FeatureFusion, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.weight1 = Parameter(torch.FloatTensor(size, size))
        self.weight2 = Parameter(torch.FloatTensor(size, size))
        self.weight3 = Parameter(torch.FloatTensor(size, size))
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight3)

    def forward(self, z1, z2, z3):
        z1 = F.dropout(z1, self.dropout)
        z2 = F.dropout(z2, self.dropout)
        z3 = F.dropout(z3, self.dropout)
        z = torch.mm(z1, self.weight1) + torch.mm(z2, self.weight2) + torch.mm(z3, self.weight3)
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
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
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












