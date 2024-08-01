from anndata import AnnData
from tqdm import tqdm
import pandas as pd
import numpy as np
from ._model import *
from ._utils import shuffling
from typing import Optional


class Runner:
    def __init__(
            self,
            adata: AnnData,
            hidden_size_v: Optional[list] = None,
            hidden_size: Optional[list] = None,
            device: str = 'cuda:0',
            verbose: bool = True
    ):
        self.views = 3
        self.adata = adata
        self.g1 = self.adata.uns['g1']
        self.g2 = self.adata.uns['g2']
        self.g3 = self.adata.uns['g3']

        self.feat1 = self.g1.ndata['feat']
        self.feat2 = self.g2.ndata['feat']
        self.feat3 = self.g3.ndata['feat']
        self.in_feats = [self.feat1.shape[1], self.feat2.shape[1], self.feat3.shape[1]]

        self.adj1 = self.g1.ndata['adj'].to_dense()
        self.adj2 = self.g2.ndata['adj'].to_dense()
        self.adj3 = self.g3.ndata['adj'].to_dense()

        self.mik = np.hstack((self.g1.ndata['mik'], self.g2.ndata['mik'], self.g3.ndata['mik']))

        self.edges = self.g1.number_of_edges() + self.g2.number_of_edges() + self.g3.number_of_edges()
        self.device = device
        self.hidden_size_v = hidden_size_v
        self.hidden_size = hidden_size
        self.verbose = verbose

        if self.hidden_size_v is None:
            self.hidden_size_v = [10]

        if self.hidden_size is None:
            self.hidden_size = [32, 10]

        if self.verbose:
            print("-------Prepare training...")
            print("Views: {}".format(self.views))
            print("Views-1 DataSize: {} * {}".format(self.feat1.shape[0], self.feat1.shape[1]))
            print("Views-2 DataSize: {} * {}".format(self.feat2.shape[0], self.feat2.shape[1]))
            print("Views-3 DataSize: {} * {}".format(self.feat3.shape[0], self.feat3.shape[1]))
            print("Views-1 Graph Edges: {}".format(self.g1.number_of_edges()))
            print("Views-2 Graph Edges: {}".format(self.g2.number_of_edges()))
            print("Views-3 Graph Edges: {}".format(self.g3.number_of_edges()))
            print("Mutual Information Matrix Size for training: {}".format(self.mik.shape))

    def fit(self, lr: Optional[float] = 0.01, epochs: Optional[int] = 200,):

        # to device
        self.feat1 = self.feat1.to(self.device)
        self.feat2 = self.feat2.to(self.device)
        self.feat3 = self.feat3.to(self.device)
        self.g1 = self.g1.to(self.device)
        self.g2 = self.g2.to(self.device)
        self.g3 = self.g3.to(self.device)
        self.adj1 = self.adj1.to(self.device)
        self.adj2 = self.adj2.to(self.device)
        self.adj3 = self.adj3.to(self.device)

        # model
        self.model = GAE(self.in_feats, self.hidden_size_v, self.hidden_size, self.views, self.adj1.shape[0])
        self.model_d = Discriminator(latent_dim=self.hidden_size[-1])
        self.model = self.model.to(self.device)
        self.model_d = self.model_d.to(self.device)

        # optimizer
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        # loss
        pos_weight = torch.Tensor(
            [float(self.g1.adjacency_matrix().to_dense().shape[0] ** 2 - self.edges / 2) / self.edges * 2]
        )
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
        criterion_m = torch.nn.MSELoss().to(self.device)

        if self.verbose:
            print("-------Start training...")

        self.model.train()
        loss_all = []
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            adj_r, adj_logits, z = self.model.forward(self.g1, self.g2, self.g3,
                                                      self.feat1, self.feat2, self.feat3, self.device)
            loss_gre = (criterion_m(adj_r, self.adj1) + criterion_m(adj_r, self.adj2) + criterion_m(
                adj_r, self.adj3)) / self.views

            loss_rec = (criterion(adj_logits[0], self.adj1) + criterion(adj_logits[1], self.adj2) + criterion(
                adj_logits[2], self.adj3)) / self.views

            global_info_loss = 0
            for i in range(self.mik.shape[1]):
                z_shuffle = shuffling(z, latent=self.hidden_size[-1], device=self.device)
                z_z_shuffle = torch.cat((z, z_shuffle), 1)
                z_z_shuffle_scores = self.model_d(z_z_shuffle)
                z_z = torch.cat((z, z[self.mik[:, i]]), 1)
                z_z_scores = self.model_d(z_z)
                global_info_loss += - torch.mean(
                    torch.log(z_z_scores + 1e-6) + torch.log(1 - z_z_shuffle_scores + 1e-6)
                )
            loss = loss_gre + loss_rec + global_info_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_description('Train Epoch: {}'.format(epoch + 1))
            pbar.set_postfix(loss=f"{loss:.4f}")
            loss_all.append(loss.data.cpu().numpy())

        if self.verbose:
            print("Training done.")

        self.model.eval()
        _, _, z = self.model.forward(self.g1, self.g2, self.g3, self.feat1, self.feat2, self.feat3, self.device)

        self.adata.uns['loss'] = loss_all
        self.adata.obsm['X_scniche'] = z.data.cpu().numpy()
        return self.adata


class Runner_batch:
    def __init__(
            self,
            adata: AnnData,
            hidden_size_v: Optional[list] = None,
            hidden_size: Optional[list] = None,
            device: str = 'cuda:0',
            verbose: bool = True
    ):
        self.views = 3
        self.adata = adata
        self.dataloader = self.adata.uns['dataloader']

        self.feat1 = self.adata.obsm['X_cn_norm']
        self.feat2 = self.adata.obsm['X_data']
        self.feat3 = self.adata.obsm['X_data_nbr']
        self.in_feats = [self.feat1.shape[1], self.feat2.shape[1], self.feat3.shape[1]]

        self.device = device
        self.hidden_size_v = hidden_size_v
        self.hidden_size = hidden_size
        self.verbose = verbose

        if self.hidden_size_v is None:
            self.hidden_size_v = [10]

        if self.hidden_size is None:
            self.hidden_size = [32, 10]

        if self.verbose:
            print("-------Prepare training...")
            print("Views: {}".format(self.views))
            print("Views-1 DataSize: {} * {}".format(self.feat1.shape[0], self.feat1.shape[1]))
            print("Views-2 DataSize: {} * {}".format(self.feat2.shape[0], self.feat2.shape[1]))
            print("Views-3 DataSize: {} * {}".format(self.feat3.shape[0], self.feat3.shape[1]))
            print("Batch size: {}".format(len(self.dataloader)))

    def fit(self, lr: Optional[float] = 0.01, epochs: Optional[int] = 200,):

        # model
        batch_size = len(self.adata.uns['batch_idx'][0])
        self.model = GAE(self.in_feats, self.hidden_size_v, self.hidden_size, self.views, batch_size)
        self.model_d = Discriminator(latent_dim=self.hidden_size[-1])
        self.model = self.model.to(self.device)
        self.model_d = self.model_d.to(self.device)

        # optimizer
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        if self.verbose:
            print("-------Start training...")

        self.model.train()
        loss_all = []
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            batch_loss = 0
            for batch in self.dataloader:
                g1 = batch[0]
                g2 = batch[1]
                g3 = batch[2]

                feat1 = g1.ndata['feat']
                feat2 = g2.ndata['feat']
                feat3 = g3.ndata['feat']

                adj1 = g1.ndata['adj'].to_dense()
                adj2 = g2.ndata['adj'].to_dense()
                adj3 = g3.ndata['adj'].to_dense()

                mik = np.hstack((g1.ndata['mik'], g2.ndata['mik'], g3.ndata['mik']))
                edges = g1.number_of_edges() + g2.number_of_edges() + g3.number_of_edges()

                # loss
                pos_weight = torch.Tensor(
                    [float(g1.adjacency_matrix().to_dense().shape[0] ** 2 - edges / 2) / edges * 2]
                )
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.device)
                criterion_m = torch.nn.MSELoss().to(self.device)

                # to device
                feat1 = feat1.to(self.device)
                feat2 = feat2.to(self.device)
                feat3 = feat3.to(self.device)
                g1 = g1.to(self.device)
                g2 = g2.to(self.device)
                g3 = g3.to(self.device)
                adj1 = adj1.to(self.device)
                adj2 = adj2.to(self.device)
                adj3 = adj3.to(self.device)

                adj_r, adj_logits, z = self.model.forward(g1, g2, g3, feat1, feat2, feat3, self.device)
                loss_gre = (criterion_m(adj_r, adj1) + criterion_m(adj_r, adj2) + criterion_m(
                    adj_r, adj3)) / self.views
                loss_rec = (criterion(adj_logits[0], adj1) + criterion(adj_logits[1], adj2) + criterion(
                    adj_logits[2], adj3)) / self.views

                global_info_loss = 0
                for i in range(mik.shape[1]):
                    z_shuffle = shuffling(z, latent=self.hidden_size[-1], device=self.device)
                    z_z_shuffle = torch.cat((z, z_shuffle), 1)
                    z_z_shuffle_scores = self.model_d(z_z_shuffle)
                    z_z = torch.cat((z, z[mik[:, i]]), 1)
                    z_z_scores = self.model_d(z_z)
                    global_info_loss += - torch.mean(
                        torch.log(z_z_scores + 1e-6) + torch.log(1 - z_z_shuffle_scores + 1e-6)
                    )

                loss = loss_gre + loss_rec + global_info_loss
                optim.zero_grad()
                loss.backward()
                optim.step()

                batch_loss += loss.item()

            loss_all.append(batch_loss)
            pbar.set_description('Train Epoch: {}'.format(epoch))
            pbar.set_postfix(loss=f"{batch_loss:.4f}")

        if self.verbose:
            print("Training done.")

        self.model.eval()
        emb = []
        for batch in tqdm(self.dataloader):
            g1 = batch[0]
            g2 = batch[1]
            g3 = batch[2]

            feat1 = g1.ndata['feat']
            feat2 = g2.ndata['feat']
            feat3 = g3.ndata['feat']

            g1 = g1.to(self.device)
            g2 = g2.to(self.device)
            g3 = g3.to(self.device)
            feat1 = feat1.to(self.device)
            feat2 = feat2.to(self.device)
            feat3 = feat3.to(self.device)

            _, _, z = self.model.forward(g1, g2, g3, feat1, feat2, feat3, self.device)
            emb.append(list(z.data.cpu().numpy()))

        emb = np.array(emb)
        emb = pd.DataFrame(np.reshape(emb, (-1, emb.shape[2])))

        idx = np.array(self.adata.uns['batch_idx']).flatten().tolist()
        emb.index = idx
        emb = emb[~emb.index.duplicated()]
        emb.index = self.adata.obs_names[emb.index]
        emb = emb.reindex(self.adata.obs_names)

        self.adata.uns['loss'] = loss_all
        self.adata.obsm['X_scniche'] = np.array(emb)
        return self.adata














