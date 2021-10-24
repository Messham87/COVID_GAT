import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer

class OneLayerGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(OneLayerGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.lin1 = nn.Linear(12128, 379)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.flatten(self.out_att(x, adj))
        x = self.lin1(x)
        return F.relu(x)

class GATMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GATMLP, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=True)
        # self.attentions2 = [GraphAttentionLayer(nfeat * nhid, nhid, dropout=dropout, alpha=alpha, concat=False) for
        # _ in range(nheads)] for i, attention2 in enumerate(self.attentions2): self.add_module('attention2_{
        # }'.format(i), attention2) self.out_att2 = GraphAttentionLayer(nfeat*nhid, nclass, dropout=dropout,
        # alpha=alpha, concat=False)
        self.lin1 = nn.Linear(nhid*nheads*379, 12112)
        self.lin2 = nn.Linear(12112, 6056)
        self.lin3 = nn.Linear(6056, 3028)
        self.lin4 = nn.Linear(3028, 379)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.flatten(self.out_att(x, adj))
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = self.lin3(x)
        x = torch.sigmoid(x)
        x = self.lin4(x)
        # x = self.lin4(x)
        return F.relu(x)