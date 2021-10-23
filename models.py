import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        # self.attentions2 = [GraphAttentionLayer(nfeat * nhid, nhid, dropout=dropout, alpha=alpha, concat=False) for
        # _ in range(nheads)] for i, attention2 in enumerate(self.attentions2): self.add_module('attention2_{
        # }'.format(i), attention2) self.out_att2 = GraphAttentionLayer(nfeat*nhid, nclass, dropout=dropout,
        # alpha=alpha, concat=False)
        self.lin1 = nn.Linear(12128, 6056)
        self.lin2 = nn.Linear(6056, 3028)
        self.lin3 = nn.Linear(3028, 379)
        #self.lin4 = nn.Linear(379, 379)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions2], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.flatten(self.out_att(x, adj))
        # print(x.shape)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        # x = self.lin4(x)
        return F.relu(x)