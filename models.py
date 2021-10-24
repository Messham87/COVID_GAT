import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, GraphConvolution

class OneLayerGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(OneLayerGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=True)
        self.lin1 = nn.Linear(nhid * nheads, 379)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.out_att(x, adj))
        # x = torch.flatten(x)
        x = self.lin1(x)
        return F.relu(x)

class TwoLayerGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(TwoLayerGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=True)
        self.attentions2 = [GraphAttentionLayer(nhid * nheads, int(nhid/2), dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)
        self.out_att2 = GraphAttentionLayer(int((nhid/2) * nheads), int((nhid/2) * nheads), dropout=dropout, alpha=alpha, concat=True)
        self.lin1 = nn.Linear(int((nhid/2) * nheads), 379)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.out_att(x, adj))
        x = torch.cat([att(x, adj) for att in self.attentions2], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.out_att2(x, adj))
        # x = torch.flatten(x)
        x = self.lin1(x)
        return F.relu(x)

class ThreeLayerGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(ThreeLayerGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=True)
        self.attentions2 = [GraphAttentionLayer(nhid * nheads, int(nhid/2), dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)
        self.out_att2 = GraphAttentionLayer(int((nhid/2) * nheads), int((nhid/2) * nheads), dropout=dropout, alpha=alpha, concat=True)
        self.lin1 = nn.Linear(int((nhid/2) * nheads) * 379, 379)
        self.attentions3 = [GraphAttentionLayer(int(nhid * nheads/2), int(nhid / 4), dropout=dropout, alpha=alpha, concat=True)
                            for _ in
                            range(nheads)]
        for i, attention in enumerate(self.attentions3):
            self.add_module('attention2_{}'.format(i), attention)
        self.out_att3 = GraphAttentionLayer(int((nhid / 4) * nheads), int((nhid / 4) * nheads), dropout=dropout,
                                            alpha=alpha, concat=True)
        self.lin1 = nn.Linear(int((nhid / 4) * nheads), 379)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.out_att(x, adj))
        x = torch.cat([att(x, adj) for att in self.attentions2], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.out_att2(x, adj))
        x = torch.cat([att(x, adj) for att in self.attentions3], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.out_att3(x, adj))
        # x = torch.flatten(x)
        x = self.lin1(x)
        return F.relu(x)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        """Kipf and Welling basic GCN"""
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = torch.sigmoid(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.relu(x)

class MLP(nn.Module):
    def __init__(self, nfeat, dropout):
        """Simple MLP"""
        super(MLP, self).__init__()
        self.dropout = dropout
        self.lin1 = nn.Linear(276, 12112)
        self.lin2 = nn.Linear(12112, 6056)
        self.lin3 = nn.Linear(6056, 3028)
        self.lin4 = nn.Linear(3028, 379)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = self.lin3(x)
        x = torch.sigmoid(x)
        x = self.lin4(x)
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
        self.lin1 = nn.Linear(nhid*nheads, 12112)
        self.lin2 = nn.Linear(12112, 6056)
        self.lin3 = nn.Linear(6056, 3028)
        self.lin4 = nn.Linear(3028, 379)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.out_att(x, adj))
        # x = torch.flatten(x)
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = self.lin3(x)
        x = torch.sigmoid(x)
        x = self.lin4(x)
        return F.relu(x)

class TwoLayerGATMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(TwoLayerGATMLP, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=True)
        self.attentions2 = [GraphAttentionLayer(nhid * nheads, int(nhid/2), dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)
        self.out_att2 = GraphAttentionLayer(int((nhid/2) * nheads), int((nhid/2) * nheads), dropout=dropout, alpha=alpha, concat=True)
        self.lin1 = nn.Linear(int((nhid/2) * nheads), 379)
        self.lin2 = nn.Linear(12112, 6056)
        self.lin3 = nn.Linear(6056, 3028)
        self.lin4 = nn.Linear(3028, 379)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.out_att(x, adj))
        x = torch.cat([att(x, adj) for att in self.attentions2], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.out_att2(x, adj))
        # x = torch.flatten(x)
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = self.lin3(x)
        x = torch.sigmoid(x)
        x = self.lin4(x)
        return F.relu(x)

class ThreeLayerGATMLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(ThreeLayerGATMLP, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=True)
        self.attentions2 = [GraphAttentionLayer(nhid * nheads, int(nhid/2), dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)
        self.out_att2 = GraphAttentionLayer(int((nhid/2) * nheads), int((nhid/2) * nheads), dropout=dropout, alpha=alpha, concat=True)
        self.lin1 = nn.Linear(int((nhid/2) * nheads) * 379, 379)
        self.attentions3 = [GraphAttentionLayer(int(nhid * nheads/2), int(nhid / 4), dropout=dropout, alpha=alpha, concat=True)
                            for _ in
                            range(nheads)]
        for i, attention in enumerate(self.attentions3):
            self.add_module('attention2_{}'.format(i), attention)
        self.out_att3 = GraphAttentionLayer(int((nhid / 4) * nheads), int((nhid / 4) * nheads), dropout=dropout,
                                            alpha=alpha, concat=True)
        self.lin1 = nn.Linear(int((nhid / 2) * nheads), 379)
        self.lin2 = nn.Linear(12112, 6056)
        self.lin3 = nn.Linear(6056, 3028)
        self.lin4 = nn.Linear(3028, 379)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.out_att(x, adj))
        x = torch.cat([att(x, adj) for att in self.attentions2], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.out_att2(x, adj))
        x = torch.cat([att(x, adj) for att in self.attentions3], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.out_att3(x, adj))
        # x = torch.flatten(x)
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = self.lin3(x)
        x = torch.sigmoid(x)
        x = self.lin4(x)
        return F.relu(x)