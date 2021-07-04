import numpy as np
import scipy.sparse as sp
import torch
from utils import normalize_adj, normalize_features, unpicklefile

path="./data/cora/"
dataset="cora"
"""Load citation network dataset (cora only for now)"""
print('Loading {} dataset...'.format(dataset))

idx_features_labels = unpicklefile('idx_features_y')
features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
labels = idx_features_labels[:, -1]

# build graph
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
idx_map = {j: i for i, j in enumerate(idx)}
edges_unordered = unpicklefile('edge_list')
edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

# build symmetric adjacency matrix
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

features = normalize_features(features)
adj = normalize_adj(adj + sp.eye(adj.shape[0]))

idx_train = range(140)
idx_val = range(200, 500)
idx_test = range(500, 1500)

adj = torch.FloatTensor(np.array(adj.todense()))
features = torch.FloatTensor(np.array(features.todense()))
labels = torch.LongTensor(np.where(labels)[1])

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

print(idx_features_labels)
print(edges)
idx_features_labels
edges