import numpy as np
import scipy.sparse as sp
import torch
import pickle
from sklearn.model_selection import train_test_split


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/", dataset="covid"):
    path = "./data/"
    dataset = "covid"
    """Load citation network dataset (cora """
    print('Loading {} dataset...'.format(dataset))
    feature_path = str(path + dataset + '/idx_features_y')
    idx_features_labels = unpicklefile(feature_path)
    features = idx_features_labels[:, 1:-1].astype('float64')
    labels = np.array(idx_features_labels[:, -1]).astype('float64')
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    idx_path = str(path + dataset + '/edge_list')
    edges_unordered = unpicklefile(idx_path)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idxs = np.array(range(len(features)))
    idx_train, idx_rest = train_test_split(idxs, test_size=0.4)
    idx_val, idx_test = train_test_split(idx_rest, test_size=0.5)
    # idx_train = range(200)
    # idx_val = range(200, 290)
    # idx_test = range(290, 379)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1), dtype='float')
    r_inv = rowsum
    r_inv[np.nonzero(rowsum)] = np.power(rowsum[np.nonzero(rowsum)], -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def picklefile(tofile, content):
  with open(tofile, 'wb') as f:
    pickle.dump(content, f)

def unpicklefile(fromfile):
  with open(fromfile, 'rb') as f:
    unpickled = pickle.load(f)
  return unpickled

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)