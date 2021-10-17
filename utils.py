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
    train_feature_path = str(path + dataset + '/train_idx_features_y')
    train_idx_features_labels = unpicklefile(train_feature_path)
    train_features = train_idx_features_labels[:, 1:-1].astype('float64')
    train_labels = np.array(train_idx_features_labels[:, -1]).astype('float64')
    valid_feature_path = str(path + dataset + '/valid_idx_features_y')
    valid_idx_features_labels = unpicklefile(valid_feature_path)
    valid_features = valid_idx_features_labels[:, 1:-1].astype('float64')
    valid_labels = np.array(valid_idx_features_labels[:, -1]).astype('float64')
    test_feature_path = str(path + dataset + '/test_idx_features_y')
    test_idx_features_labels = unpicklefile(test_feature_path)
    test_features = test_idx_features_labels[:, 1:-1].astype('float64')
    test_labels = np.array(test_idx_features_labels[:, -1]).astype('float64')
    # build graph
    idx = np.array(train_idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    idx_path = str(path + dataset + '/edge_list')
    edges_unordered = unpicklefile(idx_path)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(train_labels.shape[0], train_labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    adj = torch.FloatTensor(np.array(adj.todense()))
    train_features = torch.FloatTensor(train_features)
    valid_features = torch.FloatTensor(valid_features)
    test_features = torch.FloatTensor(test_features)
    train_labels = torch.LongTensor(train_labels)
    valid_labels = torch.LongTensor(valid_labels)
    test_labels = torch.LongTensor(test_labels)

    return adj, train_features, train_labels, valid_features, valid_labels, test_features, test_labels

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