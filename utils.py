import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import math
import random
import torch
from sklearn.metrics import f1_score


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str, label_rate=None, inductive=False, loop=True, norm=None):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    if dataset_str == 'reddit':
        adj, features, y_train, y_val, y_test, idx_train, idx_val, idx_test = loadRedditFromNPZ("data/")
        labels = np.zeros(adj.shape[0])
        labels[idx_train]  = y_train
        labels[idx_val]  = y_val
        labels[idx_test]  = y_test
        adj = adj + adj.T
        # features = features.toarray()
        features = (features-features[idx_train].mean(axis=0))/features[idx_train].std(axis=0)
    
    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
    
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)
    
        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended
    
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.convert_matrix.to_scipy_sparse_matrix(nx.from_dict_of_lists(graph)).astype(np.float64)
    
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        
        if label_rate:
            idx_temp = list(range(len(labels)))
            random.shuffle(idx_temp) 
            idx_train = idx_temp[:int(label_rate * len(labels))]
            idx_train = sorted(idx_train)
            idx_val = idx_temp[int(label_rate * len(labels)) : int(label_rate * len(labels)) + 500]
            idx_val = sorted(idx_val)
            idx_test = idx_temp[-len(ty):]
        else:
            idx_train = range(len(y))  
            idx_val = range(len(y), len(y)+500)
            idx_test = test_idx_range.tolist()
        
        features = normalize(features)
        features = features.toarray()
        labels_temp = np.zeros(labels.shape[0])
        labels_temp[np.where(labels)[0]] = np.where(labels)[1]
        labels = labels_temp
    
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    if norm is not None:
        norm_dict = {'sym': mat_normalize, 'rw': normalize, 'max': maxeig_normalize}
        norm_fn = norm_dict[norm]
        
    if inductive:
        mask = np.zeros(adj.shape[0])
        mask[idx_train] = 1.
        train_mask = sp.diags(mask)
        adj_train = train_mask.dot(adj).dot(train_mask)
        mask[idx_val] = 1.
        val_mask = sp.diags(mask)
        adj_val = val_mask.dot(adj).dot(val_mask)
        if loop:
            adj_train = adj_train + sp.eye(adj.shape[0])
            adj_val = adj_val + sp.eye(adj.shape[0])
            adj = adj + sp.eye(adj.shape[0])
        if norm is not None:
            adj_train = norm_fn(adj)
            adj_val = norm_fn(adj)
            adj = norm_fn(adj)
        adj_train = sparse_mx_to_torch_sparse_tensor(adj_train)
        adj_val = sparse_mx_to_torch_sparse_tensor(adj_val)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        if loop:
            adj = adj + sp.eye(adj.shape[0])
        if norm is not None:
            adj = norm_fn(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adj_train = None
        adj_val = None

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj_train, adj_val, adj, features, labels, idx_train, idx_val, idx_test # adj(2708,2708), features(2708,1433), labels(2708)


def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def mat_normalize(mx):
    """Normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def maxeig_normalize(mx):
    """Normalize sparse matrix"""
    maxeig = eigsh(mx, 1, which='LA')[0][0]
    mx /= maxeig
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col)).astype(int)).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    # macro = f1_score(labels, preds, average='macro')
    return micro