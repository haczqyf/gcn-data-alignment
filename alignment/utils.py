import scipy.sparse as sp
import numpy as np
from sklearn import preprocessing

def load_data(dataset="cora"):
    """Load citation network dataset"""
    import networkx as nx
    print('Loading {} dataset...'.format(dataset))

    idx_feature_labels = np.genfromtxt("data/{}/{}.content".format(dataset, dataset), dtype=np.dtype(str))
    feature = sp.csr_matrix(idx_feature_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_feature_labels[:, -1])

    # build graph
    idx = np.array(idx_feature_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("data/{}/{}.cites".format(dataset, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    G = nx.read_edgelist("data/{}/{}.cites".format(dataset, dataset), nodetype=int, delimiter='\t')

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], feature.shape[1]))

    return idx, G, feature.toarray(), adj.toarray(), labels

def encode_onehot(labels):
    # classes = set(labels)
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def preprocess_feature(feature):
    """Row-normalize feature matrix"""
    feature = preprocessing.normalize(feature, norm='l1', axis=1)
    return feature

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized.toarray()

# helper function
def find_repeat(source, elmt): # The source may be a list or string.
    elmt_index = []
    s_index = 0;e_index = len(source)
    while(s_index < e_index):
        try:
            temp = source.index(elmt, s_index, e_index)
            elmt_index.append(temp)
            s_index = temp + 1
        except ValueError:
            break
            
    return elmt_index