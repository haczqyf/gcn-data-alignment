import collections
import random

import pandas as pd
import networkx as nx
import numpy as np
import scipy.sparse as sp

def rdm_feature(X, percent):
    """Swap randomly rows in feature matrix with percentage p (p>0)."""

    X_l = X.tolist()
    n = len(X_l)

    X_indeX_l = list(range(0,n))

    sample_index = sorted(random.sample(X_indeX_l, int(n*percent/100)))
    # To actually copy the list
    sample_index_init = sample_index[:]

    random.shuffle(sample_index)
    # To actually copy the list
    sample_index_shuffled = sample_index[:]

    X_l_shuffled = []
    for i in range(0,n):
        if i not in sample_index_init:
            X_l_shuffled.append(X_l[i])
        else:
            related_index_shuffled = sample_index_shuffled[sample_index_init.index(i)]
            X_l_shuffled.append(X_l[related_index_shuffled])

    num_diff = 0
    for i,j in zip(X_l,X_l_shuffled):
        if (i != j):
            num_diff = num_diff + 1

    num_diff_rate = num_diff / len(X_l)

    X_l_shuffled = np.array(X_l_shuffled,dtype=np.float32)

    return X_l_shuffled
    
def rdm_graph(G, nodelist, percent):
    """Reshuffle graph edges with percentage p (p>0)."""
    
    n = len((G.edges()))

    # Sort each edge
    edgelist = list(G.edges())
    edgelist = [(int(i),int(j)) for (i,j) in edgelist]

    # Sample a subset of edges from edgelist with the rate p
    sample = random.sample(list(G.edges()), int(n*percent/100))
    sample = [(int(i),int(j)) for (i,j) in sample]
    sample_sorted = sorted(sample, key=lambda tup: (tup[0],tup[1]))

    unchanged_edgelist = list(set(edgelist).difference(set(sample)))

    node1_list, node2_list = zip(*sample_sorted)
    node_list = list(node1_list + node2_list)
    counter=collections.Counter(node_list)
    node_dict = dict(counter)

    G_sample_random = nx.configuration_model(list(node_dict.values()))

    node_label_mapping = {v: k for v, k in enumerate(list(node_dict.keys()))}
    random_edges = list(G_sample_random.edges())
    random_edges_after_mapping = [(node_label_mapping[i],node_label_mapping[j]) for (i,j) in random_edges]

    random_edges_after_mapping_new = []
    for (i,j) in random_edges_after_mapping:
        if i > j:
            m = i
            i = j
            j = m
        random_edges_after_mapping_new.append((i,j))

    random_edges_after_mapping_new_sorted = sorted(random_edges_after_mapping_new, key=lambda tup: (tup[0],tup[1]))
    new_edgelist = list(unchanged_edgelist + random_edges_after_mapping_new_sorted)
    new_edgelist_sorted = sorted(new_edgelist, key=lambda tup: (tup[0],tup[1]))

    G_new = nx.MultiGraph()
    G_new.add_edges_from(new_edgelist_sorted)

    G_new = nx.Graph(G_new)
    nb_multi_edges = len(new_edgelist_sorted) - len(G_new.edges())

    G_new.remove_edges_from(list(nx.selfloop_edges(G_new)))
    nb_self_loops = len(new_edgelist_sorted) - len(G_new.edges()) - nb_multi_edges
    
    return np.array(nx.adjacency_matrix(G_new, nodelist=nodelist).toarray(),dtype=np.float32)


