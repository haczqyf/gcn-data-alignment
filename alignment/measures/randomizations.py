import collections
import random

import pandas as pd
import networkx as nx
import numpy as np
import scipy.sparse as sp

def rdm_feature(X, percent):

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
    
def get_network_sorted_edges(G):
    """
    """
    edges = list(G.edges())
    edges = [(int(i),int(j)) for (i,j) in edges]
    edges_sorted = []
    for (i,j) in edges:
        if i > j:
            m = i
            i = j
            j = m
        edges_sorted.append((i,j))
    edges_sorted = sorted(edges_sorted, key=lambda tup: (tup[0],tup[1]))

    return edges_sorted

def compare_two_networks_edges(G1, G2):
    """
    How many edges in G1 are not in G2
    where G1 and G2 have no multi-edges or self-loops
    """

    edges1 = get_network_sorted_edges(G1)
    edges2 = get_network_sorted_edges(G2)

    nb_edges_G1 = len(G1.edges())
    nb_edges_only_G1 = len(list(set(edges1).difference(set(edges2))))
    percent_edges_only_G1 = nb_edges_only_G1 / nb_edges_G1

    return percent_edges_only_G1

def rdm_graph(G, nodelist, percent):
    # G = nx.read_edgelist(path, delimiter='\t')
    n = len((G.edges()))
    # print("number of nodes: {}, number of edges: {}".format(len(G.nodes()),len(G.edges())))

    # Sort each edge
    edgelist = list(G.edges())
    edgelist = [(int(i),int(j)) for (i,j) in edgelist]


    # Sample a subset of edges from edgelist with the rate p
    sample = random.sample(list(G.edges()), int(n*percent/100))
    sample = [(int(i),int(j)) for (i,j) in sample]
    sample_sorted = sorted(sample, key=lambda tup: (tup[0],tup[1]))

    unchanged_edgelist = list(set(edgelist).difference(set(sample)))

    # print("number of total edges: {},".format(len(edgelist)),
    #       "number of sampled edges: {},".format(len(sample)),
    #       "number of unchaged edges: {}".format(len(unchanged_edgelist)))

    node1_list, node2_list = zip(*sample_sorted)
    node_list = list(node1_list + node2_list)
    counter=collections.Counter(node_list)
    node_dict = dict(counter)

    G_sample_random = nx.configuration_model(list(node_dict.values()))
    # print("randomised edges: {}".format(len(G_sample_random.edges())))

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
    # print("number of edges of network after randomisation including self-loops and multiple-edges: {}".format(len(new_edgelist_sorted)))

    G_new = nx.MultiGraph()
    G_new.add_edges_from(new_edgelist_sorted)

    G_new = nx.Graph(G_new)
    nb_multi_edges = len(new_edgelist_sorted) - len(G_new.edges())
    # print("number of multiple-edges: {}".format(nb_multi_edges))

    G_new.remove_edges_from(list(nx.selfloop_edges(G_new)))
    nb_self_loops = len(new_edgelist_sorted) - len(G_new.edges()) - nb_multi_edges
    # print("number of self-loops: {}".format(nb_self_loops))

    # print("Final number of edges of randomised network: {}".format(len(G_new.edges())))

    percent_edges_changed = compare_two_networks_edges(G,G_new)
    # print("Final percent of edges changed of randomised network: {:.4f}".format(percent_edges_changed))

    # return G_new, len(G_new.edges()), n, nb_multi_edges, nb_self_loops, percent_edges_changed
    return np.array(nx.adjacency_matrix(G_new, nodelist=nodelist).toarray(),dtype=np.float32)


