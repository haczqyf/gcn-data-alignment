import networkx as nx
import pandas as pd

from alignment.measures.randomizations import rdm_graph, rdm_feature
from alignment.utils import preprocess_feature, preprocess_adj



class Ingredients:
    """
    GCN ingredients (features, graph and ground truth) class.

    Properties:
        X(ndarray): feature matrix
        A(ndarray): graph adjacency matrix
        Y(ndarray): ground truth assignment matrix
        nodelist(list): list of node ids corresponding to the order of nodes in A
        G(NetworkX graph): NetworkX graph of A
    """
    def __init__(
        self,
        X,
        A,
        Y,
        nodelist,
        G,
    ):  
        
        self.X = X
        self.A = A
        self.Y = Y
        self.G = G
        self.nodelist = nodelist

    def _get_X_rdm(self, p):
        """Get randomized feature matrix with percentage p (p>0)."""

        return rdm_feature(self.X, percent=p)

    def _get_A_rdm(self, p):
        """Get randomized graph adjacency matrix with percentage p (p>0)."""

        return rdm_graph(self.G, nodelist=self.nodelist, percent=p)

    def get_X_gcn(self, p):
        """Wrapper for _get_X_rdm with percentage p (0<=p<=100)."""

        if p == 0:
            return preprocess_feature(self.X)
        else:
            return preprocess_feature(self._get_X_rdm(p))

        return rdm_feature(self.X, percent=p)

    def get_A_gcn(self, p):
        """Wrapper for _get_A_rdm with percentage p (0<=p<=100)."""

        if p == 0:
            return preprocess_adj(self.A)
        else:
            return preprocess_adj(self._get_A_rdm(p))

        return rdm_feature(self.X, percent=p)

    def get_Y_gcn(self):
        """Get ground truth assignment matrix."""

        return self.Y
