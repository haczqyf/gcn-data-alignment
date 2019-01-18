import networkx as nx
import pandas as pd

from alignment.measures.randomizations import rdm_graph, rdm_feature
from alignment.utils import preprocess_feature, preprocess_adj



class Ingredients:
    """
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

    def get_X_rdm(self, p):
        """
        """
        return rdm_feature(self.X, percent=p)

    def get_A_rdm(self, p):
        """
        """
        return rdm_graph(self.G, nodelist=self.nodelist, percent=p)

    def get_X_gcn(self, p):
        """
        """
        if p == 0:
            return preprocess_feature(self.X)
        else:
            return preprocess_feature(self.get_X_rdm(p))

        return rdm_feature(self.X, percent=p)

    def get_A_gcn(self, p):
        """
        """
        if p == 0:
            return preprocess_adj(self.A)
        else:
            return preprocess_adj(self.get_A_rdm(p))

        return rdm_feature(self.X, percent=p)

    def get_Y_gcn(self):
        """
        """
        return self.Y
