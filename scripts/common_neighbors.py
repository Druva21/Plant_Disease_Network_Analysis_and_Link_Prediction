import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd

def common_neighbors_scores(G, edge_list):
    # edge_list: list of (u,v, label) where label 1=positive edge, 0=negative
    preds = []
    y = []
    for u,v,label in edge_list:
        cn = len(list(nx.common_neighbors(G, u, v)))
        preds.append(cn)
        y.append(label)
    return y, preds

if __name__ == '__main__':
    print('This module provides common_neighbors_scores(G, edge_list)') 
