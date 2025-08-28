import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score

def jaccard_scores(G, edge_list):
    preds = []
    y = []
    for u,v,label in edge_list:
        # Jaccard coefficient provided by networkx returns numerator/denom
        try:
            neigh_u = set(G.neighbors(u))
            neigh_v = set(G.neighbors(v))
            inter = len(neigh_u & neigh_v)
            union = len(neigh_u | neigh_v)
            score = inter/union if union>0 else 0.0
        except Exception:
            score = 0.0
        preds.append(score)
        y.append(label)
    return y, preds

if __name__ == '__main__':
    print('This module provides jaccard_scores(G, edge_list)') 
