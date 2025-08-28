import networkx as nx
import math

def adamic_adar_scores(G, edge_list):
    preds = []
    y = []
    for u,v,label in edge_list:
        score = 0.0
        for w in nx.common_neighbors(G, u, v):
            deg = G.degree(w)
            if deg > 1:
                score += 1.0 / math.log(deg)
        preds.append(score)
        y.append(label)
    return y, preds

if __name__ == '__main__':
    print('This module provides adamic_adar_scores(G, edge_list)') 
