import os
from scripts.preprocess import build_graph_from_csv
import networkx as nx
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support
from scripts.common_neighbors import common_neighbors_scores
from scripts.jaccard import jaccard_scores
from scripts.adamic_adar import adamic_adar_scores
from scripts.resource_allocation import resource_allocation_scores

def train_test_edges(G, test_frac=0.15, seed=42):
    # Create positive edges list and sample negative edges
    random.seed(seed)
    edges = list(G.edges())
    num_test = max(1, int(len(edges)*test_frac))
    test_pos = random.sample(edges, num_test)
    G_train = G.copy()
    G_train.remove_edges_from(test_pos)
    # sample negative edges (node pairs not connected in G)
    nodes = list(G.nodes())
    test_neg = []
    while len(test_neg)<num_test:
        u,v = random.sample(nodes,2)
        if not G.has_edge(u,v) and (u,v) not in test_neg and (v,u) not in test_neg:
            test_neg.append((u,v))
    # Build labeled edge list
    test_edges = [(u,v,1) for u,v in test_pos] + [(u,v,0) for u,v in test_neg]
    return G_train, test_edges

def evaluate(y_true, y_score):
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = None
    try:
        ap = average_precision_score(y_true, y_score)
    except Exception:
        ap = None
    return auc, ap

def run_heuristic(method_name, func, G_train, test_edges):
    y_true, y_scores = func(G_train, test_edges)
    auc, ap = evaluate(y_true, y_scores)
    print(f'{method_name} -> ROC AUC: {auc}, PR AUC: {ap}')
    return {'method':method_name, 'auc':auc, 'ap':ap, 'y_true':y_true, 'y_scores':y_scores}

if __name__ == '__main__':
    csv_path = 'data/complete_plant_disease_database.csv'
    if not os.path.exists(csv_path):
        print('Dataset not found at', csv_path)
        print('Please download the Complete Plant Disease Database from Kaggle and place it at data/complete_plant_disease_database.csv')
        exit(1)

    G = build_graph_from_csv(csv_path, src_col='Host_Species', tgt_col='Antagonist_Species', weight_col=None)
    print('Built graph: nodes=', G.number_of_nodes(), 'edges=', G.number_of_edges())

    G_train, test_edges = train_test_edges(G, test_frac=0.15)
    results = []
    results.append(run_heuristic('Common Neighbors', common_neighbors_scores, G_train, test_edges))
    results.append(run_heuristic('Jaccard', jaccard_scores, G_train, test_edges))
    results.append(run_heuristic('Adamic-Adar', adamic_adar_scores, G_train, test_edges))
    results.append(run_heuristic('Resource Allocation', resource_allocation_scores, G_train, test_edges))

    # Save a simple bar plot of AUCs (if available)
    methods = [r['method'] for r in results]
    aucs = [r['auc'] if r['auc'] is not None else 0 for r in results]
    plt.figure(figsize=(8,4))
    plt.bar(methods, aucs)
    plt.title('Heuristic Methods ROC AUC (if computable)')
    plt.ylabel('ROC AUC')
    plt.savefig('results/heuristics_auc.png')
    plt.close()
    print('Saved results/heuristics_auc.png')
