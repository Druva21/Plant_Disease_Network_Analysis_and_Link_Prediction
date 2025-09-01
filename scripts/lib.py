
import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, confusion_matrix

RANDOM_STATE = 42

def _detect_columns(df):
    cols = {c.lower(): c for c in df.columns}
    # Try common names
    host_col = cols.get("host") or cols.get("plant") or cols.get("crop")
    ant_col = cols.get("antagonist") or cols.get("pathogen") or cols.get("disease")
    if not host_col or not ant_col:
        raise ValueError("Could not detect host/antagonist columns. Expected columns like 'host' and 'antagonist' (or 'pathogen').")
    return host_col, ant_col

def load_graph(csv_path="data/complete_plant_disease_database.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. See data/README.")
    df = pd.read_csv(csv_path)
    host_col, ant_col = _detect_columns(df)

    # Clean strings
    df = df[[host_col, ant_col]].dropna()
    df[host_col] = df[host_col].astype(str).str.strip().str.lower()
    df[ant_col] = df[ant_col].astype(str).str.strip().str.lower()

    # Build bipartite graph: prefix node ids to avoid collisions
    G = nx.Graph()
    for h, a in df[[host_col, ant_col]].itertuples(index=False):
        G.add_node(f"h::{h}", bipartite=0)
        G.add_node(f"a::{a}", bipartite=1)
        G.add_edge(f"h::{h}", f"a::{a}")
    return G

def train_test_edge_split(G, test_size=0.2, random_state=RANDOM_STATE):
    edges = list(G.edges())
    train_edges, test_edges = train_test_split(edges, test_size=test_size, random_state=random_state)
    G_train = G.copy()
    G_train.remove_edges_from(test_edges)

    # Ensure graph stays connected enough (fallback: add back if isolates created)
    isolates = list(nx.isolates(G_train))
    if isolates:
        # add back edges touching isolates from test set
        to_add = [e for e in test_edges if e[0] in isolates or e[1] in isolates]
        G_train.add_edges_from(to_add)
        test_edges = [e for e in test_edges if e not in to_add]

    # Generate negative samples for test set (non-edges)
    # Sample same count as positives
    non_edges = list(nx.non_edges(G))
    rng = np.random.default_rng(random_state)
    test_neg_idx = rng.choice(len(non_edges), size=len(test_edges), replace=False)
    test_neg = [non_edges[i] for i in test_neg_idx]

    return G_train, test_edges, test_neg

def _lp_generator(method, G, pairs):
    if method == "common_neighbors":
        # CN isn't a generator over given pairs; compute lengths manually
        for u, v in pairs:
            score = len(list(nx.common_neighbors(G, u, v)))
            yield (u, v, score)
    elif method == "jaccard":
        yield from nx.jaccard_coefficient(G, ebunch=pairs)
    elif method == "adamic_adar":
        yield from nx.adamic_adar_index(G, ebunch=pairs)
    elif method == "resource_allocation":
        yield from nx.resource_allocation_index(G, ebunch=pairs)
    else:
        raise ValueError(f"Unknown method {method}")

def score_edges(G_train, pos_edges, neg_edges, method):
    # Combine to compute in one pass
    pairs = pos_edges + neg_edges
    scores_map = {}
    for u, v, p in _lp_generator(method, G_train, pairs):
        scores_map[(u, v)] = p
    # Fallback 0 for missing values
    pos_scores = np.array([scores_map.get((u, v), 0.0) for u, v in pos_edges], dtype=float)
    neg_scores = np.array([scores_map.get((u, v), 0.0) for u, v in neg_edges], dtype=float)
    return pos_scores, neg_scores

def evaluate(pos_scores, neg_scores):
    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    scores = np.concatenate([pos_scores, neg_scores]).astype(float)

    # Normalize scores to [0,1] for thresholding
    if np.max(scores) > np.min(scores):
        norm = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        norm = scores.copy()

    roc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.5
    pr = average_precision_score(y_true, scores)

    # Choose threshold that maximizes F1 on PR curve
    prec, rec, thresh = precision_recall_curve(y_true, norm)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
    best_idx = int(np.nanargmax(f1s))
    best_f1 = float(f1s[best_idx])
    thr = float(thresh[min(best_idx, len(thresh)-1)]) if len(thresh) else 0.5

    y_pred = (norm >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cm = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    return {"roc_auc": float(roc), "pr_auc": float(pr), "f1": best_f1, "threshold": thr, "confusion_matrix": cm}

def run_method(method_name, csv_path="data/complete_plant_disease_database.csv", test_size=0.2):
    G = load_graph(csv_path)
    G_train, pos_test, neg_test = train_test_edge_split(G, test_size=test_size)
    pos_scores, neg_scores = score_edges(G_train, pos_test, neg_test, method=method_name)
    metrics = evaluate(pos_scores, neg_scores)
    return metrics
