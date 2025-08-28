import pandas as pd
import networkx as nx
import os

def build_graph_from_csv(csv_path='data/complete_plant_disease_database.csv', src_col='Host_Species', tgt_col='Antagonist_Species', weight_col=None):
    """Load the CSV and build a bipartite or homogeneous graph depending on columns.
    Expects CSV to contain host and antagonist species columns. Adjust column names if different."""
    df = pd.read_csv(csv_path)
    # Basic cleaning: drop NaNs in key columns
    df = df.dropna(subset=[src_col, tgt_col]).copy()
    G = nx.Graph()
    # Add edges; if weight_col provided use it
    for _, row in df.iterrows():
        u = str(row[src_col]).strip()
        v = str(row[tgt_col]).strip()
        if weight_col and pd.notna(row.get(weight_col)):
            w = float(row[weight_col])
            if G.has_edge(u, v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w)
        else:
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)
    return G

if __name__ == '__main__':
    import sys
    csv = sys.argv[1] if len(sys.argv)>1 else 'data/complete_plant_disease_database.csv'
    print('Building graph from', csv)
    G = build_graph_from_csv(csv)
    print('Nodes:', G.number_of_nodes(), 'Edges:', G.number_of_edges())
