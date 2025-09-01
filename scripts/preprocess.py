
from lib import load_graph
import argparse, networkx as nx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/complete_plant_disease_database.csv", help="Path to Kaggle CSV")
    args = parser.parse_args()
    G = load_graph(args.csv)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    # Print small summary by node types
    part0 = sum(1 for _, d in G.nodes(data=True) if d.get("bipartite") == 0)
    part1 = G.number_of_nodes() - part0
    print(f"Hosts: {part0} | Antagonists: {part1}")
if __name__ == "__main__":
    main()
