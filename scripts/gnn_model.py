
import os, json
import torch
from lib import load_graph, train_test_edge_split
try:
    from torch_geometric.utils import from_networkx
    from torch import nn
    from torch_geometric.nn import GCNConv
    PYG_AVAILABLE = True
except Exception as e:
    PYG_AVAILABLE = False
    IMPORT_ERROR = str(e)

def run_gnn(csv_path="data/complete_plant_disease_database.csv", hidden_dim=32, epochs=50, lr=1e-2):
    G = load_graph(csv_path)
    G_train, pos_test, neg_test = train_test_edge_split(G)

    data = from_networkx(G_train)
    # Node features: simple identity (degree) feature
    deg = torch.tensor([G_train.degree(n) for n in G_train.nodes()], dtype=torch.float).view(-1,1)
    data.x = deg
    data = data

    # Build index mapping from node to integer
    node_to_idx = {n:i for i, n in enumerate(G_train.nodes())}

    pos_idx = torch.tensor([[node_to_idx[u], node_to_idx[v]] for u,v in pos_test], dtype=torch.long).t()
    neg_idx = torch.tensor([[node_to_idx[u], node_to_idx[v]] for u,v in neg_test], dtype=torch.long).t()

    class LinkPredictor(torch.nn.Module):
        def __init__(self, in_dim, hidden):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, hidden)
        def encode(self, x, edge_index):
            h = torch.relu(self.conv1(x, edge_index))
            h = self.conv2(h, edge_index)
            return h
        def decode(self, z, edge_idx):
            src, dst = edge_idx
            return (z[src] * z[dst]).sum(dim=-1)

    model = LinkPredictor(in_dim=data.x.size(1), hidden=hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def bce_logits(pos, neg):
        y = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        logits = torch.cat([pos, neg])
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        z = model.encode(data.x, data.edge_index)
        pos_logit = model.decode(z, pos_idx)
        neg_logit = model.decode(z, neg_idx)
        loss = bce_logits(pos_logit, neg_logit)
        loss.backward()
        opt.step()
        if (ep+1) % 10 == 0:
            print(f"Epoch {ep+1}/{epochs} - loss {loss.item():.4f}")

    # Simple AUC proxy via logits (no sklearn here to keep deps minimal)
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        pos_logit = model.decode(z, pos_idx).sigmoid()
        neg_logit = model.decode(z, neg_idx).sigmoid()
        scores = torch.cat([pos_logit, neg_logit]).cpu().numpy()
        y_true = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)]).cpu().numpy()
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
        roc = float(roc_auc_score(y_true, scores))
        pr = float(average_precision_score(y_true, scores))
        # fixed threshold 0.5
        y_pred = (scores >= 0.5).astype(int)
        from sklearn.metrics import f1_score
        f1 = float(f1_score(y_true, y_pred))
    except Exception:
        roc = pr = f1 = None

    return {"roc_auc": roc, "pr_auc": pr, "f1": f1}

if __name__ == "__main__":
    if not PYG_AVAILABLE:
        print(json.dumps({"warning": "PyTorch Geometric not available", "detail": IMPORT_ERROR}, indent=2))
        print("Install it and re-run: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html")
    else:
        m = run_gnn()
        print(json.dumps({"method": "GNN", **m}, indent=2))
