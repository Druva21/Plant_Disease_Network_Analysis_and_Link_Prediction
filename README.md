# Plant_Disease_Link_Prediction

This repository contains code for **Plant Disease Network Analysis and Link Prediction**.
Methods implemented include traditional link-prediction heuristics (Common Neighbors, Jaccard, Adamic-Adar, Resource Allocation)
and a Graph Neural Network (GNN) pipeline (PyTorch Geometric compatible).

**Dataset**: Used the *Complete Plant Disease Database* (available on Kaggle). _Dataset is NOT included in this repo._
Please download the dataset from Kaggle and place it in the `data/` folder as `complete_plant_disease_database.csv` before running the scripts.

## Repo structure
```
Plant_Disease_Link_Prediction/
├─ data/                    # Place dataset CSV here (not included)
├─ scripts/
│  ├─ preprocess.py         # build graph, feature extraction
│  ├─ common_neighbors.py
│  ├─ jaccard.py
│  ├─ adamic_adar.py
│  ├─ resource_allocation.py
│  ├─ gnn_model.py          # GNN training (PyTorch Geometric)
│  └─ main.py               # run all methods & compare metrics
├─ notebooks/               # optional exploratory notebooks
├─ results/                 # saved metrics & plots
├─ images/                  # visualizations
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## How to run (example)
1. Create a virtual environment and install requirements:
```bash
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Download dataset from Kaggle and place it as `data/complete_plant_disease_database.csv`.

3. Run the main comparison script:
```bash
python scripts/main.py
```

Notes:
- `gnn_model.py` uses PyTorch Geometric; if you don't have GPU or PyG installed, the script will skip the GNN and run heuristics only.
- Results (metrics, plots) will be saved to the `results/` and `images/` folders.


