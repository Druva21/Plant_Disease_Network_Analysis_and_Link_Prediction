
from lib import run_method
import json

if __name__ == "__main__":
    metrics = run_method("common_neighbors")
    print(json.dumps({"method": "Common Neighbors", **metrics}, indent=2))
