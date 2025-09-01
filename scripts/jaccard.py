
from lib import run_method
import json

if __name__ == "__main__":
    metrics = run_method("jaccard")
    print(json.dumps({"method": "Jaccard", **metrics}, indent=2))
