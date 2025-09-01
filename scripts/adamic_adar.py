
from lib import run_method
import json

if __name__ == "__main__":
    metrics = run_method("adamic_adar")
    print(json.dumps({"method": "Adamic-Adar", **metrics}, indent=2))
