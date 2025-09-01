
from lib import run_method
import json

if __name__ == "__main__":
    metrics = run_method("resource_allocation")
    print(json.dumps({"method": "Resource Allocation", **metrics}, indent=2))
