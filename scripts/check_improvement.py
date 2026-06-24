import json
import sys
from pathlib import Path

def main():
    comparison_path = Path("artifacts/model_evaluation/metrics_comparison.json")
    if not comparison_path.exists():
        print("No comparison metrics found. This might be the first run. Proceeding with deployment.")
        sys.exit(0)

    try:
        with open(comparison_path, "r") as f:
            data = json.load(f)
        
        current_r2 = data.get("current", {}).get("r2")
        previous_r2 = data.get("previous", {}).get("r2")
        
        if current_r2 is None or previous_r2 is None:
            print("Missing R2 metrics in comparison. Proceeding with deployment.")
            sys.exit(0)
            
        if current_r2 > previous_r2:
            print(f"Model improved: R2 {current_r2:.4f} > {previous_r2:.4f}")
            sys.exit(0)
        else:
            print(f"Model did not improve: R2 {current_r2:.4f} <= {previous_r2:.4f}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error checking metrics: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
