"""Mean and standard deviation over seeds of every metric in results/<dataset>/seed_*/metrics.json."""
import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ["AUC-ROC", "AUC-PR", "R-AUC-ROC", "R-AUC-PR", "VUS-ROC", "VUS-PR"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default="./results")
    root = Path(parser.parse_args().save_path)
    rows = []
    for data_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = [json.loads(f.read_text()) for f in sorted(data_dir.glob("seed_*/metrics.json"))]
        if not runs:
            continue
        row = {"dataset": data_dir.name, "seeds": len(runs)}
        for m in METRICS:
            values = np.array([r[m] for r in runs])
            row[m] = f"{values.mean():.3f} ± {values.std():.3f}"
        rows.append(row)
    header = ["dataset", "seeds"] + METRICS
    print(" | ".join(header))
    for row in rows:
        print(" | ".join(str(row[h]) for h in header))
