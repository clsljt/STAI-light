import argparse
import json

from exp.exp_anomaly_detection import Exp_Anomaly_Detection

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STAI: unsupervised multivariate time-series anomaly detection")
    parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test")
    parser.add_argument("--data", required=True, help="dataset name, i.e. the CSV stem listed in DETECT_META.csv")
    parser.add_argument("--root_path", default="./dataset", help="folder with DETECT_META.csv and data/<name>.csv")
    parser.add_argument("--config", default="./configs/stai_lite.json")
    parser.add_argument("--save_path", default="./results")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    exp = Exp_Anomaly_Detection(args, config)
    if args.mode in ("train", "train_test"):
        exp.train()
    if args.mode in ("test", "train_test"):
        exp.test()
