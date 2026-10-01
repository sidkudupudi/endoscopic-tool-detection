"""
Runs YOLO training across all K folds produced by kfold_split.py and
reports mean +/- std mAP50 / mAP50-95 across folds.

Usage:
    python run_kfold.py --folds-dir data/kfold --k 5
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def run(folds_dir: Path, k: int, epochs: int = 60):
    map50s, map5095s = [], []

    for i in range(k):
        data_yaml = folds_dir / f"fold{i}" / "data.yaml"
        print(f"\n=== Training fold {i} ===")
        model = YOLO("yolo11n.pt")
        model.train(data=str(data_yaml), epochs=epochs, imgsz=640, device=0,
                     project="runs/kfold", name=f"fold{i}", verbose=False)
        metrics = model.val(data=str(data_yaml))
        map50 = metrics.box.map50
        map5095 = metrics.box.map
        print(f"fold{i}: mAP50={map50:.4f} mAP50-95={map5095:.4f}")
        map50s.append(map50)
        map5095s.append(map5095)

    print("\n=== K-Fold Results ===")
    print(f"mAP50:    mean={np.mean(map50s):.4f}  std={np.std(map50s):.4f}  per-fold={['%.4f' % m for m in map50s]}")
    print(f"mAP50-95: mean={np.mean(map5095s):.4f}  std={np.std(map5095s):.4f}  per-fold={['%.4f' % m for m in map5095s]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    run(args.folds_dir, args.k, args.epochs)
