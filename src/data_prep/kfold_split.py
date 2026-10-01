"""
Generates K stratified folds from the already-converted YOLO dataset
(data/yolo/images + labels, all images pooled together) and writes
K separate data.yaml files. Run this once, then train K times and
average the mAP results -- gives a defensible mean +/- std instead of
a single lucky/unlucky 85/15 split.

Usage:
    python kfold_split.py --src data/yolo --dst data/kfold --k 5
"""
import argparse
import random
import shutil
from pathlib import Path


def convert(src: Path, dst: Path, k: int, seed: int = 42):
    # Pool all images from the existing train+val split back together
    all_images = list((src / "images" / "train").glob("*.jpg")) + \
                 list((src / "images" / "val").glob("*.jpg"))
    random.seed(seed)
    random.shuffle(all_images)

    fold_size = len(all_images) // k
    folds = [all_images[i * fold_size:(i + 1) * fold_size] for i in range(k)]
    # dump remainder into last fold
    folds[-1].extend(all_images[k * fold_size:])

    def label_path_for(img_path: Path) -> Path:
        # labels live alongside in a parallel train/ or val/ folder
        for split in ("train", "val"):
            candidate = src / "labels" / split / f"{img_path.stem}.txt"
            if candidate.exists():
                return candidate
        return None

    for fold_idx in range(k):
        fold_dst = dst / f"fold{fold_idx}"
        (fold_dst / "images" / "train").mkdir(parents=True, exist_ok=True)
        (fold_dst / "images" / "val").mkdir(parents=True, exist_ok=True)
        (fold_dst / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (fold_dst / "labels" / "val").mkdir(parents=True, exist_ok=True)

        val_images = folds[fold_idx]
        train_images = [img for i, f in enumerate(folds) if i != fold_idx for img in f]

        for img in train_images:
            lbl = label_path_for(img)
            shutil.copy(img, fold_dst / "images" / "train" / img.name)
            if lbl:
                shutil.copy(lbl, fold_dst / "labels" / "train" / lbl.name)

        for img in val_images:
            lbl = label_path_for(img)
            shutil.copy(img, fold_dst / "images" / "val" / img.name)
            if lbl:
                shutil.copy(lbl, fold_dst / "labels" / "val" / lbl.name)

        with open(fold_dst / "data.yaml", "w") as f:
            f.write(
                f"path: {fold_dst.resolve()}\n"
                "train: images/train\n"
                "val: images/val\n"
                "names:\n"
                "  0: tool\n"
            )

        print(f"fold{fold_idx}: train={len(train_images)} val={len(val_images)}")

    print(f"\nDone. Now run for each fold (i=0..{k-1}):")
    print(f"  yolo detect train data={dst}/fold<i>/data.yaml model=yolo11n.pt epochs=60 imgsz=640 device=0")
    print("Then collect the mAP50 printed at the end of each run and compute mean/std.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    convert(args.src, args.dst, args.k)
