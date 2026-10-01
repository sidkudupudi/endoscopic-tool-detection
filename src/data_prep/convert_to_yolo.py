"""
Converts the Kvasir-Instrument dataset (images/ + json bbox annotations)
into YOLO format (images/ + labels/ with normalized xywh) and writes a
train/val split plus a data.yaml for ultralytics training.

Usage:
    python convert_to_yolo.py --src data/raw --dst data/yolo
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image


def convert(src: Path, dst: Path, val_ratio: float = 0.15, seed: int = 42):
    img_dir = src / "images"
    ann_dir = src / "bbox"  # Kvasir-Instrument ships per-image bbox jsons here
    combined_json = None
    if not ann_dir.exists():
        # Some mirrors ship one combined JSON keyed by image stem instead of
        # per-image files, e.g. bounding_boxes.json -> {"stem": {"bbox": [...]}}
        candidates = list(src.glob("*.json"))
        if candidates:
            with open(candidates[0]) as f:
                combined_json = json.load(f)
        ann_dir = src

    images = sorted([p for p in img_dir.glob("*.jpg")])
    random.seed(seed)
    random.shuffle(images)
    n_val = max(1, int(len(images) * val_ratio))
    splits = {"val": images[:n_val], "train": images[n_val:]}

    for split, imgs in splits.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            stem = img_path.stem
            if combined_json is not None:
                ann = combined_json.get(stem)
                if ann is None:
                    continue
            else:
                ann_path = ann_dir / f"{stem}.json"
                if not ann_path.exists():
                    continue
                with open(ann_path) as f:
                    ann = json.load(f)

            w, h = Image.open(img_path).size
            lines = []
            # Kvasir-Instrument bbox json: {"bbox": [{"xmin":.., "ymin":.., "xmax":.., "ymax":..}, ...]}
            boxes = ann.get("bbox", ann.get("boxes", []))
            for b in boxes:
                xmin, ymin, xmax, ymax = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
                cx = ((xmin + xmax) / 2) / w
                cy = ((ymin + ymax) / 2) / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            shutil.copy(img_path, dst / "images" / split / img_path.name)
            with open(dst / "labels" / split / f"{stem}.txt", "w") as f:
                f.write("\n".join(lines))

    with open(dst / "data.yaml", "w") as f:
        f.write(
            f"path: {dst.resolve()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: tool\n"
        )

    print(f"Done. train={len(splits['train'])} val={len(splits['val'])}")
    print(f"data.yaml written to {dst / 'data.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()
    convert(args.src, args.dst, args.val_ratio)
