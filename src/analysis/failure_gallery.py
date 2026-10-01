"""
Runs the trained model on the validation set, ranks predictions by
confidence, and saves a grid image of the worst cases (lowest-confidence
true detections + any frames with zero detections) for qualitative
failure analysis -- useful in interviews to show you looked at where
the model actually struggles, not just the aggregate mAP.

Usage:
    python failure_gallery.py --model runs/detect/train/weights/best.pt \
        --images data/yolo/images/val --out analysis/failure_gallery.jpg --n 9
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def build_gallery(model_path: str, images_dir: Path, out_path: Path, n: int, grid_cols: int = 3):
    model = YOLO(model_path)
    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

    scored = []
    for img_path in image_paths:
        result = model.predict(str(img_path), conf=0.1, verbose=False)[0]
        if len(result.boxes) == 0:
            scored.append((img_path, 0.0, result))  # zero-detection frame -- worst case
        else:
            min_conf = float(result.boxes.conf.min())
            scored.append((img_path, min_conf, result))

    # worst (lowest confidence, including zero-detections) first
    scored.sort(key=lambda x: x[1])
    worst = scored[:n]

    tiles = []
    tile_size = 300
    for img_path, conf, result in worst:
        annotated = result.plot()
        annotated = cv2.resize(annotated, (tile_size, tile_size))
        label = f"{img_path.name} min_conf={conf:.2f}"
        cv2.putText(annotated, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        tiles.append(annotated)

    while len(tiles) % grid_cols != 0:
        tiles.append(np.zeros((tile_size, tile_size, 3), dtype=np.uint8))

    rows = [np.hstack(tiles[i:i + grid_cols]) for i in range(0, len(tiles), grid_cols)]
    grid = np.vstack(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"Saved failure gallery ({len(worst)} cases) to {out_path}")
    for img_path, conf, _ in worst:
        print(f"  {img_path.name}: min_conf={conf:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("analysis/failure_gallery.jpg"))
    parser.add_argument("--n", type=int, default=9)
    args = parser.parse_args()
    build_gallery(args.model, args.images, args.out, args.n)
