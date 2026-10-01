"""
Render individual example images (one file per case) with ground-truth boxes (white) and YOLO11n predictions (orange):

* in-domain:     random Kvasir-Instrument validation frames (prediction confidence >= 0.4, as in the GUI)
* cross-dataset: random m2cai16 laparoscopic frames, never seen in training (confidence >= 0.4)
* failures:      the validation frames whose least-confident detection is lowest, ranked like failure_gallery.py
                 (confidence >= 0.1; frames with no detection rank first)

Usage:
    python src/analysis/render_examples.py --model runs/detect/train/weights/best.pt \
        --val data/yolo --m2cai data/m2cai16_yolo --out results/examples --n 3 --seed 0
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

WHITE, ORANGE, INK = (255, 255, 255), (52, 104, 235), (20, 20, 20)


def gt_boxes(label_path: Path, w: int, h: int):
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().split("\n"):
            if line.strip():
                _, cx, cy, bw, bh = map(float, line.split())
                boxes.append((int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h)))
    return boxes


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1)


def render(model, img_path: Path, label_path: Path, out_path: Path, title: str, conf: float):
    img = cv2.imread(str(img_path)); h, w = img.shape[:2]
    gts = gt_boxes(label_path, w, h)
    r = model.predict(str(img_path), conf=conf, verbose=False)[0]
    preds = [(tuple(map(int, b)), float(c)) for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    t = max(2, w // 300)
    for g in gts:                                     # drawn thicker, so it stays visible under a matching prediction
        cv2.rectangle(img, g[:2], g[2:], WHITE, t + 4)
    for (x1, y1, x2, y2), c in preds:
        cv2.rectangle(img, (x1, y1), (x2, y2), ORANGE, t)
        cv2.putText(img, f"tool {c:.2f}", (x1 + 4, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORANGE, 2, cv2.LINE_AA)
    matched = sum(any(iou(g, p) >= 0.5 for p, _ in preds) for g in gts)
    scale = 900 / w
    img = cv2.resize(img, (900, round(h * scale)), interpolation=cv2.INTER_AREA)
    head = np.full((58, 900, 3), 251, np.uint8)
    cv2.putText(head, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.66, INK, 2, cv2.LINE_AA)
    detail = (f"ground truth (white): {len(gts)} tool(s), {matched} found at IoU >= 0.5   |   "
              f"predictions (orange, conf >= {conf}): {len(preds)}")
    cv2.putText(head, detail, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 108, 104), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([head, img]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return len(gts), matched, len(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", type=Path, required=True, help="YOLO dataset root with images/val and labels/val")
    ap.add_argument("--m2cai", type=Path, required=True, help="m2cai16 YOLO root with images/ and labels/")
    ap.add_argument("--out", type=Path, default=Path("results/examples"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    model = YOLO(a.model); rng = random.Random(a.seed)

    val_imgs = sorted((a.val / "images/val").glob("*.jpg"))
    for i, p in enumerate(rng.sample(val_imgs, a.n), 1):
        print("in-domain", p.name, render(model, p, a.val / "labels/val" / f"{p.stem}.txt", a.out / f"in_domain_{i}.jpg",
                                         f"Kvasir-Instrument validation frame {p.stem[:12]}", 0.4))

    m2_imgs = sorted((a.m2cai / "images").glob("*.jpg"))
    for i, p in enumerate(rng.sample(m2_imgs, a.n), 1):
        print("cross-dataset", p.name, render(model, p, a.m2cai / "labels" / f"{p.stem}.txt", a.out / f"cross_dataset_{i}.jpg",
                                             f"m2cai16 laparoscopic frame {p.stem} (other domain, not trained on)", 0.4))

    scored = []
    for p in val_imgs:                                   # same ranking as failure_gallery.py
        r = model.predict(str(p), conf=0.1, verbose=False)[0]
        scored.append((0.0 if len(r.boxes) == 0 else float(r.boxes.conf.min()), p))
    for i, (c, p) in enumerate(sorted(scored)[: a.n], 1):
        print("failure", p.name, c, render(model, p, a.val / "labels/val" / f"{p.stem}.txt", a.out / f"failure_{i}.jpg",
                                          f"Least-confident validation frame #{i} (lowest detection confidence {c:.2f})", 0.1))


if __name__ == "__main__":
    main()
