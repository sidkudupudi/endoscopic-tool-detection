"""
Converts m2cai16-tool-locations (PASCAL VOC2007 XML format) into a
YOLO-format validation-only set, for cross-dataset evaluation of your
Kvasir-Instrument-trained (single-class) model.

All 7 tool classes (Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator,
SpecimenBag) are mapped down to a single class 0 ("tool"), since we're
testing whether the model generalizes to detecting tool presence/location
across datasets -- not testing tool-type classification, which your model
was never trained to do.

Usage:
    python convert_m2cai16_to_yolo.py \
        --src m2cai16-tool-locations/m2cai16-tool-locations \
        --dst data/m2cai16_yolo
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def convert(src: Path, dst: Path):
    ann_dir = src / "Annotations"
    img_dir = src / "JPEGImages"

    if not ann_dir.exists() or not img_dir.exists():
        raise FileNotFoundError(
            f"Expected {ann_dir} and {img_dir} to exist -- check the actual "
            "folder names with `find` and adjust --src accordingly."
        )

    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)

    xml_files = sorted(ann_dir.glob("*.xml"))
    n_converted = 0
    n_boxes = 0

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text
        img_path = img_dir / filename
        if not img_path.exists():
            candidates = list(img_dir.glob(f"{xml_path.stem}.*"))
            if not candidates:
                continue
            img_path = candidates[0]

        size = root.find("size")
        w = float(size.find("width").text)
        h = float(size.find("height").text)

        lines = []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            cx = ((xmin + xmax) / 2) / w
            cy = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h

            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_boxes += 1

        if not lines:
            continue

        shutil.copy(img_path, dst / "images" / img_path.name)
        with open(dst / "labels" / f"{img_path.stem}.txt", "w") as f:
            f.write("\n".join(lines))
        n_converted += 1

    with open(dst / "data.yaml", "w") as f:
        f.write(
            f"path: {dst.resolve()}\n"
            "val: images\n"
            "names:\n"
            "  0: tool\n"
        )

    print(f"Converted {n_converted} frames, {n_boxes} boxes (7 classes merged into 1).")
    print(f"data.yaml written to {dst / 'data.yaml'}")
    print(f"\nRun: yolo detect val model=runs/detect/train/weights/best.pt data={dst / 'data.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args()
    convert(args.src, args.dst)
