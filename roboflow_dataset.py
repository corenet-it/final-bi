from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from config import DEFAULT_DB_PATH, ROBOFLOW_CLASSES, ROBOFLOW_DATASET_DIR
from database import fetch_detections
from logging_utils import get_logger


logger = get_logger("roboflow_dataset")


def class_names() -> list[str]:
    return list(dict.fromkeys(ROBOFLOW_CLASSES))


def class_index(class_name: str) -> int | None:
    try:
        return class_names().index(class_name)
    except ValueError:
        return None


def ensure_dataset(root: Path = ROBOFLOW_DATASET_DIR) -> Path:
    for split in ["train", "val"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    yaml_path = write_data_yaml(root)
    logger.info("roboflow_dataset_ready root=%s yaml=%s", root, yaml_path)
    return yaml_path


def write_data_yaml(root: Path = ROBOFLOW_DATASET_DIR) -> Path:
    yaml_path = root / "data.yaml"
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names()))
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def bbox_to_yolo(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(image_width), x1))
    x2 = max(0.0, min(float(image_width), x2))
    y1 = max(0.0, min(float(image_height), y1))
    y2 = max(0.0, min(float(image_height), y2))
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    cx = x1 + box_width / 2
    cy = y1 + box_height / 2
    return (
        cx / image_width,
        cy / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def split_for_track(track_id: int | None, frame_number: int) -> str:
    value = track_id if track_id is not None else frame_number
    return "val" if int(value) % 5 == 0 else "train"


def save_sample(
    frame: Any,
    bbox: tuple[float, float, float, float],
    *,
    class_name: str,
    video_id: int,
    roi_zone_id: int,
    track_id: int | None,
    frame_number: int,
    root: Path = ROBOFLOW_DATASET_DIR,
) -> dict[str, str] | None:
    idx = class_index(class_name)
    if idx is None:
        return None

    import cv2

    ensure_dataset(root)
    height, width = frame.shape[:2]
    x_center, y_center, box_width, box_height = bbox_to_yolo(bbox, width, height)
    if box_width <= 0 or box_height <= 0:
        logger.warning("roboflow_sample_skipped reason=invalid_bbox class=%s frame=%s", class_name, frame_number)
        return None

    split = split_for_track(track_id, frame_number)
    stem = f"video{video_id}_roi{roi_zone_id}_frame{frame_number}_{class_name}_track{track_id or 'none'}"
    image_path = root / "images" / split / f"{stem}.jpg"
    label_path = root / "labels" / split / f"{stem}.txt"

    ok = cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        logger.error("roboflow_image_write_failed path=%s", image_path)
        return None

    label_path.write_text(
        f"{idx} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n",
        encoding="utf-8",
    )
    logger.info("roboflow_sample_saved image=%s label=%s class=%s split=%s", image_path, label_path, class_name, split)
    return {"image_path": str(image_path), "label_path": str(label_path), "split": split}


def export_crops_as_yolo_dataset(db_path: Path, root: Path, limit: int = 300) -> int:
    ensure_dataset(root)
    rows = fetch_detections(db_path)
    copied = 0
    for row in rows:
        class_name = str(row.get("class_name") or "")
        idx = class_index(class_name)
        if idx is None:
            continue
        crop_path = row.get("vehicle_crop_path")
        if not crop_path:
            continue
        source = Path(str(crop_path))
        if not source.exists():
            continue

        split = split_for_track(row.get("track_id"), row.get("frame_number") or copied)
        stem = source.stem
        target_image = root / "images" / split / f"{stem}.jpg"
        target_label = root / "labels" / split / f"{stem}.txt"
        target_image.write_bytes(source.read_bytes())
        target_label.write_text(f"{idx} 0.500000 0.500000 1.000000 1.000000\n", encoding="utf-8")
        copied += 1
        if copied >= limit:
            break
    logger.info("roboflow_crops_exported root=%s copied=%s limit=%s", root, copied, limit)
    return copied


def dataset_summary(root: Path = ROBOFLOW_DATASET_DIR) -> dict[str, int]:
    return {
        "train_images": len(list((root / "images" / "train").glob("*.jpg"))),
        "val_images": len(list((root / "images" / "val").glob("*.jpg"))),
        "train_labels": len(list((root / "labels" / "train").glob("*.txt"))),
        "val_labels": len(list((root / "labels" / "val").glob("*.txt"))),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Roboflow-ready YOLO dataset helper for ASSBI.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create Roboflow/YOLO dataset folders and data.yaml.")
    init.add_argument("--root", type=Path, default=ROBOFLOW_DATASET_DIR)

    crops = sub.add_parser("export-crops", help="Export saved vehicle crops as YOLO-labelled images.")
    crops.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    crops.add_argument("--root", type=Path, default=ROBOFLOW_DATASET_DIR)
    crops.add_argument("--limit", type=int, default=300)

    summary = sub.add_parser("summary", help="Print dataset image/label counts.")
    summary.add_argument("--root", type=Path, default=ROBOFLOW_DATASET_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "init":
        yaml_path = ensure_dataset(args.root)
        print(f"Roboflow dataset ready: {args.root}")
        print(f"Data YAML: {yaml_path}")
        return
    if args.command == "export-crops":
        copied = export_crops_as_yolo_dataset(args.db, args.root, args.limit)
        print(f"Exported {copied} crop images with YOLO labels into {args.root}")
        print("Upload this folder to Roboflow or use it as a starter dataset.")
        return
    if args.command == "summary":
        print(dataset_summary(args.root))


if __name__ == "__main__":
    main()
