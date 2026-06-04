from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config import DEFAULT_DB_PATH, DEFAULT_MODEL, FINE_TUNING_DIR, TARGET_CLASSES
from database import fetch_detections
from logging_utils import get_logger


logger = get_logger("fine_tuning")
CLASS_ORDER = ["bicycle", "car", "motorcycle", "bus", "truck"]


def dataset_dir(root: Path | str | None = None) -> Path:
    return Path(root) if root else FINE_TUNING_DIR / "vehicle_dataset"


def write_data_yaml(root: Path, class_names: list[str]) -> Path:
    yaml_path = root / "data.yaml"
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
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
    logger.info("fine_tuning_yaml_written path=%s classes=%s", yaml_path, ",".join(class_names))
    return yaml_path


def init_dataset(root: Path) -> Path:
    for split in ["train", "val"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "review_crops").mkdir(parents=True, exist_ok=True)
    yaml_path = write_data_yaml(root, CLASS_ORDER)
    logger.info("fine_tuning_dataset_initialized root=%s", root)
    return yaml_path


def export_review_crops(db_path: Path | str, output_dir: Path, limit: int = 300) -> int:
    rows = fetch_detections(db_path)
    copied = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        class_name = str(row.get("class_name") or "unknown")
        if class_name not in TARGET_CLASSES:
            continue
        crop_path = row.get("vehicle_crop_path")
        if not crop_path:
            continue
        source = Path(str(crop_path))
        if not source.exists():
            continue

        target_dir = output_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        if not target.exists():
            shutil.copy2(source, target)
            copied += 1
        if copied >= limit:
            break

    logger.info("fine_tuning_review_crops_exported output_dir=%s copied=%s limit=%s", output_dir, copied, limit)
    return copied


def train_model(data_yaml: Path, model: str, epochs: int, imgsz: int, batch: int, project: Path, name: str) -> Path:
    from ultralytics import YOLO

    logger.info(
        "fine_tuning_train_started data=%s model=%s epochs=%s imgsz=%s batch=%s project=%s name=%s",
        data_yaml,
        model,
        epochs,
        imgsz,
        batch,
        project,
        name,
    )
    yolo = YOLO(model)
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project),
        name=name,
    )
    run_dir = Path(getattr(results, "save_dir", project / name))
    best = run_dir / "weights" / "best.pt"
    logger.info("fine_tuning_train_finished run_dir=%s best_weights=%s", run_dir, best)
    return best


def validate_model(data_yaml: Path, weights: str, imgsz: int) -> None:
    from ultralytics import YOLO

    logger.info("fine_tuning_validate_started data=%s weights=%s imgsz=%s", data_yaml, weights, imgsz)
    yolo = YOLO(weights)
    metrics = yolo.val(data=str(data_yaml), imgsz=imgsz)
    logger.info("fine_tuning_validate_finished metrics=%s", metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASSBI YOLO fine tuning helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-dataset", help="Create YOLO dataset folders and data.yaml.")
    init.add_argument("--root", type=Path, default=dataset_dir())

    crops = sub.add_parser("export-review-crops", help="Copy saved vehicle crops for manual review/annotation.")
    crops.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    crops.add_argument("--output", type=Path, default=dataset_dir() / "review_crops")
    crops.add_argument("--limit", type=int, default=300)

    train = sub.add_parser("train", help="Train YOLO model on a labelled YOLO dataset.")
    train.add_argument("--data", type=Path, default=dataset_dir() / "data.yaml")
    train.add_argument("--model", default=DEFAULT_MODEL)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--project", type=Path, default=FINE_TUNING_DIR / "runs")
    train.add_argument("--name", default="vehicle_surveillance")

    val = sub.add_parser("validate", help="Validate trained YOLO weights.")
    val.add_argument("--data", type=Path, default=dataset_dir() / "data.yaml")
    val.add_argument("--weights", required=True)
    val.add_argument("--imgsz", type=int, default=640)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "init-dataset":
        yaml_path = init_dataset(args.root)
        print(f"Dataset created: {args.root}")
        print(f"Data YAML: {yaml_path}")
        print("Put labelled YOLO images/labels into images/train, images/val, labels/train, labels/val.")
        return

    if args.command == "export-review-crops":
        copied = export_review_crops(args.db, args.output, args.limit)
        print(f"Copied {copied} crop images into {args.output}")
        print("Use these crops for review. For YOLO fine tuning, annotate full frames or prepared images with YOLO labels.")
        return

    if args.command == "train":
        best = train_model(args.data, args.model, args.epochs, args.imgsz, args.batch, args.project, args.name)
        print(f"Best weights: {best}")
        print(f"Use it with: ASSBI_YOLO_MODEL={best}")
        return

    if args.command == "validate":
        validate_model(args.data, args.weights, args.imgsz)
        print("Validation finished. Check terminal output and logs.")


if __name__ == "__main__":
    main()
