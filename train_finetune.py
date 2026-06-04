"""Fine-tune yolo11n on the cleaned ASSBI vehicle dataset.

Works on any machine (Linux/Mac). It first regenerates data.yaml so the
absolute `path:` matches THIS machine, then auto-selects the fastest device
(Apple Silicon `mps`, NVIDIA `cuda`, otherwise `cpu`).

Run:  python train_finetune.py
Output: data/fine_tuning/runs/vehicle_surveillance/weights/best.pt
"""
from pathlib import Path

import torch
from ultralytics import YOLO

from roboflow_dataset import write_data_yaml

BASE = "yolo11n.pt"          # ultralytics auto-downloads this
PROJECT = "data/fine_tuning/runs"
NAME = "vehicle_surveillance"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    # Regenerate data.yaml so `path:` is correct on this machine.
    data_yaml = write_data_yaml()
    device = pick_device()
    print(f"Using device={device}  data={data_yaml}")

    model = YOLO(BASE)
    model.train(
        data=str(data_yaml),
        epochs=40,
        imgsz=512,
        batch=16,
        device=device,
        workers=8,
        patience=10,        # early stop if val stops improving
        cache=False,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    best = Path(PROJECT) / NAME / "weights" / "best.pt"
    print(f"BEST_WEIGHTS={best.resolve()}")
    print(f"Use it with:  ASSBI_YOLO_MODEL={best.resolve()}")


if __name__ == "__main__":
    main()
