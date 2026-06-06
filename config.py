from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).resolve().parent / ".env")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DB_PATH = Path(os.getenv("SURVEILLANCE_DB", DATA_DIR / "surveillance.db"))
MATPLOTLIB_CACHE_DIR = Path(os.getenv("MPLCONFIGDIR", DATA_DIR / "matplotlib_cache"))
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
LOG_DIR = Path(os.getenv("ASSBI_LOG_DIR", DATA_DIR / "logs"))
LOG_FILE = Path(os.getenv("ASSBI_LOG_FILE", LOG_DIR / "app.log"))
FINE_TUNING_DIR = Path(os.getenv("ASSBI_FINE_TUNING_DIR", DATA_DIR / "fine_tuning"))
ROBOFLOW_DATASET_DIR = Path(os.getenv("ASSBI_ROBOFLOW_DATASET_DIR", DATA_DIR / "roboflow_dataset"))
ENABLE_ROBOFLOW_DATASET_EXPORT = os.getenv("ENABLE_ROBOFLOW_DATASET_EXPORT", "1").strip().lower() not in {"0", "false", "no"}
ENABLE_VEHICLE_RECOGNITION = os.getenv("ASSBI_ENABLE_VEHICLE_RECOGNITION", "0").strip().lower() not in {"0", "false", "no"}
TRACKING_FRAME_STRIDE = max(1, int(os.getenv("ASSBI_TRACKING_FRAME_STRIDE", "1")))
TRACKING_INFERENCE_WIDTH = max(0, int(os.getenv("ASSBI_TRACKING_INFERENCE_WIDTH", "960")))
TRACKING_DISPLAY_WIDTH = max(0, int(os.getenv("ASSBI_TRACKING_DISPLAY_WIDTH", "1280")))
YTDLP_COOKIES_FROM_BROWSER = os.getenv("ASSBI_YTDLP_COOKIES_FROM_BROWSER", "chrome").strip()
YTDLP_COOKIE_FILE = os.getenv("ASSBI_YTDLP_COOKIE_FILE", "").strip()

DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=7uG-gbg0I8Y"
DEFAULT_MODEL = os.getenv("ASSBI_YOLO_MODEL", "yolo11n.pt")
TARGET_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
ROBOFLOW_CLASSES = [
    item.strip()
    for item in os.getenv("ASSBI_ROBOFLOW_CLASSES", "car,bus,truck,motorcycle,bicycle").split(",")
    if item.strip()
]

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
