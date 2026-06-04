from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from config import DEFAULT_MODEL, ENABLE_ROBOFLOW_DATASET_EXPORT, TARGET_CLASSES
from database import create_video, insert_many_detections, upsert_roi_zone
from logging_utils import get_logger
from roboflow_dataset import save_sample as save_roboflow_sample
from vehicle_recognition import recognize_vehicle_model


logger = get_logger("tracker")


def resolve_video_source(source_url: str) -> str:
    if "youtube.com" not in source_url and "youtu.be" not in source_url:
        logger.info("direct_video_source source_url=%s", source_url)
        return source_url
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("YouTube stream uchun yt-dlp o'rnatilishi kerak.") from exc

    options = {"format": "best[ext=mp4]/best", "quiet": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(source_url, download=False)
        logger.info("youtube_source_resolved source_url=%s", source_url)
        return info["url"]


def normalized_polygon_to_pixels(points: list[list[float]], width: int, height: int) -> list[tuple[int, int]]:
    return [(int(x * width), int(y * height)) for x, y in points]


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[int, int]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def validate_model_path(model_name: str) -> None:
    if ("/" in model_name or model_name.endswith(".pt")) and not Path(model_name).exists():
        raise FileNotFoundError(
            f"YOLO model topilmadi: {model_name}. "
            "Roboflow model ishlatmoqchi bo'lsangiz, real best.pt pathini bering. "
            "Aks holda default yolo11n.pt modelidan foydalaning."
        )


def track_video(
    *,
    source_url: str,
    db_path: Path | str,
    roi_name: str,
    roi_points: list[list[float]],
    model_name: str = DEFAULT_MODEL,
    confidence: float = 0.35,
    frame_stride: int = 1,
    max_frames: int | None = 500,
    progress_callback: Callable[[dict], None] | None = None,
) -> int:
    logger.info(
        "track_video_started source_url=%s roi_name=%s model=%s confidence=%s frame_stride=%s max_frames=%s",
        source_url,
        roi_name,
        model_name,
        confidence,
        frame_stride,
        max_frames,
    )
    validate_model_path(model_name)
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Tracking uchun opencv-python va ultralytics paketlari kerak.") from exc

    stream_url = resolve_video_source(source_url)
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        logger.error("track_video_open_failed source_url=%s", source_url)
        raise RuntimeError("Video ochilmadi. YouTube URL yoki network accessni tekshiring.")

    model = YOLO(model_name)
    video_id = create_video(source_url, title="Tracked source", db_path=db_path)
    roi_zone_id = upsert_roi_zone(roi_name, json.dumps(roi_points), db_path=db_path)
    class_names = model.names

    frame_number = 0
    saved = 0
    batch: list[dict] = []
    seen_tracks: set[tuple[int, int, str, int]] = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_number += 1
        if max_frames and frame_number > max_frames:
            break
        if frame_number % frame_stride != 0:
            continue

        height, width = frame.shape[:2]
        roi_pixels = normalized_polygon_to_pixels(roi_points, width, height)
        timestamp_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        detected_at = datetime.utcnow().isoformat(timespec="seconds")

        results = model.track(frame, persist=True, conf=confidence, verbose=False)
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = str(class_names[class_id])
                if class_name not in TARGET_CLASSES:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                if roi_pixels and not point_in_polygon((cx, cy), roi_pixels):
                    continue
                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is None:
                    continue
                track_key = (video_id, roi_zone_id, class_name, track_id)
                if track_key in seen_tracks:
                    continue
                seen_tracks.add(track_key)
                vehicle_info = recognize_vehicle_model(
                    frame,
                    (x1, y1, x2, y2),
                    class_name=class_name,
                    video_id=video_id,
                    roi_zone_id=roi_zone_id,
                    track_id=track_id,
                )
                if ENABLE_ROBOFLOW_DATASET_EXPORT:
                    save_roboflow_sample(
                        frame,
                        (x1, y1, x2, y2),
                        class_name=class_name,
                        video_id=video_id,
                        roi_zone_id=roi_zone_id,
                        track_id=track_id,
                        frame_number=frame_number,
                    )
                batch.append(
                    {
                        "video_id": video_id,
                        "roi_zone_id": roi_zone_id,
                        "frame_number": frame_number,
                        "timestamp_sec": timestamp_sec,
                        "detected_at": detected_at,
                        "class_name": class_name,
                        "confidence": float(box.conf[0]),
                        "track_id": track_id,
                        **vehicle_info,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "cx": cx,
                        "cy": cy,
                    }
                )

        if len(batch) >= 50:
            inserted = insert_many_detections(batch, db_path)
            saved += inserted
            logger.info("track_video_batch_saved inserted=%s total_saved=%s frame_number=%s", inserted, saved, frame_number)
            batch.clear()
        if progress_callback:
            progress_callback({"frame_number": frame_number, "saved": saved + len(batch)})

    if batch:
        inserted = insert_many_detections(batch, db_path)
        saved += inserted
        logger.info("track_video_final_batch_saved inserted=%s total_saved=%s", inserted, saved)
    cap.release()
    logger.info("track_video_finished saved=%s frames=%s db_path=%s", saved, frame_number, db_path)
    return saved
