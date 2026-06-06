from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config import (
    DEFAULT_DB_PATH,
    DEFAULT_MODEL,
    DEFAULT_YOUTUBE_URL,
    ENABLE_ROBOFLOW_DATASET_EXPORT,
    ENABLE_VEHICLE_RECOGNITION,
    TARGET_CLASSES,
    TRACKING_DISPLAY_WIDTH,
    TRACKING_FRAME_STRIDE,
    TRACKING_INFERENCE_WIDTH,
)
from database import create_video, insert_many_detections, upsert_roi_zone
from logging_utils import get_logger
from roboflow_dataset import save_sample as save_roboflow_sample
from tracker import point_in_polygon, resolve_video_source
from vehicle_recognition import recognize_vehicle_model


WINDOW_NAME = "ASSBI ROI Tracking"
logger = get_logger("tracking")


def ask(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def select_roi_polygon(frame) -> list[tuple[int, int]]:
    import cv2

    points: list[tuple[int, int]] = []
    preview = frame.copy()

    def redraw() -> None:
        nonlocal preview
        preview = frame.copy()
        if points:
            for point in points:
                cv2.circle(preview, point, 5, (0, 255, 255), -1)
            for index in range(1, len(points)):
                cv2.line(preview, points[index - 1], points[index], (0, 255, 255), 2)
            if len(points) >= 3:
                cv2.line(preview, points[-1], points[0], (0, 180, 255), 2)
        cv2.putText(
            preview,
            "Click ROI points | ENTER: start | R: reset | Q: quit",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        cv2.imshow(WINDOW_NAME, preview)

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 10) and len(points) >= 3:
            return points
        if key in (ord("r"), ord("R")):
            points.clear()
            redraw()
        if key in (ord("q"), ord("Q"), 27):
            raise SystemExit("ROI selection cancelled.")


def normalize_polygon(points: list[tuple[int, int]], width: int, height: int) -> list[list[float]]:
    return [[round(x / width, 6), round(y / height, 6)] for x, y in points]


def draw_polygon(frame, points: list[tuple[int, int]]) -> None:
    import cv2

    if len(points) < 2:
        return
    for index in range(len(points)):
        cv2.line(frame, points[index], points[(index + 1) % len(points)], (0, 180, 255), 2)


def resize_for_width(frame, target_width: int):
    import cv2

    if target_width <= 0 or frame.shape[1] <= target_width:
        return frame, 1.0
    scale = target_width / frame.shape[1]
    target_height = max(1, int(frame.shape[0] * scale))
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def scale_bbox_to_original(bbox: tuple[float, float, float, float], scale: float) -> tuple[float, float, float, float]:
    if scale == 1.0:
        return bbox
    return tuple(value / scale for value in bbox)


def validate_model_path(model_name: str) -> None:
    if ("/" in model_name or model_name.endswith(".pt")) and not Path(model_name).exists():
        raise FileNotFoundError(
            f"YOLO model topilmadi: {model_name}. "
            "Agar Roboflow model hali yo'q bo'lsa ASSBI_YOLO_MODEL bermang va default yolo11n.pt ishlating. "
            "Agar model bor bo'lsa, .pt faylning to'liq pathini yozing."
        )


def run_tracking() -> None:
    import cv2
    from ultralytics import YOLO

    print("ASSBI interactive vehicle tracking")
    print("Person class bazaga yozilmaydi. Target classes:", ", ".join(sorted(TARGET_CLASSES)))
    logger.info("interactive_tracking_started target_classes=%s", ",".join(sorted(TARGET_CLASSES)))

    source_url = ask("YouTube/video URL", DEFAULT_YOUTUBE_URL)
    roi_name = f"roi_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    model_name = DEFAULT_MODEL
    confidence = 0.35
    frame_stride = TRACKING_FRAME_STRIDE
    max_frames = 0
    print(
        "Auto settings:",
        f"model={model_name}",
        f"confidence={confidence}",
        f"frame_stride={frame_stride}",
        f"inference_width={TRACKING_INFERENCE_WIDTH or 'original'}",
        f"vehicle_recognition={'on' if ENABLE_VEHICLE_RECOGNITION else 'off'}",
        "max_frames=unlimited",
    )
    validate_model_path(model_name)
    logger.info(
        "tracking_settings source_url=%s roi_name=%s model=%s confidence=%s frame_stride=%s max_frames=unlimited",
        source_url,
        roi_name,
        model_name,
        confidence,
        frame_stride,
    )

    stream_url = resolve_video_source(source_url)
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        logger.error("video_open_failed source_url=%s", source_url)
        raise RuntimeError("Video ochilmadi. URL yoki network accessni tekshiring.")

    ok, first_frame = cap.read()
    if not ok:
        logger.error("first_frame_read_failed source_url=%s", source_url)
        raise RuntimeError("Video frame o'qilmadi.")

    height, width = first_frame.shape[:2]
    roi_pixels = select_roi_polygon(first_frame)
    roi_normalized = normalize_polygon(roi_pixels, width, height)
    roi_zone_id = upsert_roi_zone(roi_name, json.dumps(roi_normalized), DEFAULT_DB_PATH)
    video_id = create_video(source_url, title="Tracked source", db_path=DEFAULT_DB_PATH)
    logger.info(
        "roi_selected roi_name=%s roi_zone_id=%s video_id=%s points=%s frame_width=%s frame_height=%s",
        roi_name,
        roi_zone_id,
        video_id,
        len(roi_pixels),
        width,
        height,
    )

    model = YOLO(model_name)
    class_names = model.names
    saved = 0
    frame_number = 1
    batch: list[dict] = []
    seen_tracks: set[tuple[int, int, str, int]] = set()
    display_ids: dict[tuple[int, int, str, int], int] = {}
    pending_unique = 0
    start_message = "Tracking started. Q: stop"
    print(start_message)
    logger.info("model_loaded model=%s video_id=%s roi_zone_id=%s", model_name, video_id, roi_zone_id)

    while True:
        frame = first_frame if frame_number == 1 else None
        if frame is None:
            ok, frame = cap.read()
            if not ok:
                break
        if max_frames and frame_number > max_frames:
            break

        if frame_number % frame_stride == 0:
            timestamp_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            detected_at = datetime.utcnow().isoformat(timespec="seconds")
            inference_frame, inference_scale = resize_for_width(frame, TRACKING_INFERENCE_WIDTH)
            results = model.track(inference_frame, persist=True, conf=confidence, verbose=False)

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = str(class_names[class_id])
                    if class_name not in TARGET_CLASSES:
                        continue
                    x1, y1, x2, y2 = scale_bbox_to_original(tuple(float(v) for v in box.xyxy[0]), inference_scale)
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    if not point_in_polygon((cx, cy), roi_pixels):
                        continue

                    track_id = int(box.id[0]) if box.id is not None else None
                    if track_id is None:
                        continue
                    track_key = (video_id, roi_zone_id, class_name, track_id)
                    is_new_track = track_key not in seen_tracks
                    if is_new_track:
                        seen_tracks.add(track_key)
                        pending_unique += 1
                        display_ids[track_key] = saved + pending_unique
                        vehicle_info = {}
                        if ENABLE_VEHICLE_RECOGNITION:
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

                    display_id = display_ids.get(track_key, saved + pending_unique)
                    label = f"{class_name} #{display_id}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            if len(batch) >= 50:
                inserted = insert_many_detections(batch, DEFAULT_DB_PATH)
                saved += inserted
                logger.info("tracking_batch_saved inserted=%s total_saved=%s frame_number=%s", inserted, saved, frame_number)
                batch.clear()
                pending_unique = 0

        draw_polygon(frame, roi_pixels)
        cv2.putText(frame, f"Unique saved: {saved + pending_unique} | Frame: {frame_number}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        display_frame, _ = resize_for_width(frame, TRACKING_DISPLAY_WIDTH)
        cv2.imshow(WINDOW_NAME, display_frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break
        frame_number += 1

    if batch:
        inserted = insert_many_detections(batch, DEFAULT_DB_PATH)
        saved += inserted
        logger.info("tracking_final_batch_saved inserted=%s total_saved=%s", inserted, saved)
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} detections into {DEFAULT_DB_PATH}")
    logger.info("interactive_tracking_finished saved=%s db_path=%s frames=%s", saved, DEFAULT_DB_PATH, frame_number)


if __name__ == "__main__":
    run_tracking()
