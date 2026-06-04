from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from config import DATA_DIR
from logging_utils import get_logger


logger = get_logger("vehicle_recognition")

UNKNOWN_RESULT = {
    "vehicle_make": "unknown",
    "vehicle_model": "unknown",
    "vehicle_color": "unknown",
    "vehicle_body_type": "unknown",
    "vehicle_model_confidence": 0.0,
    "vehicle_recognition_source": "unknown",
    "vehicle_crop_path": None,
}


def _enabled() -> bool:
    return os.getenv("ENABLE_VEHICLE_MODEL_RECOGNITION", "1").strip().lower() not in {"0", "false", "no"}


def _crop_frame(frame: Any, bbox: tuple[float, float, float, float]) -> Any | None:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    height, width = frame.shape[:2]
    pad_x = max(6, int((x2 - x1) * 0.08))
    pad_y = max(6, int((y2 - y1) * 0.08))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None
    return frame[y1:y2, x1:x2]


def save_vehicle_crop(
    frame: Any,
    bbox: tuple[float, float, float, float],
    *,
    video_id: int,
    roi_zone_id: int,
    class_name: str,
    track_id: int,
) -> tuple[Path | None, bytes | None]:
    import cv2

    crop = _crop_frame(frame, bbox)
    if crop is None:
        logger.warning("vehicle_crop_skipped reason=too_small video_id=%s roi_zone_id=%s class=%s track_id=%s", video_id, roi_zone_id, class_name, track_id)
        return None, None
    crop_dir = DATA_DIR / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / f"video{video_id}_roi{roi_zone_id}_{class_name}_track{track_id}.jpg"
    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        logger.error("vehicle_crop_encode_failed path=%s", crop_path)
        return None, None
    image_bytes = bytes(encoded)
    crop_path.write_bytes(image_bytes)
    logger.info("vehicle_crop_saved path=%s class=%s track_id=%s", crop_path, class_name, track_id)
    return crop_path, image_bytes


def _safe_str(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def recognize_vehicle_model(
    frame: Any,
    bbox: tuple[float, float, float, float],
    *,
    class_name: str,
    video_id: int,
    roi_zone_id: int,
    track_id: int,
) -> dict:
    result = dict(UNKNOWN_RESULT)
    crop_path, image_bytes = save_vehicle_crop(
        frame,
        bbox,
        video_id=video_id,
        roi_zone_id=roi_zone_id,
        class_name=class_name,
        track_id=track_id,
    )
    if crop_path:
        result["vehicle_crop_path"] = str(crop_path)

    if not _enabled():
        result["vehicle_recognition_source"] = "disabled"
        logger.info("vehicle_recognition_disabled class=%s track_id=%s", class_name, track_id)
        return result
    if not image_bytes or not os.getenv("OPENAI_API_KEY"):
        result["vehicle_recognition_source"] = "not_configured"
        logger.info("vehicle_recognition_not_configured class=%s track_id=%s", class_name, track_id)
        return result

    try:
        from openai import OpenAI

        data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
        client = OpenAI()
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini")),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You identify vehicle make/model from CCTV crops. "
                        "Return JSON only. If the crop is blurry, partial, too small, "
                        "or the make/model is not clearly visible, use unknown and confidence <= 0.4. "
                        "Do not guess a brand/model from shape alone."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Vehicle class from detector: {class_name}. "
                                "Identify make, model, dominant color, and body type. "
                                "JSON keys: make, model, color, body_type, confidence."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        make = _safe_str(parsed.get("make"))
        model = _safe_str(parsed.get("model"))
        if confidence < 0.55:
            make = "unknown"
            model = "unknown"

        result.update(
            {
                "vehicle_make": make,
                "vehicle_model": model,
                "vehicle_color": _safe_str(parsed.get("color")),
                "vehicle_body_type": _safe_str(parsed.get("body_type")),
                "vehicle_model_confidence": max(0.0, min(1.0, confidence)),
                "vehicle_recognition_source": "openai_vision",
            }
        )
        logger.info(
            "vehicle_recognized class=%s track_id=%s make=%s model=%s color=%s body_type=%s confidence=%.3f",
            class_name,
            track_id,
            result["vehicle_make"],
            result["vehicle_model"],
            result["vehicle_color"],
            result["vehicle_body_type"],
            result["vehicle_model_confidence"],
        )
    except Exception as exc:
        result["vehicle_recognition_source"] = f"error:{type(exc).__name__}"
        logger.exception("vehicle_recognition_failed class=%s track_id=%s error=%s", class_name, track_id, type(exc).__name__)

    return result
