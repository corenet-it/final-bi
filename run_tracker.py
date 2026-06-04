from __future__ import annotations

import argparse
import json

from config import DEFAULT_DB_PATH, DEFAULT_MODEL, DEFAULT_YOUTUBE_URL
from tracker import track_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO tracking and save detections to SQLite.")
    parser.add_argument("--url", default=DEFAULT_YOUTUBE_URL)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--roi-name", default="main_road_roi")
    parser.add_argument(
        "--roi",
        default="[[0.18,0.28],[0.84,0.24],[0.90,0.86],[0.12,0.82]]",
        help="Normalized ROI polygon JSON, e.g. [[0.1,0.2],[0.9,0.2],[0.9,0.8],[0.1,0.8]]",
    )
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args()

    count = track_video(
        source_url=args.url,
        db_path=args.db,
        roi_name=args.roi_name,
        roi_points=json.loads(args.roi),
        model_name=args.model,
        confidence=args.confidence,
        frame_stride=args.stride,
        max_frames=args.max_frames,
    )
    print(f"Saved {count} detections into {args.db}")


if __name__ == "__main__":
    main()
