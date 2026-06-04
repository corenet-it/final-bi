from __future__ import annotations

import json
import os
from pathlib import Path

from analytics import (
    class_counts,
    compact_context_for_llm,
    confidence_by_class,
    filtered_detections,
    hourly_counts,
    kpi_summary,
    load_detection_frame,
    local_answer,
    low_confidence_detections,
    roi_counts,
    time_range_summary,
    unique_class_counts,
    vehicle_body_type_counts,
    vehicle_color_counts,
    vehicle_model_counts,
    video_counts,
)
from config import OPENAI_MODEL
from database import fetch_chat_history, save_chat_query
from logging_utils import get_logger


logger = get_logger("ai_assistant")

SYSTEM_PROMPT = """
You are a Business Intelligence assistant for an AI smart surveillance dashboard.
Answer in Uzbek unless the user asks otherwise. Use only the provided SQLite
analytics context. Be concise, mention relevant KPI values, and do not invent
events that are not present in the data.
Use tools to retrieve the exact data needed before answering. If the user asks
about charts, trends, comparisons, ROI, model, color, or confidence, call the
matching tool first and explain which chart is most useful.
You may hold a short conversation, remember recent user questions, and answer
follow-up questions, but stay inside the ASSBI project scope: vehicle tracking,
YOLO/OpenCV, ROI, SQLite data, Streamlit dashboard, BI analytics, model quality,
privacy/security, and project usage. If the user asks an unrelated off-topic
question, politely redirect them to the surveillance BI data or project.
""".strip()

DOMAIN_KEYWORDS = {
    "assbi",
    "dashboard",
    "streamlit",
    "sqlite",
    "database",
    "data",
    "chart",
    "diagram",
    "analytics",
    "analitika",
    "kpi",
    "object",
    "obyekt",
    "jism",
    "jismlar",
    "aniqlandi",
    "topildi",
    "saqlandi",
    "saved",
    "count",
    "counts",
    "son",
    "nechta",
    "qancha",
    "jami",
    "umumiy",
    "total",
    "statistika",
    "hisobot",
    "natija",
    "results",
    "detections",
    "tracking",
    "track",
    "detection",
    "detect",
    "yolo",
    "opencv",
    "roi",
    "zona",
    "hudud",
    "chegara",
    "vehicle",
    "car",
    "mashina",
    "motorcycle",
    "motobike",
    "truck",
    "bus",
    "bicycle",
    "confidence",
    "aniqlik",
    "model",
    "rang",
    "color",
    "privacy",
    "security",
    "xavfsizlik",
    "memory",
    "oldingi",
    "avvalgi",
    "savol",
    "suhbat",
    "qanday ishlat",
    "run",
    "url",
}

SMALL_TALK = {"salom", "rahmat", "tushunarli", "ok", "ha", "yoq", "yo'q", "yaxshi"}
ANALYTICS_INTENT_WORDS = {
    "nechta",
    "qancha",
    "qancha?",
    "soni",
    "son",
    "jami",
    "umumiy",
    "count",
    "total",
    "aniqlandi",
    "topildi",
    "saqlandi",
    "ko'p",
    "eng",
    "kam",
    "solishtir",
    "taqqosla",
    "chart",
    "diagram",
    "diagramma",
    "statistika",
    "hisobot",
}


def _is_in_scope(question: str, history: list[dict] | None = None) -> bool:
    q = question.lower().strip()
    if not q:
        return True
    if any(word in q for word in DOMAIN_KEYWORDS):
        return True
    tokens = {token.strip(" ?!.,:;()[]{}'\"") for token in q.split()}
    if tokens & ANALYTICS_INTENT_WORDS:
        return True
    if q.endswith("?") and len(q.split()) <= 8:
        return True
    if q in SMALL_TALK or len(q.split()) <= 3 and any(word in q for word in SMALL_TALK):
        return True
    if history and any(word in q for word in ["shu", "oldingi", "avvalgi", "davom", "yana", "solishtir", "chart", "diagram"]):
        return True
    return False


def _memory_text(history: list[dict], limit: int = 6) -> str:
    if not history:
        return "No previous conversation."
    recent = history[-limit:]
    lines = []
    for i, item in enumerate(recent, start=1):
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip().replace("\n", " ")
        if len(answer) > 500:
            answer = answer[:500] + "..."
        lines.append(f"{i}. User: {question}\n   Assistant: {answer}")
    return "\n".join(lines)


def _last_user_topic(history: list[dict]) -> str:
    if not history:
        return ""
    for item in reversed(history):
        question = str(item.get("question", "")).strip()
        if question:
            return question
    return ""


def _data_quality_summary(df) -> dict:
    if df.empty:
        return {"status": "empty", "message": "No detections are available."}
    summary = kpi_summary(df)
    low = low_confidence_detections(df, threshold=0.45, limit=10)
    missing_track = int(df["track_id"].isna().sum()) if "track_id" in df.columns else 0
    duplicate_unique_cols = ["video_id", "roi_zone_id", "class_name", "track_id"]
    duplicate_rows = 0
    if set(duplicate_unique_cols).issubset(df.columns):
        duplicate_rows = int(df.dropna(subset=["track_id"]).duplicated(duplicate_unique_cols).sum())
    return {
        "total": summary["total"],
        "unique_tracks": summary["unique_tracks"],
        "avg_confidence": round(summary["avg_confidence"], 4),
        "low_confidence_count_at_0_45": int(len(low)),
        "missing_track_id_rows": missing_track,
        "duplicate_unique_track_rows": duplicate_rows,
        "person_rows": int((df["class_name"] == "person").sum()) if "class_name" in df.columns else 0,
    }


def _frame_json(df, *, limit: int | None = None) -> str:
    data = df.copy()
    if limit is not None:
        data = data.head(limit)
    for column in data.columns:
        if str(data[column].dtype).startswith("datetime64"):
            data[column] = data[column].astype(str)
    return data.to_json(orient="records")


def _tool_result(name: str, db_path: Path | str, arguments: str | None) -> str:
    logger.info("assistant_tool_called name=%s arguments=%s", name, arguments or "{}")
    df = load_detection_frame(db_path)
    args = json.loads(arguments or "{}")

    if name == "get_short_memory":
        return json.dumps(fetch_chat_history(db_path, limit=int(args.get("limit", 8))), default=str)
    if name == "get_system_capabilities":
        return json.dumps(
            {
                "tracking": "tracking.py asks for a URL, opens video, lets the user draw ROI, and saves vehicle-only unique tracks.",
                "dashboard": "app.py provides Streamlit Overview, Ask Data, and Database tabs.",
                "database": "SQLite stores videos, ROI zones, detections, track events, and chat query memory.",
                "assistant": "Ask Data uses OpenAI function calling when configured, otherwise local rule-based analytics answers.",
                "scope": "The assistant can discuss ASSBI data, vehicle analytics, ROI, model quality, privacy/security, and project usage. It avoids unrelated off-topic chat.",
            },
            default=str,
        )
    if name == "get_data_quality_summary":
        return json.dumps(_data_quality_summary(df), default=str)
    if name == "get_kpi_summary":
        return json.dumps(kpi_summary(df), default=str)
    if name == "get_class_counts":
        return _frame_json(class_counts(df))
    if name == "get_unique_class_counts":
        return _frame_json(unique_class_counts(df))
    if name == "get_hourly_counts":
        hourly = hourly_counts(df).copy()
        if "hour" in hourly.columns:
            hourly["hour"] = hourly["hour"].astype(str)
        return _frame_json(hourly.tail(int(args.get("limit", 48))))
    if name == "get_roi_counts":
        return _frame_json(roi_counts(df).head(int(args.get("limit", 30))))
    if name == "get_video_counts":
        return _frame_json(video_counts(df).head(int(args.get("limit", 20))))
    if name == "get_vehicle_model_counts":
        return _frame_json(vehicle_model_counts(df).head(int(args.get("limit", 20))))
    if name == "get_vehicle_color_counts":
        return _frame_json(vehicle_color_counts(df).head(int(args.get("limit", 20))))
    if name == "get_vehicle_body_type_counts":
        return _frame_json(vehicle_body_type_counts(df).head(int(args.get("limit", 20))))
    if name == "get_confidence_by_class":
        return _frame_json(confidence_by_class(df))
    if name == "get_time_range_summary":
        return json.dumps(time_range_summary(df), default=str)
    if name == "get_low_confidence_detections":
        return _frame_json(
            low_confidence_detections(
                df,
                threshold=float(args.get("threshold", 0.45)),
                limit=int(args.get("limit", 20)),
            )
        )
    if name == "search_detections":
        return _frame_json(
            filtered_detections(
                df,
                class_name=args.get("class_name"),
                roi_name=args.get("roi_name"),
                vehicle_color=args.get("vehicle_color"),
                vehicle_make=args.get("vehicle_make"),
                limit=int(args.get("limit", 50)),
            )
        )
    if name == "get_recent_detections":
        limit = int(args.get("limit", 20))
        return _frame_json(df, limit=limit)
    return json.dumps({"error": f"Unknown tool: {name}"})


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_short_memory",
            "description": "Return recent Ask Data conversation history for follow-up questions.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 20}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_capabilities",
            "description": "Explain what the ASSBI tracker, database, dashboard, and chatbot can do.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_data_quality_summary",
            "description": "Return data quality summary including confidence, duplicate track rows, missing track IDs, and person rows.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_kpi_summary",
            "description": "Return total detections, unique tracks, top class, and average confidence.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_class_counts",
            "description": "Return detection counts grouped by vehicle class.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_unique_class_counts",
            "description": "Return unique tracked vehicle counts grouped by class. Use for non-duplicate object counts.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hourly_counts",
            "description": "Return recent hourly vehicle detection counts for trend analysis.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 200}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_roi_counts",
            "description": "Return detection and unique track counts grouped by ROI zone.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_video_counts",
            "description": "Return detection and unique track counts grouped by video source URL.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_detections",
            "description": "Return recent detection rows for audit-style questions.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_detections",
            "description": "Return detection rows filtered by class, ROI name, vehicle color, or vehicle make.",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_name": {"type": "string", "enum": ["bicycle", "car", "motorcycle", "bus", "truck"]},
                    "roi_name": {"type": "string"},
                    "vehicle_color": {"type": "string"},
                    "vehicle_make": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vehicle_model_counts",
            "description": "Return recognized vehicle make/model counts from saved detection rows.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vehicle_color_counts",
            "description": "Return saved vehicle color counts.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vehicle_body_type_counts",
            "description": "Return saved vehicle body type counts, such as sedan, SUV, truck, or unknown.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_confidence_by_class",
            "description": "Return average, min, max confidence grouped by vehicle class.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_low_confidence_detections",
            "description": "Return detections with low YOLO confidence for quality review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {"type": "number", "minimum": 0.01, "maximum": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time_range_summary",
            "description": "Return first detection, last detection, and total covered duration.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


def _local_memory_answer(question: str, df, history: list[dict]) -> str:
    q = question.lower().strip()
    if q in SMALL_TALK or any(q.startswith(word) for word in ["salom", "rahmat", "ok", "yaxshi"]):
        summary = kpi_summary(df)
        if df.empty:
            return "Salom. Men ASSBI dashboard bo'yicha yordam beraman. Hozir bazada tracking ma'lumoti yo'q, avval `python code/tracking.py` orqali video tracking qiling."
        return (
            "Salom. Men ASSBI vehicle analytics bo'yicha yordam beraman. "
            f"Hozir bazada {summary['total']:,} ta detection va {summary['unique_tracks']:,} ta unique track bor. "
            "ROI, class, confidence, model/rang yoki dashboard bo'yicha savol berishingiz mumkin."
        )

    if any(word in q for word in ["oldingi", "avvalgi", "shu", "davom", "esla", "memory"]):
        topic = _last_user_topic(history)
        if topic:
            base = local_answer(topic, df)
            follow_up = local_answer(question, df)
            return (
                f"Oldingi kontekst: \"{topic}\".\n"
                f"Shu kontekst bo'yicha asosiy javob: {base}\n"
                f"Hozirgi savolga mos qo'shimcha: {follow_up}"
            )
        return "Hozircha eslab qolingan oldingi analytics savol yo'q. ROI, class, confidence yoki unique track haqida savol bering."

    return local_answer(question, df)


def answer_question(question: str, db_path: Path | str, use_openai: bool = True) -> str:
    df = load_detection_frame(db_path)
    history = fetch_chat_history(db_path, limit=8)
    logger.info("assistant_question_received chars=%s use_openai=%s history=%s", len(question), use_openai, len(history))

    if not _is_in_scope(question, history):
        answer = (
            "Bu savol ASSBI smart surveillance BI loyihasi doirasidan tashqarida. "
            "Men vehicle tracking, ROI, YOLO/OpenCV, SQLite database, Streamlit dashboard, "
            "confidence, data quality, privacy/security yoki loyiha ishlatish bo'yicha yordam bera olaman."
        )
        save_chat_query(question, answer, db_path)
        logger.info("assistant_question_rejected_offtopic chars=%s", len(question))
        return answer

    fallback = _local_memory_answer(question, df, history)

    if not use_openai or not os.getenv("OPENAI_API_KEY"):
        save_chat_query(question, fallback, db_path)
        logger.info("assistant_answered_local reason=%s", "disabled" if not use_openai else "missing_api_key")
        return fallback

    try:
        from openai import OpenAI

        client = OpenAI()
        memory = _memory_text(history)
        compact_data = compact_context_for_llm(df)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    "Short memory from recent Ask Data messages:\n"
                    f"{memory}\n\n"
                    "Compact current database context:\n"
                    f"{compact_data}\n\n"
                    "Use the memory only for conversation continuity. Use tools for exact values when the user asks for analytics."
                ),
            },
            {"role": "user", "content": question},
        ]
        answer = ""
        for _ in range(3):
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", OPENAI_MODEL),
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
            )
            message = response.choices[0].message
            if not message.tool_calls:
                answer = (message.content or "").strip()
                logger.info("assistant_openai_answered_without_tools chars=%s", len(answer))
                break
            messages.append(message.model_dump(exclude_none=True))
            for tool_call in message.tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": _tool_result(tool_call.function.name, db_path, tool_call.function.arguments),
                    }
                )
        if not answer:
            final_response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", OPENAI_MODEL),
                messages=messages,
                temperature=0,
            )
            answer = (final_response.choices[0].message.content or "").strip()
        if not answer:
            answer = fallback
    except Exception as exc:
        answer = f"{fallback}\n\nOpenAI javobi olinmadi: {exc}"
        logger.exception("assistant_openai_failed error=%s", type(exc).__name__)

    save_chat_query(question, answer, db_path)
    logger.info("assistant_answer_saved chars=%s", len(answer))
    return answer
