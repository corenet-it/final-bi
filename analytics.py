from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from config import TARGET_CLASSES
from database import fetch_detections


def load_detection_frame(db_path: Path | str) -> pd.DataFrame:
    rows = fetch_detections(db_path)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["class_name"].isin(TARGET_CLASSES)].copy()
    if df.empty:
        return df
    df["detected_at"] = pd.to_datetime(df["detected_at"], errors="coerce")
    df["hour"] = df["detected_at"].dt.floor("h")
    return df


def kpi_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total": 0,
            "unique_tracks": 0,
            "top_class": "-",
            "avg_confidence": 0.0,
            "videos": 0,
            "roi_zones": 0,
            "recognized_models": 0,
            "unknown_models": 0,
            "recognition_rate": 0.0,
            "first_detection": None,
            "last_detection": None,
        }
    top_class = df["class_name"].value_counts().idxmax()
    unique_cols = ["video_id", "roi_zone_id", "class_name", "track_id"]
    unique_tracks = df.dropna(subset=["track_id"]).drop_duplicates(unique_cols).shape[0]
    recognized = pd.Series(dtype=bool)
    if {"vehicle_make", "vehicle_model"}.issubset(df.columns):
        recognized = (df["vehicle_make"].fillna("unknown") != "unknown") | (df["vehicle_model"].fillna("unknown") != "unknown")
    return {
        "total": int(len(df)),
        "unique_tracks": int(unique_tracks),
        "top_class": str(top_class),
        "avg_confidence": float(df["confidence"].mean()),
        "videos": int(df["video_id"].dropna().nunique()) if "video_id" in df.columns else 0,
        "roi_zones": int(df["roi_zone_id"].dropna().nunique()) if "roi_zone_id" in df.columns else 0,
        "recognized_models": int(recognized.sum()) if not recognized.empty else 0,
        "unknown_models": int((~recognized).sum()) if not recognized.empty else 0,
        "recognition_rate": float(recognized.mean()) if not recognized.empty else 0.0,
        "first_detection": df["detected_at"].min(),
        "last_detection": df["detected_at"].max(),
    }


def class_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["class_name", "count"])
    return df.groupby("class_name").size().reset_index(name="count").sort_values("count", ascending=False)


def unique_class_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "track_id" not in df.columns:
        return pd.DataFrame(columns=["class_name", "unique_tracks"])
    unique_cols = ["video_id", "roi_zone_id", "class_name", "track_id"]
    unique = df.dropna(subset=["track_id"]).drop_duplicates(unique_cols)
    if unique.empty:
        return pd.DataFrame(columns=["class_name", "unique_tracks"])
    return unique.groupby("class_name").size().reset_index(name="unique_tracks").sort_values("unique_tracks", ascending=False)


def hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["hour", "class_name", "count"])
    return df.groupby(["hour", "class_name"]).size().reset_index(name="count").sort_values("hour")


def roi_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["roi_name", "count", "unique_tracks"])
    data = df.copy()
    data["roi_name"] = data.get("roi_name", pd.Series(index=data.index, dtype=str)).fillna("unknown")
    grouped = data.groupby("roi_name").size().reset_index(name="count")
    unique = (
        data.dropna(subset=["track_id"])
        .drop_duplicates(["video_id", "roi_zone_id", "class_name", "track_id"])
        .groupby("roi_name")
        .size()
        .reset_index(name="unique_tracks")
    )
    return grouped.merge(unique, on="roi_name", how="left").fillna({"unique_tracks": 0}).sort_values("count", ascending=False)


def video_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["source_url", "count", "unique_tracks"])
    data = df.copy()
    data["source_url"] = data.get("source_url", pd.Series(index=data.index, dtype=str)).fillna("unknown")
    grouped = data.groupby("source_url").size().reset_index(name="count")
    unique = (
        data.dropna(subset=["track_id"])
        .drop_duplicates(["video_id", "roi_zone_id", "class_name", "track_id"])
        .groupby("source_url")
        .size()
        .reset_index(name="unique_tracks")
    )
    return grouped.merge(unique, on="source_url", how="left").fillna({"unique_tracks": 0}).sort_values("count", ascending=False)


def vehicle_model_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "vehicle_model" not in df.columns:
        return pd.DataFrame(columns=["vehicle_label", "count"])
    labelled = df.copy()
    labelled["vehicle_make"] = labelled["vehicle_make"].fillna("unknown")
    labelled["vehicle_model"] = labelled["vehicle_model"].fillna("unknown")
    labelled = labelled[(labelled["vehicle_make"] != "unknown") | (labelled["vehicle_model"] != "unknown")]
    if labelled.empty:
        return pd.DataFrame(columns=["vehicle_label", "count"])
    labelled["vehicle_label"] = (labelled["vehicle_make"] + " " + labelled["vehicle_model"]).str.strip()
    return labelled.groupby("vehicle_label").size().reset_index(name="count").sort_values("count", ascending=False)


def vehicle_color_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "vehicle_color" not in df.columns:
        return pd.DataFrame(columns=["vehicle_color", "count"])
    data = df.copy()
    data["vehicle_color"] = data["vehicle_color"].fillna("unknown").replace("", "unknown")
    return data.groupby("vehicle_color").size().reset_index(name="count").sort_values("count", ascending=False)


def vehicle_body_type_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "vehicle_body_type" not in df.columns:
        return pd.DataFrame(columns=["vehicle_body_type", "count"])
    data = df.copy()
    data["vehicle_body_type"] = data["vehicle_body_type"].fillna("unknown").replace("", "unknown")
    return data.groupby("vehicle_body_type").size().reset_index(name="count").sort_values("count", ascending=False)


def confidence_by_class(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["class_name", "avg_confidence", "min_confidence", "max_confidence", "count"])
    return (
        df.groupby("class_name")["confidence"]
        .agg(avg_confidence="mean", min_confidence="min", max_confidence="max", count="size")
        .reset_index()
        .sort_values("avg_confidence", ascending=False)
    )


def low_confidence_detections(df: pd.DataFrame, threshold: float = 0.45, limit: int = 20) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    cols = [
        "detected_at",
        "class_name",
        "confidence",
        "track_id",
        "roi_name",
        "vehicle_make",
        "vehicle_model",
        "vehicle_color",
    ]
    existing_cols = [col for col in cols if col in df.columns]
    return df[df["confidence"] <= threshold].sort_values("confidence").head(limit)[existing_cols]


def filtered_detections(
    df: pd.DataFrame,
    *,
    class_name: str | None = None,
    roi_name: str | None = None,
    vehicle_color: str | None = None,
    vehicle_make: str | None = None,
    limit: int = 50,
) -> pd.DataFrame:
    if df.empty:
        return df
    data = df.copy()
    if class_name:
        data = data[data["class_name"].str.lower() == class_name.lower()]
    if roi_name and "roi_name" in data.columns:
        data = data[data["roi_name"].fillna("").str.lower().str.contains(roi_name.lower(), regex=False)]
    if vehicle_color and "vehicle_color" in data.columns:
        data = data[data["vehicle_color"].fillna("").str.lower().str.contains(vehicle_color.lower(), regex=False)]
    if vehicle_make and "vehicle_make" in data.columns:
        data = data[data["vehicle_make"].fillna("").str.lower().str.contains(vehicle_make.lower(), regex=False)]
    return data.head(limit)


def time_range_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"first_detection": None, "last_detection": None, "duration_minutes": 0.0}
    first = df["detected_at"].min()
    last = df["detected_at"].max()
    duration = (last - first).total_seconds() / 60 if pd.notna(first) and pd.notna(last) else 0.0
    return {
        "first_detection": first,
        "last_detection": last,
        "duration_minutes": float(max(duration, 0.0)),
    }


def recommended_chart_names(question: str) -> list[str]:
    q = question.lower()
    charts: list[str] = []
    if any(word in q for word in ["model", "brand", "make", "marka"]):
        charts.append("vehicle_models")
    if any(word in q for word in ["rang", "color"]):
        charts.append("vehicle_colors")
    if any(word in q for word in ["body", "type", "tur", "sedan", "suv"]):
        charts.append("body_types")
    if any(word in q for word in ["roi", "zona", "hudud", "chegara"]):
        charts.append("roi")
    if any(word in q for word in ["vaqt", "time", "soat", "trend", "qachon"]):
        charts.append("trend")
    if any(word in q for word in ["confidence", "aniqlik", "ishonch"]):
        charts.append("confidence")
    if any(word in q for word in ["unique", "track", "id", "takror"]):
        charts.append("unique_classes")
    if not charts:
        charts = ["class_counts", "trend", "unique_classes"]
    return charts[:4]


def local_answer(question: str, df: pd.DataFrame) -> str:
    if df.empty:
        return "Bazaga hali tracking ma'lumotlari yig'ilmagan."

    q = question.lower()
    tokens = {token.strip(" ?!.,:;()[]{}'\"") for token in q.split()}
    counts = class_counts(df)
    total = len(df)
    top = counts.iloc[0]

    class_hits = [c for c in sorted(df["class_name"].dropna().unique()) if re.search(rf"\b{re.escape(c)}\b", q)]
    if len(class_hits) > 1:
        unique = unique_class_counts(df)
        unique_map = dict(zip(unique["class_name"], unique["unique_tracks"])) if not unique.empty else {}
        parts = []
        for cls in class_hits:
            value = int((df["class_name"] == cls).sum())
            unique_value = int(unique_map.get(cls, 0))
            parts.append(f"{cls}: {value} detection, {unique_value} unique track ({value / total:.1%})")
        leader = max(class_hits, key=lambda cls: int((df["class_name"] == cls).sum()))
        return f"Solishtirish: {'; '.join(parts)}. Ko'proq uchragani: {leader}."
    if class_hits:
        cls = class_hits[0]
        value = int((df["class_name"] == cls).sum())
        return f"{cls} bo'yicha {value} ta detection topildi. Bu umumiy {total} ta detection ichida {value / total:.1%} ulush beradi."

    if tokens & {"top", "eng", "ko'p", "kop"}:
        return f"Eng ko'p uchragan obyekt: {top['class_name']} ({int(top['count'])} ta detection)."

    if "confidence" in q or "aniqlik" in q:
        conf = confidence_by_class(df)
        best = conf.iloc[0]
        return (
            f"O'rtacha model confidence qiymati {df['confidence'].mean():.2f}. "
            f"Minimal: {df['confidence'].min():.2f}, maksimal: {df['confidence'].max():.2f}. "
            f"Eng yuqori o'rtacha confidence: {best['class_name']} ({best['avg_confidence']:.2f})."
        )

    if "track" in q or "unique" in q:
        return f"Unique track soni: {kpi_summary(df)['unique_tracks']}."

    if "roi" in q or "zona" in q or "hudud" in q:
        rois = roi_counts(df)
        top_roi = rois.iloc[0]
        return f"Eng faol ROI: {top_roi['roi_name']} ({int(top_roi['count'])} detection, {int(top_roi['unique_tracks'])} unique track)."

    if "rang" in q or "color" in q:
        colors = vehicle_color_counts(df)
        top_color = colors.iloc[0]
        return f"Eng ko'p uchragan rang: {top_color['vehicle_color']} ({int(top_color['count'])} ta)."

    if "model" in q or "brand" in q or "marka" in q:
        models = vehicle_model_counts(df)
        if models.empty:
            return "Vehicle make/model bo'yicha aniq ma'lumot topilmadi. OpenAI vision yoqilgan bo'lsa keyingi trackinglarda model yoziladi."
        top_model = models.iloc[0]
        return f"Eng ko'p tanilgan vehicle model: {top_model['vehicle_label']} ({int(top_model['count'])} ta)."

    if "recent" in q or "oxirgi" in q:
        recent = df.head(5)
        items = ", ".join(f"{row.class_name} #{int(row.track_id) if pd.notna(row.track_id) else '-'}" for row in recent.itertuples())
        return f"Oxirgi detectionlar: {items}."

    return (
        f"Bazada {total} ta detection bor. Eng ko'p obyekt {top['class_name']} "
        f"({int(top['count'])} ta). Unique tracklar: {kpi_summary(df)['unique_tracks']}."
    )


def compact_context_for_llm(df: pd.DataFrame) -> str:
    if df.empty:
        return "No detections are available."
    counts = class_counts(df).to_dict(orient="records")
    unique_counts = unique_class_counts(df).to_dict(orient="records")
    hourly = hourly_counts(df).tail(24).to_dict(orient="records")
    models = vehicle_model_counts(df).head(20).to_dict(orient="records")
    rois = roi_counts(df).head(10).to_dict(orient="records")
    colors = vehicle_color_counts(df).head(10).to_dict(orient="records")
    conf = confidence_by_class(df).to_dict(orient="records")
    summary = kpi_summary(df)
    return (
        f"KPIs: {summary}\nClass counts: {counts}\nUnique class counts: {unique_counts}\n"
        f"Vehicle model counts: {models}\nROI counts: {rois}\nColor counts: {colors}\n"
        f"Confidence by class: {conf}\nRecent hourly counts: {hourly}"
    )
