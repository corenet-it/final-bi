from __future__ import annotations

from pathlib import Path

import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from ai_assistant import answer_question
from analytics import (
    class_counts,
    confidence_by_class,
    hourly_counts,
    kpi_summary,
    load_detection_frame,
    recommended_chart_names,
    roi_counts,
    unique_class_counts,
    vehicle_body_type_counts,
    vehicle_color_counts,
    vehicle_model_counts,
    video_counts,
)
from config import DEFAULT_DB_PATH
from database import clear_chat_history, fetch_chat_history, init_db
from logging_utils import get_logger

load_dotenv()
logger = get_logger("app")

st.set_page_config(page_title="Smart Surveillance BI", page_icon="BI", layout="wide")

st.sidebar.title("ASSBI Control")
db_path = Path(st.sidebar.text_input("SQLite database", value=str(DEFAULT_DB_PATH)))
init_db(db_path)
logger.info("dashboard_loaded db_path=%s", db_path)

if st.sidebar.button("Refresh analytics"):
    logger.info("dashboard_refresh_clicked db_path=%s", db_path)
    st.rerun()

df = load_detection_frame(db_path)
kpis = kpi_summary(df)

st.title("AI Smart Surveillance BI Dashboard")
st.caption("SQLite analytics, Streamlit reporting, and AI assistant for vehicle tracking data")

tab_overview, tab_assistant, tab_data = st.tabs(["Overview", "Ask Data", "Database"])


def _format_dt(value) -> str:
    if value is None or str(value) in {"NaT", "nan"}:
        return "-"
    return str(value).replace("T", " ")[:19]


def render_recommended_charts(question: str, data) -> None:
    chart_names = recommended_chart_names(question)
    for index, chart_name in enumerate(chart_names):
        key = f"assistant_{chart_name}_{index}"
        if chart_name == "vehicle_models":
            model_counts = vehicle_model_counts(data)
            if not model_counts.empty:
                st.plotly_chart(
                    px.bar(model_counts.head(15), x="vehicle_label", y="count", title="Vehicle Make / Model"),
                    width="stretch",
                    key=key,
                )
        elif chart_name == "vehicle_colors":
            colors = vehicle_color_counts(data)
            if not colors.empty:
                st.plotly_chart(
                    px.bar(colors.head(15), x="vehicle_color", y="count", title="Vehicle Colors"),
                    width="stretch",
                    key=key,
                )
        elif chart_name == "body_types":
            bodies = vehicle_body_type_counts(data)
            if not bodies.empty:
                st.plotly_chart(
                    px.bar(bodies.head(15), x="vehicle_body_type", y="count", title="Vehicle Body Types"),
                    width="stretch",
                    key=key,
                )
        elif chart_name == "roi":
            rois = roi_counts(data)
            if not rois.empty:
                st.plotly_chart(
                    px.bar(rois.head(15), x="roi_name", y=["count", "unique_tracks"], barmode="group", title="ROI Performance"),
                    width="stretch",
                    key=key,
                )
        elif chart_name == "trend":
            hourly = hourly_counts(data)
            if not hourly.empty:
                st.plotly_chart(
                    px.line(hourly, x="hour", y="count", color="class_name", markers=True, title="Detection Trend"),
                    width="stretch",
                    key=key,
                )
        elif chart_name == "confidence":
            conf = confidence_by_class(data)
            if not conf.empty:
                st.plotly_chart(
                    px.bar(conf, x="class_name", y="avg_confidence", title="Average Confidence by Class"),
                    width="stretch",
                    key=key,
                )
        elif chart_name == "unique_classes":
            unique_counts = unique_class_counts(data)
            if not unique_counts.empty:
                st.plotly_chart(
                    px.bar(unique_counts, x="class_name", y="unique_tracks", title="Unique Tracks by Class"),
                    width="stretch",
                    key=key,
                )
        else:
            counts = class_counts(data)
            if not counts.empty:
                st.plotly_chart(
                    px.bar(counts, x="class_name", y="count", title="Objects by Class"),
                    width="stretch",
                    key=key,
                )

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total detections", f"{kpis['total']:,}")
    c2.metric("Unique tracks", f"{kpis['unique_tracks']:,}")
    c3.metric("Top object", kpis["top_class"])
    c4.metric("Avg confidence", f"{kpis['avg_confidence']:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Videos", f"{kpis['videos']:,}")
    c6.metric("ROI zones", f"{kpis['roi_zones']:,}")
    c7.metric("Recognized models", f"{kpis['recognized_models']:,}", f"{kpis['recognition_rate']:.0%}")
    c8.metric("Last detection", _format_dt(kpis["last_detection"]))

    left, right = st.columns([1, 1])
    with left:
        counts = class_counts(df)
        if counts.empty:
            st.info("No detections yet. Run `python code/tracking.py` first.")
        else:
            st.plotly_chart(
                px.bar(counts, x="class_name", y="count", title="Objects by Class"),
                width="stretch",
                key="overview_objects_by_class",
            )
    with right:
        unique_counts = unique_class_counts(df)
        if not unique_counts.empty:
            st.plotly_chart(
                px.bar(unique_counts, x="class_name", y="unique_tracks", title="Unique Tracks by Class"),
                width="stretch",
                key="overview_unique_tracks_by_class",
            )

    trend_col, confidence_col = st.columns([1, 1])
    with trend_col:
        hourly = hourly_counts(df)
        if not hourly.empty:
            st.plotly_chart(
                px.line(hourly, x="hour", y="count", color="class_name", markers=True, title="Vehicle Detection Trend"),
                width="stretch",
                key="overview_detection_trend",
            )
    with confidence_col:
        conf = confidence_by_class(df)
        if not conf.empty:
            st.plotly_chart(
                px.bar(conf, x="class_name", y="avg_confidence", title="Average Confidence by Class"),
                width="stretch",
                key="overview_confidence_by_class",
            )

    if st.toggle("Show advanced analytics", value=False):
        roi_col, video_col = st.columns([1, 1])
        with roi_col:
            rois = roi_counts(df)
            if not rois.empty:
                st.plotly_chart(
                    px.bar(rois.head(15), x="roi_name", y=["count", "unique_tracks"], barmode="group", title="ROI Activity"),
                    width="stretch",
                    key="overview_roi_activity",
                )
        with video_col:
            videos = video_counts(df)
            if not videos.empty:
                video_view = videos.head(10).copy()
                video_view["source_label"] = [f"Video {i + 1}" for i in range(len(video_view))]
                st.plotly_chart(
                    px.bar(video_view, x="source_label", y="count", title="Detections by Video Source"),
                    width="stretch",
                    key="overview_video_sources",
                )

        model_counts = vehicle_model_counts(df)
        colors = vehicle_color_counts(df)
        body_types = vehicle_body_type_counts(df)
        model_col, color_col, body_col = st.columns([1, 1, 1])
        with model_col:
            if not model_counts.empty:
                st.plotly_chart(
                    px.bar(model_counts.head(15), x="vehicle_label", y="count", title="Recognized Vehicle Models"),
                    width="stretch",
                    key="overview_vehicle_models",
                )
        with color_col:
            if not colors.empty:
                st.plotly_chart(
                    px.pie(colors.head(10), names="vehicle_color", values="count", title="Vehicle Colors"),
                    width="stretch",
                    key="overview_vehicle_colors",
                )
        with body_col:
            if not body_types.empty:
                st.plotly_chart(
                    px.bar(body_types.head(10), x="vehicle_body_type", y="count", title="Body Types"),
                    width="stretch",
                    key="overview_body_types",
                )

with tab_assistant:
    st.subheader("Ask the Collected Data")
    use_openai = st.toggle("Use OpenAI when API key exists", value=True)

    if "chat_messages" not in st.session_state:
        saved_history = fetch_chat_history(db_path, limit=8)
        st.session_state.chat_messages = []
        for item in saved_history:
            st.session_state.chat_messages.append({"role": "user", "content": item["question"]})
            st.session_state.chat_messages.append({"role": "assistant", "content": item["answer"]})
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "last_chart_context" not in st.session_state:
        st.session_state.last_chart_context = st.session_state.last_question

    memory_col, clear_col = st.columns([3, 1])
    with memory_col:
        st.caption("Short memory: oxirgi chatlar SQLite `chat_queries` jadvalida saqlanadi va follow-up savollarda ishlatiladi.")
    with clear_col:
        if st.button("Clear memory", key="clear_chat_memory"):
            clear_chat_history(db_path)
            logger.info("dashboard_memory_cleared db_path=%s", db_path)
            st.session_state.chat_messages = []
            st.session_state.last_question = ""
            st.session_state.last_chart_context = ""
            st.rerun()

    examples = [
        "Unique car va truck sonini solishtir",
        "Qaysi ROI eng faol?",
        "Ranglar bo'yicha statistika ber",
        "Confidence past detectionlar bormi?",
    ]
    example_cols = st.columns(len(examples))
    for index, example in enumerate(examples):
        if example_cols[index].button(example, key=f"example_question_{index}"):
            st.session_state.pending_question = example

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Bazadagi tracking datadan savol so'rang")
    question = st.session_state.pop("pending_question", None) or prompt
    if question:
        answer = answer_question(question, db_path, use_openai=use_openai)
        st.session_state.chat_messages.append({"role": "user", "content": question})
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.session_state.last_question = question
        previous_context = st.session_state.last_chart_context or ""
        st.session_state.last_chart_context = f"{previous_context}\n{question}".strip()
        st.rerun()

    if st.session_state.last_question:
        st.divider()
        st.caption("Question-based charts")
        render_recommended_charts(st.session_state.last_chart_context or st.session_state.last_question, df)

with tab_data:
    st.subheader("Detections Table")
    if df.empty:
        st.info("Database is empty.")
    else:
        filters = st.multiselect("Class filter", sorted(df["class_name"].unique()))
        view = df[df["class_name"].isin(filters)] if filters else df
        st.dataframe(view, width="stretch", height=480)
        st.download_button("Download CSV", view.to_csv(index=False), "detections.csv", "text/csv")
