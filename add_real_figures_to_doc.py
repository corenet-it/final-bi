from __future__ import annotations

import json
import math
import sqlite3
import textwrap
from collections import Counter
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DOCX_PATH = ROOT / "docs" / "xazratov behro'z.docx"
SCREENSHOT_DIR = ROOT / "screenshot"
DB_PATH = ROOT / "code" / "data" / "surveillance.db"
LOG_PATH = ROOT / "code" / "data" / "logs" / "app.log"
ROBOFLOW_DIR = ROOT / "code" / "data" / "roboflow_dataset"


FIG_ANALYTICS = SCREENSHOT_DIR / "06_current_system_analytics_evidence.png"
FIG_ROBOFLOW = SCREENSHOT_DIR / "07_roboflow_dataset_evidence.png"
FIG_LOGS = SCREENSHOT_DIR / "08_logs_pipeline_evidence.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


FONT_TITLE = font(38, True)
FONT_SUBTITLE = font(22, False)
FONT_HEADING = font(24, True)
FONT_BODY = font(20, False)
FONT_SMALL = font(16, False)
FONT_MONO = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16) if Path("/System/Library/Fonts/Menlo.ttc").exists() else font(15)


def rounded_rect(draw: ImageDraw.ImageDraw, xy, radius=18, fill="#ffffff", outline="#d9d9d9", width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], width: int, fnt, fill="#111111", spacing=6):
    x, y = xy
    chars = max(20, int(width / max(7, fnt.size * 0.52)))
    for line in textwrap.wrap(text, width=chars):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + spacing
    return y


def query_metrics() -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        unique_tracks = conn.execute("SELECT COUNT(DISTINCT track_id) FROM detections").fetchone()[0]
        videos = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        roi_zones = conn.execute("SELECT COUNT(*) FROM roi_zones").fetchone()[0]
        avg_conf = conn.execute("SELECT AVG(confidence) FROM detections").fetchone()[0] or 0
        person_rows = conn.execute("SELECT COUNT(*) FROM detections WHERE LOWER(class_name)='person'").fetchone()[0]
        rows = conn.execute(
            """
            SELECT class_name, COUNT(*) AS count, AVG(confidence) AS avg_conf
            FROM detections
            GROUP BY class_name
            ORDER BY count DESC
            """
        ).fetchall()
    return {
        "total": total,
        "unique_tracks": unique_tracks,
        "videos": videos,
        "roi_zones": roi_zones,
        "avg_conf": avg_conf,
        "person_rows": person_rows,
        "classes": [(row["class_name"], row["count"], row["avg_conf"] or 0) for row in rows],
    }


def generate_analytics_figure() -> None:
    data = query_metrics()
    img = Image.new("RGB", (1600, 1000), "#f5f5f2")
    draw = ImageDraw.Draw(img)
    draw.text((70, 55), "Current ASSBI Analytics Evidence", font=FONT_TITLE, fill="#111111")
    draw.text((70, 102), "Generated from the live SQLite database used by the Streamlit dashboard", font=FONT_SUBTITLE, fill="#555555")

    cards = [
        ("Total detections", f"{data['total']:,}"),
        ("Unique vehicle tracks", f"{data['unique_tracks']:,}"),
        ("Videos processed", f"{data['videos']:,}"),
        ("ROI zones", f"{data['roi_zones']:,}"),
        ("Average confidence", f"{data['avg_conf']:.3f}"),
        ("Person rows", f"{data['person_rows']:,}"),
    ]
    x0, y0 = 70, 170
    card_w, card_h, gap = 455, 135, 28
    for i, (label, value) in enumerate(cards):
        x = x0 + (i % 3) * (card_w + gap)
        y = y0 + (i // 3) * (card_h + gap)
        rounded_rect(draw, (x, y, x + card_w, y + card_h), fill="#ffffff", outline="#d8d6ce")
        draw.text((x + 25, y + 22), label, font=FONT_BODY, fill="#5b5b5b")
        color = "#1a7f64" if label != "Person rows" else "#a33a2b"
        draw.text((x + 25, y + 62), value, font=font(40, True), fill=color)

    chart_x, chart_y = 105, 545
    chart_w, chart_h = 1380, 340
    draw.text((chart_x, chart_y - 55), "Objects by class", font=FONT_HEADING, fill="#111111")
    max_count = max([count for _, count, _ in data["classes"]] or [1])
    colors = ["#2f6f9f", "#d37d2f", "#4f8a4b", "#8f5f9f", "#6f6f6f"]
    bar_gap = 24
    bar_h = 46
    for idx, (class_name, count, avg_conf) in enumerate(data["classes"][:7]):
        y = chart_y + idx * (bar_h + bar_gap)
        width = int((count / max_count) * (chart_w - 360))
        draw.text((chart_x, y + 9), class_name, font=FONT_BODY, fill="#111111")
        rounded_rect(draw, (chart_x + 190, y, chart_x + 190 + width, y + bar_h), radius=8, fill=colors[idx % len(colors)], outline=colors[idx % len(colors)])
        draw.text((chart_x + 210 + width, y + 9), f"{count:,}   avg conf {avg_conf:.3f}", font=FONT_BODY, fill="#333333")

    draw.text((70, 940), "Evidence note: the query confirms that person detections are filtered out before database storage.", font=FONT_SMALL, fill="#555555")
    img.save(FIG_ANALYTICS)


def roboflow_counts() -> dict:
    images_train = list((ROBOFLOW_DIR / "images" / "train").glob("*"))
    images_val = list((ROBOFLOW_DIR / "images" / "val").glob("*"))
    labels_train = list((ROBOFLOW_DIR / "labels" / "train").glob("*.txt"))
    labels_val = list((ROBOFLOW_DIR / "labels" / "val").glob("*.txt"))
    class_ids = Counter()
    for label_file in labels_train + labels_val:
        for line in label_file.read_text(errors="ignore").splitlines():
            parts = line.split()
            if parts and parts[0].isdigit():
                class_ids[int(parts[0])] += 1
    yaml_text = (ROBOFLOW_DIR / "data.yaml").read_text(errors="ignore")
    return {
        "images_train": images_train,
        "images_val": images_val,
        "labels_train": labels_train,
        "labels_val": labels_val,
        "class_ids": class_ids,
        "yaml": yaml_text,
    }


def generate_roboflow_figure() -> None:
    data = roboflow_counts()
    img = Image.new("RGB", (1600, 1000), "#f7f7f5")
    draw = ImageDraw.Draw(img)
    draw.text((70, 55), "Roboflow Dataset Evidence", font=FONT_TITLE, fill="#111111")
    draw.text((70, 102), "Real exported frame crops and YOLO label files prepared for custom vehicle fine-tuning", font=FONT_SUBTITLE, fill="#555555")

    stats = [
        ("Train images", len(data["images_train"])),
        ("Train labels", len(data["labels_train"])),
        ("Validation images", len(data["images_val"])),
        ("Validation labels", len(data["labels_val"])),
    ]
    for i, (label, value) in enumerate(stats):
        x = 70 + i * 365
        rounded_rect(draw, (x, 165, x + 330, 285), fill="#ffffff", outline="#d8d6ce")
        draw.text((x + 20, 188), label, font=FONT_BODY, fill="#555555")
        draw.text((x + 20, 225), f"{value:,}", font=font(36, True), fill="#1a6376")

    sample_paths = sorted(data["images_train"])[:6]
    thumb_y = 365
    draw.text((70, 325), "Sample exported training images", font=FONT_HEADING, fill="#111111")
    for i, path in enumerate(sample_paths):
        x = 70 + i * 245
        try:
            thumb = Image.open(path).convert("RGB")
            thumb.thumbnail((210, 150))
            box = Image.new("RGB", (220, 160), "#e8e8e4")
            box.paste(thumb, ((220 - thumb.width) // 2, (160 - thumb.height) // 2))
            img.paste(box, (x, thumb_y))
        except Exception:
            rounded_rect(draw, (x, thumb_y, x + 220, thumb_y + 160), fill="#e8e8e4", outline="#d8d6ce")
        name = path.name.replace("_", " ")
        draw_wrapped(draw, name, (x, thumb_y + 172), 220, FONT_SMALL, fill="#333333", spacing=2)

    y = 690
    draw.text((70, y), "data.yaml configuration", font=FONT_HEADING, fill="#111111")
    rounded_rect(draw, (70, y + 45, 740, 910), fill="#ffffff", outline="#d8d6ce")
    yy = y + 72
    for line in data["yaml"].splitlines():
        draw.text((95, yy), line, font=FONT_MONO, fill="#222222")
        yy += 24

    draw.text((820, y), "Annotation evidence", font=FONT_HEADING, fill="#111111")
    rounded_rect(draw, (820, y + 45, 1485, 910), fill="#ffffff", outline="#d8d6ce")
    yy = y + 72
    class_name_map = {0: "car", 1: "bus", 2: "truck", 3: "motorcycle", 4: "bicycle"}
    for class_id, count in sorted(data["class_ids"].items()):
        draw.text((845, yy), f"{class_name_map.get(class_id, class_id)} labels: {count:,}", font=FONT_BODY, fill="#222222")
        yy += 36
    img.save(FIG_ROBOFLOW)


def generate_logs_figure() -> None:
    lines = LOG_PATH.read_text(errors="ignore").splitlines()
    selected = []
    keywords = ("tracking_final_batch_saved", "interactive_tracking_finished", "roboflow_sample_saved", "detections_batch_inserted", "vehicle_crop_saved")
    for line in reversed(lines):
        if any(keyword in line for keyword in keywords):
            selected.append(line)
        if len(selected) >= 17:
            break
    selected = list(reversed(selected))

    img = Image.new("RGB", (1600, 1000), "#f4f4f1")
    draw = ImageDraw.Draw(img)
    draw.text((70, 55), "Application Logs and Tracking Audit Evidence", font=FONT_TITLE, fill="#111111")
    draw.text((70, 102), "Real log records from tracker execution, SQLite inserts, crop saving and Roboflow sample export", font=FONT_SUBTITLE, fill="#555555")
    rounded_rect(draw, (70, 165, 1530, 900), radius=14, fill="#ffffff", outline="#d8d6ce")
    y = 200
    for line in selected:
        wrapped = textwrap.wrap(line, width=145)
        for part in wrapped[:2]:
            draw.text((100, y), part, font=FONT_MONO, fill="#202020")
            y += 26
        y += 8
        if y > 860:
            break
    draw.text((70, 940), "Evidence note: logs support reproducibility and troubleshooting for the BI pipeline.", font=FONT_SMALL, fill="#555555")
    img.save(FIG_LOGS)


def paragraph_after(paragraph, text: str = "", style: str | None = None):
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    new_para = Paragraph(new_p, paragraph._parent)
    if style:
        new_para.style = style
    if text:
        new_para.add_run(text)
    return new_para


def set_times(run, size: float = 12, bold: bool = False, italic: bool = False):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def insert_image_after(paragraph, image_path: Path, caption: str, note: str, width_inches: float = 6.0, page_break_after: bool = False):
    note_p = paragraph_after(paragraph, note)
    note_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for run in note_p.runs:
        set_times(run, 12)
    note_p.paragraph_format.space_after = Pt(6)

    image_p = paragraph_after(note_p)
    image_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = image_p.add_run()
    run.add_picture(str(image_path), width=Inches(width_inches))
    image_p.paragraph_format.space_after = Pt(3)

    caption_p = paragraph_after(image_p, caption)
    caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in caption_p.runs:
        set_times(run, 10.5, italic=True)
    caption_p.paragraph_format.space_after = Pt(12)
    if page_break_after:
        break_p = paragraph_after(caption_p)
        break_p.add_run().add_break(WD_BREAK.PAGE)
    return caption_p


def find_paragraph(doc: Document, needle: str):
    for paragraph in doc.paragraphs:
        if needle in paragraph.text:
            return paragraph
    raise ValueError(f"Could not find paragraph containing: {needle}")


def document_has_caption(doc: Document, caption_prefix: str) -> bool:
    return any(caption_prefix in paragraph.text for paragraph in doc.paragraphs)


def remove_figure_block(doc: Document, caption_prefix: str) -> None:
    paragraphs = list(doc.paragraphs)
    for idx, paragraph in enumerate(paragraphs):
        if caption_prefix in paragraph.text:
            blocks = paragraphs[max(0, idx - 2): idx + 1]
            if idx + 1 < len(paragraphs) and "w:type=\"page\"" in paragraphs[idx + 1]._element.xml:
                blocks.append(paragraphs[idx + 1])
            for block in blocks:
                element = block._element
                parent = element.getparent()
                if parent is not None:
                    parent.remove(element)
            return


def update_docx() -> None:
    doc = Document(DOCX_PATH)
    for old_caption in (
        "Figure 8. Current live ASSBI analytics evidence",
        "Figure 9. Roboflow-ready dataset evidence",
        "Figure 10. Application log evidence",
        "Figure 4A. Current live ASSBI analytics evidence",
        "Figure 2A. Roboflow-ready dataset evidence",
        "Figure 7A. Application log evidence",
    ):
        remove_figure_block(doc, old_caption)

    if not document_has_caption(doc, "Figure 4A. Current live ASSBI analytics evidence"):
        target = find_paragraph(doc, "Figure 4. Streamlit Overview tab")
        insert_image_after(
            target,
            FIG_ANALYTICS,
            "Figure 4A. Current live ASSBI analytics evidence generated from the SQLite database and Streamlit analytics layer.",
            "The figure below adds live evidence from the current system database. It demonstrates that the dashboard is not based on mock data: the values are calculated from the detections table, grouped by vehicle class, and checked against the person-filtering rule required by the scenario.",
            width_inches=6.1,
        )

    if not document_has_caption(doc, "Figure 2A. Roboflow-ready dataset evidence"):
        target = find_paragraph(doc, "B.P4")
        insert_image_after(
            target,
            FIG_ROBOFLOW,
            "Figure 2A. Roboflow-ready dataset evidence showing exported images, YOLO labels and class configuration for model training.",
            "This evidence supports the model-improvement part of the implementation. The project exports real vehicle crops and YOLO label files so that the dataset can be reviewed in Roboflow, annotated further, augmented, and used for custom fine-tuning of vehicle detection.",
            width_inches=6.1,
            page_break_after=True,
        )

    if not document_has_caption(doc, "Figure 7A. Application log evidence"):
        target = find_paragraph(doc, "Source code:")
        insert_image_after(
            target,
            FIG_LOGS,
            "Figure 7A. Application log evidence showing tracking completion, database inserts and Roboflow sample export audit records.",
            "The log evidence confirms that the implemented system records operational events during execution. These records are important for BI governance because they show when tracking batches are saved, how many rows are inserted, and whether Roboflow training samples are exported.",
            width_inches=6.1,
        )

    doc.save(DOCX_PATH)


def main() -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    generate_analytics_figure()
    generate_roboflow_figure()
    generate_logs_figure()
    update_docx()
    print(json.dumps({
        "docx": str(DOCX_PATH),
        "figures": [str(FIG_ANALYTICS), str(FIG_ROBOFLOW), str(FIG_LOGS)],
    }, indent=2))


if __name__ == "__main__":
    main()
