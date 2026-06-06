from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

from config import DEFAULT_DB_PATH
from database import connect
from logging_utils import get_logger


logger = get_logger("vector_store")

EMBED_DIM = 384
TOKEN_RE = re.compile(r"[a-zA-Z0-9_']+")

VECTOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    source_name TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    embedding_json TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rag_vectors_source ON rag_vectors(source_type, source_name);
CREATE INDEX IF NOT EXISTS idx_rag_vectors_created ON rag_vectors(created_at);
"""


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def ensure_vector_schema(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(VECTOR_SCHEMA)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if len(token) > 1]


def embed_text(text: str) -> list[float]:
    vector = [0.0] * EMBED_DIM
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % EMBED_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 6) for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))


def content_hash(source_type: str, source_name: str, content: str) -> str:
    raw = f"{source_type}\n{source_name}\n{content}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 180) -> list[str]:
    cleaned = re.sub(r"\n{3,}", "\n\n", text or "").strip()
    if not cleaned:
        return []
    paragraphs = [part.strip() for part in cleaned.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
            current = current[-overlap:] if overlap > 0 else ""
        if len(paragraph) > max_chars:
            start = 0
            while start < len(paragraph):
                end = start + max_chars
                chunks.append(paragraph[start:end].strip())
                start = max(start + 1, end - overlap)
            current = ""
        else:
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def upsert_vector(
    *,
    source_type: str,
    source_name: str,
    content: str,
    metadata: dict | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> bool:
    ensure_vector_schema(db_path)
    trimmed = content.strip()
    if not trimmed:
        return False
    digest = content_hash(source_type, source_name, trimmed)
    with connect(db_path) as conn:
        before = conn.total_changes
        conn.execute(
            """
            INSERT OR IGNORE INTO rag_vectors(
                source_type, source_name, content, content_hash,
                embedding_json, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_type,
                source_name,
                trimmed,
                digest,
                json.dumps(embed_text(trimmed)),
                json.dumps(metadata or {}, ensure_ascii=False),
                utc_now(),
            ),
        )
        return conn.total_changes > before


def index_text(
    *,
    source_type: str,
    source_name: str,
    text: str,
    metadata: dict | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    count = 0
    for index, chunk in enumerate(chunk_text(text), start=1):
        chunk_metadata = dict(metadata or {})
        chunk_metadata["chunk"] = index
        if upsert_vector(
            source_type=source_type,
            source_name=source_name,
            content=chunk,
            metadata=chunk_metadata,
            db_path=db_path,
        ):
            count += 1
    logger.info("rag_text_indexed source_type=%s source_name=%s chunks=%s", source_type, source_name, count)
    return count


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rebuild_project_index(db_path: Path | str = DEFAULT_DB_PATH) -> int:
    root = project_root()
    candidates = [
        root / "README.md",
        root / "code" / "README.md",
        root / "docs" / "README.md",
        root / "docs" / "technology_stack.md",
        root / "docs" / "fine_tuning_and_logs.md",
        root / "docs" / "roboflow_workflow.md",
    ]
    ensure_vector_schema(db_path)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM rag_vectors WHERE source_type = 'project_doc'")

    indexed = 0
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        indexed += index_text(
            source_type="project_doc",
            source_name=str(path.relative_to(root)),
            text=text,
            metadata={"path": str(path)},
            db_path=db_path,
        )
    logger.info("rag_project_index_rebuilt chunks=%s db_path=%s", indexed, db_path)
    return indexed


def index_chat_exchange(question: str, answer: str, db_path: Path | str = DEFAULT_DB_PATH) -> None:
    text = f"User question: {question.strip()}\nAssistant answer: {answer.strip()}"
    upsert_vector(
        source_type="chat_memory",
        source_name="ask_data",
        content=text,
        metadata={"kind": "chat_exchange"},
        db_path=db_path,
    )


def search_vectors(question: str, db_path: Path | str = DEFAULT_DB_PATH, *, top_k: int = 5) -> list[dict]:
    ensure_vector_schema(db_path)
    query_vector = embed_text(question)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, source_type, source_name, content, metadata_json, embedding_json, created_at
            FROM rag_vectors
            ORDER BY id DESC
            LIMIT 3000
            """
        ).fetchall()

    scored: list[dict] = []
    for row in rows:
        try:
            embedding = json.loads(row["embedding_json"])
        except json.JSONDecodeError:
            continue
        score = cosine_similarity(query_vector, embedding)
        if score <= 0:
            continue
        scored.append(
            {
                "id": row["id"],
                "source_type": row["source_type"],
                "source_name": row["source_name"],
                "content": row["content"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "created_at": row["created_at"],
                "score": round(score, 4),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def rag_context_text(question: str, db_path: Path | str = DEFAULT_DB_PATH, *, top_k: int = 5) -> str:
    matches = search_vectors(question, db_path, top_k=top_k)
    if not matches:
        return "No RAG context found."
    lines = []
    for index, match in enumerate(matches, start=1):
        content = match["content"].replace("\n", " ").strip()
        if len(content) > 900:
            content = content[:900] + "..."
        lines.append(
            f"{index}. source={match['source_type']}:{match['source_name']} "
            f"score={match['score']}\n{content}"
        )
    return "\n\n".join(lines)


def vector_stats(db_path: Path | str = DEFAULT_DB_PATH) -> dict:
    ensure_vector_schema(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT source_type, COUNT(*) AS count
            FROM rag_vectors
            GROUP BY source_type
            ORDER BY source_type
            """
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM rag_vectors").fetchone()[0]
    return {"total": int(total), "by_source": {row["source_type"]: int(row["count"]) for row in rows}}


def clear_vectors(db_path: Path | str = DEFAULT_DB_PATH, source_type: str | None = None) -> int:
    ensure_vector_schema(db_path)
    with connect(db_path) as conn:
        before = conn.total_changes
        if source_type:
            conn.execute("DELETE FROM rag_vectors WHERE source_type = ?", (source_type,))
        else:
            conn.execute("DELETE FROM rag_vectors")
        deleted = conn.total_changes - before
    logger.info("rag_vectors_cleared source_type=%s deleted=%s", source_type or "all", deleted)
    return int(deleted)
