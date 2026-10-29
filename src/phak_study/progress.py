from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3
from pathlib import Path

from .config import Settings, load_settings


def connect_db(settings: Settings | None = None) -> sqlite3.Connection:
    settings = settings or load_settings()
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.database_path)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS quiz_questions (
            id INTEGER PRIMARY KEY,
            cache_key TEXT NOT NULL UNIQUE,
            chapter INTEGER NOT NULL,
            topic TEXT NOT NULL,
            section TEXT NOT NULL,
            page_start INTEGER NOT NULL,
            page_end INTEGER NOT NULL,
            source_file TEXT NOT NULL,
            question TEXT NOT NULL,
            question_type TEXT NOT NULL,
            key_points_json TEXT NOT NULL,
            supporting_quote TEXT NOT NULL,
            quote_verified INTEGER NOT NULL,
            model_answer TEXT NOT NULL,
            source_chunk TEXT NOT NULL,
            created_at TEXT NOT NULL,
            next_due TEXT
        );
        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id INTEGER PRIMARY KEY,
            question_id INTEGER NOT NULL REFERENCES quiz_questions(id),
            question TEXT NOT NULL,
            chapter INTEGER NOT NULL,
            topic TEXT NOT NULL,
            section TEXT NOT NULL,
            answer TEXT NOT NULL,
            score INTEGER NOT NULL CHECK(score BETWEEN 0 AND 3),
            covered_points INTEGER NOT NULL,
            total_points INTEGER NOT NULL,
            timestamp TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_quiz_attempts_timestamp ON quiz_attempts(timestamp);
        CREATE INDEX IF NOT EXISTS idx_quiz_questions_topic ON quiz_questions(chapter, topic);
        """
    )
    connection.commit()
    return connection


def due_after(score: int) -> str:
    days = {0: 1, 1: 2, 2: 5, 3: 10}[score]
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def save_attempt(question_row: sqlite3.Row, answer: str, score: int, covered: int, total: int, settings: Settings) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with connect_db(settings) as db:
        db.execute(
            """INSERT INTO quiz_attempts
            (question_id, question, chapter, topic, section, answer, score, covered_points, total_points, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (question_row["id"], question_row["question"], question_row["chapter"], question_row["topic"],
             question_row["section"], answer, score, covered, total, now),
        )
        db.execute("UPDATE quiz_questions SET next_due = ? WHERE id = ?", (due_after(score), question_row["id"]))


def get_cached_questions(
    *, settings: Settings, n: int, chapter: int | None = None, topic: str | None = None, weak: bool = False
) -> list[sqlite3.Row]:
    clauses: list[str] = []
    params: list[object] = []
    if chapter is not None:
        clauses.append("q.chapter = ?")
        params.append(chapter)
    if topic:
        clauses.append("(q.topic LIKE ? OR q.section LIKE ?)")
        params.extend([f"%{topic}%", f"%{topic}%"])
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    due = datetime.now(timezone.utc).isoformat()
    review_order = "CASE WHEN q.next_due IS NOT NULL AND q.next_due <= ? THEN 0 WHEN a.id IS NULL THEN 1 ELSE 2 END"
    order = f"{review_order}, COALESCE(AVG(a.score), 4) ASC, RANDOM()" if weak else f"{review_order}, RANDOM()"
    query = f"""SELECT q.*, COUNT(a.id) AS attempts, AVG(a.score) AS average_score,
        MAX(a.timestamp) AS last_attempt
        FROM quiz_questions q LEFT JOIN quiz_attempts a ON a.question_id = q.id
        {where} GROUP BY q.id ORDER BY {order} LIMIT ?"""
    with connect_db(settings) as db:
        rows = list(db.execute(query, [*params, due, n]))
        ready = [r for r in rows if r["attempts"] == 0 or not r["next_due"] or r["next_due"] <= due]
        return ready[:n]


def has_cached_questions(*, settings: Settings, chapter: int | None = None, topic: str | None = None) -> bool:
    clauses: list[str] = []
    params: list[object] = []
    if chapter is not None:
        clauses.append("chapter = ?")
        params.append(chapter)
    if topic:
        clauses.append("(topic LIKE ? OR section LIKE ?)")
        params.extend([f"%{topic}%", f"%{topic}%"])
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with connect_db(settings) as db:
        return db.execute(f"SELECT 1 FROM quiz_questions{where} LIMIT 1", params).fetchone() is not None


def stats_report(settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    with connect_db(settings) as db:
        attempts = db.execute("SELECT COUNT(*) n, AVG(score) avg FROM quiz_attempts").fetchone()
        if not attempts["n"]:
            return "No quiz attempts yet. Run `quiz-prepare` and then `quiz` to start tracking progress."
        out = [f"Attempts: {attempts['n']} | Mean score: {attempts['avg']:.2f}/3"]
        out.append("\nWeakest sections/topics:")
        weak = db.execute(
            """SELECT chapter, topic, section, COUNT(*) attempts, AVG(score) avg_score
               FROM quiz_attempts GROUP BY chapter, topic, section
               ORDER BY avg_score ASC, attempts DESC LIMIT 10"""
        ).fetchall()
        for row in weak:
            out.append(f"  Ch. {row['chapter']} | {row['topic']} / {row['section']} — {row['avg_score']:.2f}/3 over {row['attempts']} attempt(s)")
        out.append("\nAccuracy over time (daily mean score):")
        daily = db.execute(
            "SELECT substr(timestamp, 1, 10) day, AVG(1.0 * covered_points / total_points) accuracy, COUNT(*) n FROM quiz_attempts GROUP BY day ORDER BY day DESC LIMIT 14"
        ).fetchall()
        for row in reversed(daily):
            out.append(f"  {row['day']}: {row['accuracy']:.0%} key-point coverage ({row['n']} attempt(s))")
        out.append("\nRepeatedly missed questions (score < 2 at least twice):")
        missed = db.execute(
            """SELECT question, chapter, section, COUNT(*) n, AVG(score) avg_score
               FROM quiz_attempts WHERE score < 2 GROUP BY question_id
               HAVING COUNT(*) >= 2 ORDER BY n DESC LIMIT 10"""
        ).fetchall()
        out.extend([f"  Ch. {r['chapter']} | {r['section']} — {r['question']} ({r['n']} low scores, mean {r['avg_score']:.1f})" for r in missed] or ["  None yet"])
        out.append("\nReviews due now:")
        due = db.execute(
            """SELECT chapter, section, question FROM quiz_questions
               WHERE next_due IS NOT NULL AND next_due <= ? ORDER BY next_due LIMIT 10""",
            (datetime.now(timezone.utc).isoformat(),),
        ).fetchall()
        out.extend([f"  Ch. {r['chapter']} | {r['section']} — {r['question']}" for r in due] or ["  None"])
        return "\n".join(out)
