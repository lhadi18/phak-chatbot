from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
from typing import Literal

import chromadb
from pydantic import BaseModel, Field

from .config import Settings, load_settings
from .prompts import GRADE_SYSTEM, GRADE_USER, QUESTION_SYSTEM, QUESTION_USER
from .progress import connect_db, get_cached_questions, has_cached_questions, save_attempt
from .providers import LLMProvider, create_provider


class GeneratedQuestion(BaseModel):
    question: str = Field(min_length=10)
    question_type: Literal["recall", "explain_why", "what_if"]
    key_points: list[str] = Field(min_length=2, max_length=4)
    supporting_quote: str = Field(min_length=8, max_length=280)
    model_answer: str = Field(min_length=10)


class QuestionBatch(BaseModel):
    questions: list[GeneratedQuestion] = Field(min_length=1, max_length=1)


class GradeJudgment(BaseModel):
    covered_key_points: list[int]
    what_was_right: str
    what_was_missed_or_wrong: str


def _normalize_quote(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def quote_is_verified(quote: str, source_chunk: str) -> bool:
    return _normalize_quote(quote) in _normalize_quote(source_chunk)


def _scope_chunks(settings: Settings, chapter: int | None, topic: str | None) -> list[dict]:
    collection = chromadb.PersistentClient(path=str(settings.chroma_dir)).get_collection(settings.collection_name)
    where = {"chapter_number": chapter} if chapter is not None else None
    source = collection.get(where=where, include=["documents", "metadatas"])
    rows = []
    for text, metadata in zip(source["documents"], source["metadatas"]):
        if topic and topic.casefold() not in f"{metadata.get('section', '')} {metadata.get('chapter_title', '')} {text}".casefold():
            continue
        rows.append({"text": text, "metadata": metadata})
    return rows


def _make_question(chunk: str, kind: str, provider: LLMProvider, *, quote_retry: bool = False) -> GeneratedQuestion:
    prompt = QUESTION_USER.format(kind=kind, chunk=chunk)
    if quote_retry:
        prompt += "\n\nThe previous supporting quote failed exact source verification. Select a shorter contiguous passage copied exactly from the source, preserving its words and punctuation."
    batch = provider.generate_json(
        prompt,
        QuestionBatch,
        system=QUESTION_SYSTEM,
        retries=1,
    )
    return batch.questions[0]


def prepare_questions(
    *,
    chapter: int | None,
    n: int,
    topic: str | None = None,
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
) -> int:
    settings = settings or load_settings()
    provider = provider or create_provider(settings)
    chunks = _scope_chunks(settings, chapter, topic)
    if not chunks:
        raise ValueError("No indexed chunks match that chapter/topic. Check ingestion and your selection.")
    picks = random.choices(chunks, k=n) if n > len(chunks) else random.sample(chunks, n)
    made = 0
    with connect_db(settings) as db:
        for source in picks:
            metadata = source["metadata"]
            chunk = source["text"]
            question = None
            verified = False
            for attempt in range(2):
                kind = random.choices(["explain_why", "what_if", "recall"], weights=[5, 4, 1])[0]
                candidate = _make_question(chunk, kind, provider, quote_retry=bool(attempt))
                if quote_is_verified(candidate.supporting_quote, chunk):
                    question, verified = candidate, True
                    break
                question = candidate
            assert question is not None
            if not verified:
                print(f"WARNING: supporting quote could not be verified for generated question: {question.question}")
            chapter_num = int(metadata["chapter_number"])
            topic_name = str(metadata.get("chapter_title", settings.chapters.get(chapter_num, f"Chapter {chapter_num}")))
            section = str(metadata.get("section", "Unlabeled section"))
            source_file = str(metadata.get("source_file", "PHAK PDF"))
            key = hashlib.sha256(f"{chapter_num}\0{section}\0{question.question}".encode()).hexdigest()
            db.execute(
                """INSERT OR IGNORE INTO quiz_questions
                (cache_key, chapter, topic, section, page_start, page_end, source_file, question,
                 question_type, key_points_json, supporting_quote, quote_verified, model_answer,
                 source_chunk, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                (key, chapter_num, topic_name, section, int(metadata["page_start"]), int(metadata["page_end"]),
                 source_file, question.question, question.question_type, json.dumps(question.key_points),
                 question.supporting_quote, int(verified), question.model_answer, chunk),
            )
            made += db.execute("SELECT changes()").fetchone()[0]
        db.commit()
    return made


def _get_or_prepare(
    *, settings: Settings, n: int, chapter: int | None, topic: str | None, weak: bool,
    provider: LLMProvider,
) -> list[sqlite3.Row]:
    rows = get_cached_questions(settings=settings, n=n, chapter=chapter, topic=topic, weak=weak)
    if not rows:
        if has_cached_questions(settings=settings, chapter=chapter, topic=topic):
            return []
        prepare_questions(chapter=chapter, n=n, topic=topic, settings=settings, provider=provider)
        rows = get_cached_questions(settings=settings, n=n, chapter=chapter, topic=topic, weak=weak)
    return rows


def grade_answer(question: str, key_points: list[str], answer: str, source_chunk: str, provider: LLMProvider) -> dict:
    prompt = GRADE_USER.format(
        question=question,
        source_chunk=source_chunk,
        key_points="\n".join(f"{index}. {point}" for index, point in enumerate(key_points)),
        answer=answer,
    )
    judgment = provider.generate_json(prompt, GradeJudgment, system=GRADE_SYSTEM, retries=1)
    covered = sorted({index for index in judgment.covered_key_points if 0 <= index < len(key_points)})
    ratio = len(covered) / max(len(key_points), 1)
    score = 0 if not covered else 1 if ratio < 0.5 else 2 if ratio < 0.85 else 3
    return {"score": score, "covered": covered, "right": judgment.what_was_right,
            "missed": judgment.what_was_missed_or_wrong}


def run_quiz(
    *, chapter: int | None, topic: str | None, n: int, weak: bool = False,
    settings: Settings | None = None, provider: LLMProvider | None = None,
) -> None:
    settings = settings or load_settings()
    provider = provider or create_provider(settings)
    questions = _get_or_prepare(settings=settings, n=n, chapter=chapter, topic=topic, weak=weak, provider=provider)
    if not questions:
        print("No cached questions found or generated.")
        return
    for number, row in enumerate(questions, 1):
        points = json.loads(row["key_points_json"])
        pages = str(row["page_start"]) if row["page_start"] == row["page_end"] else f"{row['page_start']}-{row['page_end']}"
        print(f"\nQuestion {number}/{len(questions)} [{row['question_type']}]")
        print(row["question"])
        answer = input("Your answer (or q to stop): ").strip()
        if answer.lower() == "q":
            break
        grade = grade_answer(row["question"], points, answer, row["source_chunk"], provider)
        save_attempt(row, answer, grade["score"], len(grade["covered"]), len(points), settings)
        print(f"\nScore: {grade['score']}/3 ({len(grade['covered'])}/{len(points)} key points covered)")
        print(f"Got right: {grade['right']}")
        print(f"Missed or wrong: {grade['missed']}")
        print(f"Model answer: {row['model_answer']}")
        verified = "verified" if row["quote_verified"] else "UNVERIFIED — check against the source"
        print(f"Reread: Chapter {row['chapter']} — {row['topic']} / {row['section']} / {row['source_file']} PDF page(s) {pages} [{verified}]")
        print(f"Supporting quote: “{row['supporting_quote']}”")
