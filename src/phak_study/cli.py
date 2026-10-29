from __future__ import annotations

import argparse

from .ask import ask, print_source_list
from .config import load_settings
from .ingest import ingest
from .progress import stats_report
from .quiz import prepare_questions, run_quiz
from .retrieval import print_results, retrieve


def main() -> None:
    parser = argparse.ArgumentParser(prog="phak-study")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Extract, chunk, embed locally, and persist configured PHAK chapters")
    search = sub.add_parser("retrieve", help="Inspect the best matching source chunks")
    search.add_argument("question")
    search.add_argument("--chapter", type=int)
    search.add_argument("--top-k", type=int)

    ask_cmd = sub.add_parser("ask", help="Ask a question grounded in retrieved PHAK excerpts")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--chapter", type=int)
    ask_cmd.add_argument("--top-k", type=int)
    ask_cmd.add_argument("--hide-excerpts", action="store_true")

    prepare = sub.add_parser("quiz-prepare", help="Pre-generate and cache quiz questions")
    prepare.add_argument("--chapter", type=int)
    prepare.add_argument("--topic")
    prepare.add_argument("--n", type=int, default=30)

    quiz = sub.add_parser("quiz", help="Answer and get scored on cached or newly generated questions")
    quiz.add_argument("--chapter", type=int)
    quiz.add_argument("--topic")
    quiz.add_argument("--n", type=int, default=10)
    quiz.add_argument("--weak", action="store_true", help="Prioritise weak areas and review-due questions")

    sub.add_parser("stats", help="Show quiz performance and due reviews")
    args = parser.parse_args()
    settings = load_settings()

    try:
        if args.command == "ingest":
            ingest(settings)
        elif args.command == "retrieve":
            print_results(retrieve(args.question, chapter=args.chapter, top_k=args.top_k, settings=settings))
        elif args.command == "ask":
            streamed = {"started": False}

            def show_token(fragment: str) -> None:
                if not streamed["started"]:
                    print("Answer: ", end="", flush=True)
                    streamed["started"] = True
                print(fragment, end="", flush=True)

            result = ask(
                args.question,
                chapter=args.chapter,
                top_k=args.top_k,
                settings=settings,
                on_token=show_token,
            )
            if result.sources:
                if not streamed["started"]:
                    print(f"Answer: {result.answer}", end="")
                print("\n")
                if result.citation_warning:
                    print(result.citation_warning)
            else:
                print(f"Answer: {result.answer}\n")
            for index, source in enumerate(result.sources, 1):
                meta = source.metadata
                pages = str(meta["page_start"]) if meta["page_start"] == meta["page_end"] else f"{meta['page_start']}-{meta['page_end']}"
                print(f"[S{index}] Chapter {meta['chapter_number']} — {meta['chapter_title']} / {meta['section']} / PDF page(s) {pages}")
            if not args.hide_excerpts:
                print_source_list(result.sources)
        elif args.command == "quiz-prepare":
            if args.chapter is None and not args.topic:
                parser.error("quiz-prepare requires --chapter or --topic")
            if args.n < 1:
                parser.error("--n must be at least 1")
            count = prepare_questions(chapter=args.chapter, topic=args.topic, n=args.n, settings=settings)
            print(f"Cached {count} new question(s). Repeated questions already in the cache were skipped.")
        elif args.command == "quiz":
            if args.chapter is None and not args.topic and not args.weak:
                parser.error("quiz requires --chapter, --topic, or --weak")
            if args.n < 1:
                parser.error("--n must be at least 1")
            run_quiz(chapter=args.chapter, topic=args.topic, n=args.n, weak=args.weak, settings=settings)
        elif args.command == "stats":
            print(stats_report(settings))
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
