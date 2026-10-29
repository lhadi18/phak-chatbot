from __future__ import annotations

from phak_study.retrieval import retrieve


# Replace/add entries after inspecting the exact PHAK edition you download.
# Expected section text is matched case-insensitively against returned metadata.
CASES = [
    ("Why does a wing produce lift?", 4, "lift"),
    ("How does angle of attack affect lift and stall?", 5, "angle of attack"),
    ("What does the rudder control?", 6, "rudder"),
    ("What is the purpose of the pitot-static system?", 7, "pitot"),
    ("How does stable air affect clouds and visibility?", 12, "stable air"),
]


def main() -> None:
    hits = 0
    for question, chapter, expected in CASES:
        results = retrieve(question, chapter=chapter)
        hit = any(expected.casefold() in str(r.metadata.get("section", "")).casefold() for r in results)
        hits += int(hit)
        print(f"{'HIT' if hit else 'MISS'} | expected section contains {expected!r} | {question}")
        if not hit:
            for item in results:
                print(f"  got: Chapter {item.metadata['chapter_number']} / {item.metadata['section']} / PDF p. {item.metadata['page_start']}")
    print(f"Hit rate at configured k: {hits}/{len(CASES)} = {hits / len(CASES):.0%}")


if __name__ == "__main__":
    main()
