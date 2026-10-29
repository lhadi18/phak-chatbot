from __future__ import annotations

from phak_study.config import load_settings
from phak_study.providers import create_provider
from phak_study.quiz import grade_answer


CASES = [
    {
        "name": "good answer",
        "question": "Why does increasing angle of attack increase lift before a stall?",
        "points": ["increased angle of attack increases pressure difference", "lift increases up to critical angle"],
        "source": "As angle of attack increases, the pressure difference across the wing increases and lift increases until the critical angle of attack is exceeded.",
        "answer": "A higher angle of attack creates a greater pressure difference, so lift rises until the critical angle is passed.",
        "expected": (3, 3),
    },
    {
        "name": "partial answer",
        "question": "Why does increasing angle of attack increase lift before a stall?",
        "points": ["increased angle of attack increases pressure difference", "lift increases up to critical angle"],
        "source": "As angle of attack increases, the pressure difference across the wing increases and lift increases until the critical angle of attack is exceeded.",
        "answer": "Lift increases when angle of attack increases.",
        "expected": (1, 2),
    },
    {
        "name": "wrong answer",
        "question": "Why does increasing angle of attack increase lift before a stall?",
        "points": ["increased angle of attack increases pressure difference", "lift increases up to critical angle"],
        "source": "As angle of attack increases, the pressure difference across the wing increases and lift increases until the critical angle of attack is exceeded.",
        "answer": "Angle of attack has no effect on lift; only aircraft speed matters.",
        "expected": (0, 0),
    },
]


def main() -> None:
    settings = load_settings()
    provider = create_provider(settings)
    passed = 0
    for case in CASES:
        result = grade_answer(case["question"], case["points"], case["answer"], case["source"], provider)
        low, high = case["expected"]
        ok = low <= result["score"] <= high
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} {case['name']}: score {result['score']}/3 (expected {low}-{high})")
        print(f"  Covered key points: {result['covered']}; right: {result['right']}; missed/wrong: {result['missed']}")
    print(f"Grading sanity check: {passed}/{len(CASES)} passed with provider={settings.llm_provider}, model={settings.llm_model}")
    if passed != len(CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
