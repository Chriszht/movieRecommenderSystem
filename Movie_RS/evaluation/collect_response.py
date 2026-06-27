"""
Interactive questionnaire collector.

Prompts one participant's answers for a single condition (A or B) and
appends them as a new row to `evaluation/results.csv`.

Usage
-----
    python evaluation/collect_response.py

Type the Likert answers as integers 1-5; leave a comment blank by
pressing Enter.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

CSV_PATH = Path(__file__).with_name("results.csv")
HEADER = [
    "participant_id", "order", "condition",
    "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8",
    "comment_like", "comment_improve",
]
QUESTIONS = [
    ("Q1", "Recommendation match"),
    ("Q2", "Diversity"),
    ("Q3", "Novelty"),
    ("Q4", "Transparency"),
    ("Q5", "Control"),
    ("Q6", "Ease-of-use"),
    ("Q7", "Satisfaction"),
    ("Q8", "Trust"),
]


def prompt_choice(label: str, valid: list[str]) -> str:
    while True:
        v = input(f"{label} [{'/'.join(valid)}]: ").strip().upper()
        if v in valid:
            return v
        print(f"  please answer one of {valid}")


def prompt_int(label: str, lo: int, hi: int) -> int:
    while True:
        raw = input(f"{label} ({lo}-{hi}): ").strip()
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  please type an integer in [{lo}, {hi}]")


def ensure_header() -> None:
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        return
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(HEADER)


def main() -> None:
    print("=== A/B questionnaire collector ===")
    pid = input("Participant ID (e.g. P01): ").strip() or "P??"
    order = prompt_choice("Order they used", ["AB", "BA"])
    cond = prompt_choice("Condition they just used", ["A", "B"])

    scores = {}
    for code, label in QUESTIONS:
        scores[code] = prompt_int(f"  {code}. {label}", 1, 5)

    like = input("Q9  What did they like most? (Enter to skip)\n  > ").strip()
    improve = input("Q10 What would they change?   (Enter to skip)\n  > ").strip()

    ensure_header()
    row = [pid, order, cond, *[scores[c] for c, _ in QUESTIONS], like, improve]
    with CSV_PATH.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)

    n_rows = sum(1 for _ in CSV_PATH.open(encoding="utf-8")) - 1
    print(f"\nSaved. {CSV_PATH} now has {n_rows} response row(s).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted, nothing saved.")
        sys.exit(1)
