"""Score transcriptions against target sentences and compare humans with models.

Usage: python scoring.py corpus.json model_results.json human_results.jsonl
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

PASS_THRESHOLD = 0.5


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def word_accuracy(predicted: str, expected: str) -> float:
    """1 - WER (word-level edit distance over expected length), floored at 0."""
    hyp, ref = normalize(predicted).split(), normalize(expected).split()
    row = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, row[0] = row[:], i
        for j in range(1, len(hyp) + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
    return max(0.0, 1 - row[-1] / len(ref))


def overlap_ratio(predicted: str, expected: str) -> float:
    """Fraction of expected words that appear somewhere in the predicted text."""
    predicted_words = set(normalize(predicted).split())
    expected_words = set(normalize(expected).split())
    if not expected_words:
        return 0.0
    return len(expected_words & predicted_words) / len(expected_words)


def paired_gap(humans: list, model: dict) -> float:
    """Mean of (human score - model score on the same clip) over all human answers."""
    return mean(h["score"] - model[h["clip"]] for h in humans)


def auc(human_scores: list, model_scores: list) -> float:
    """Probability a random human answer outscores a random model answer, ties counted as half."""
    wins = sum((h > m) + 0.5 * (h == m) for h in human_scores for m in model_scores)
    return wins / (len(human_scores) * len(model_scores))


def bootstrap_gap_ci(humans: list, model: dict, reps: int = 10000, seed: int = 0) -> tuple:
    """95% CI for paired_gap, resampling whole sessions since answers within a session are correlated."""
    rng = random.Random(seed)
    by_session = defaultdict(list)
    for h in humans:
        by_session[h["session_id"]].append(h)
    sessions = list(by_session.values())
    gaps = sorted(
        paired_gap([h for s in rng.choices(sessions, k=len(sessions)) for h in s], model) for _ in range(reps)
    )
    return gaps[int(0.025 * reps)], gaps[int(0.975 * reps)]


def main(corpus_path: Path, results_path: Path, submissions_path: Path) -> None:
    expected = {c["id"]: c["target"] for c in json.loads(corpus_path.read_text())["clips"]}
    results = json.loads(results_path.read_text())["models"]
    submissions = [json.loads(line) for line in submissions_path.read_text().splitlines() if line.strip()]

    humans = [
        {**s, "score": word_accuracy(s["answer"], expected[s["clip"]]), "overlap": overlap_ratio(s["answer"], expected[s["clip"]])}
        for s in submissions
    ]
    human_scores = [h["score"] for h in humans]

    def summary(scores, overlaps):
        return (f"pass {mean(s >= PASS_THRESHOLD for s in scores):.1%}  overlap {mean(overlaps):.2f}  "
                f"1-WER {mean(scores):.2f}  perfect {mean(s == 1 for s in scores):.0%}")

    print(f"{'human':<20} {summary(human_scores, [h['overlap'] for h in humans])}  (n={len(humans)})")
    for name, m in results.items():
        model = {clip: word_accuracy(r["predicted"], expected[clip]) for clip, r in m["per_clip"].items()}
        overlaps = [overlap_ratio(r["predicted"], expected[clip]) for clip, r in m["per_clip"].items()]
        lo, hi = bootstrap_gap_ci(humans, model)
        print(f"{name:<20} {summary(list(model.values()), overlaps)}  "
              f"gap {paired_gap(humans, model):+.2f} [{lo:+.2f}, {hi:+.2f}]  AUC {auc(human_scores, list(model.values())):.2f}")


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:4]))
