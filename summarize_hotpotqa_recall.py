"""Summarize accuracy and exact Top-K selection recall for sparse methods."""

import argparse
from collections import Counter
import csv
import json
import re
import string


def normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def qa_f1(prediction, answer):
    prediction = normalize_answer(prediction).split()
    answer = normalize_answer(answer).split()
    common = Counter(prediction) & Counter(answer)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction)
    recall = same / len(answer)
    return 2 * precision * recall / (precision + recall)


def read_method(path, shadow=False):
    rows = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            prediction = row["prediction"] if shadow else row["pred"]
            f1 = max(qa_f1(prediction, answer) for answer in row["answers"])
            rows.append((row, f1))
    return rows


def summarize(name, rows):
    hits = sum(row["recall_hits"] for row, _ in rows)
    total = sum(row["recall_total"] for row, _ in rows)
    selected = sum(row["selected_tokens_total"] for row, _ in rows)
    events = sum(row["recall_events"] for row, _ in rows)
    return {
        "method": name,
        "samples": len(rows),
        "mean_f1": 100 * sum(f1 for _, f1 in rows) / len(rows),
        "selection_recall_percent": 100 * hits / total,
        "mean_selected_tokens": selected / events,
        "recall_events": events,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quest", required=True)
    parser.add_argument("--clusterkv", required=True)
    parser.add_argument("--shadowkv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    groups = {
        "Quest": read_method(args.quest),
        "ClusterKV": read_method(args.clusterkv),
        "ShadowKV": read_method(args.shadowkv, shadow=True),
    }
    reference_ids = [row["id"] for row, _ in groups["Quest"]]
    for method, rows in groups.items():
        if [row["id"] for row, _ in rows] != reference_ids:
            raise ValueError(f"{method} did not run the same examples in the same order")

    summary = [summarize(method, rows) for method, rows in groups.items()]
    with open(args.output, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(
            f"{row['method']}: n={row['samples']} F1={row['mean_f1']:.2f} "
            f"Recall@128={row['selection_recall_percent']:.2f}% "
            f"selected={row['mean_selected_tokens']:.2f}"
        )


if __name__ == "__main__":
    main()
