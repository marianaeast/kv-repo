"""Summarize the four controlled HotpotQA accuracy runs."""

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


def read_jsonl(path, shadow=False):
    rows = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            prediction = row["prediction"] if shadow else row["pred"]
            score = max(qa_f1(prediction, answer) for answer in row["answers"])
            rows.append((row["id"], score, row["prompt_tokens"]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fullkv", required=True)
    parser.add_argument("--naive", required=True)
    parser.add_argument("--quest", required=True)
    parser.add_argument("--clusterkv", required=True)
    parser.add_argument("--shadowkv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    groups = {
        "FullKV": read_jsonl(args.fullkv),
        "Naive": read_jsonl(args.naive),
        "Quest": read_jsonl(args.quest),
        "ClusterKV": read_jsonl(args.clusterkv),
        "ShadowKV": read_jsonl(args.shadowkv, shadow=True),
    }
    reference_ids = [row[0] for row in groups["FullKV"]]
    for method, rows in groups.items():
        if [row[0] for row in rows] != reference_ids:
            raise ValueError(f"{method} did not run the same examples in the same order")

    summary = []
    for method, rows in groups.items():
        summary.append({
            "method": method,
            "samples": len(rows),
            "mean_f1": 100 * sum(row[1] for row in rows) / len(rows),
            "mean_prompt_tokens": sum(row[2] for row in rows) / len(rows),
        })
    with open(args.output, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(f"{row['method']}: n={row['samples']} F1={row['mean_f1']:.2f}")


if __name__ == "__main__":
    main()
