"""Run ShadowKV on local LongBench HotpotQA with a controlled token budget.

The upstream ``sparse_budget`` counts only dynamically retrieved chunks and then
adds local and outlier chunks.  For the controlled comparison we disable those
two extra pools. With --strict_total_budget, prompt and generated tokens compete
for the same K slots; partial chunks are masked and no extra tokens are added.
Without that flag, the legacy variant retains alignment and generated tokens.
"""

import argparse
from collections import Counter
import json
import os
import re
import string
import sys

import pyarrow.parquet as pq
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models import choose_model_class
from models.budgeted_kv_cache import BudgetedShadowKVCache


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--total_budget", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=160)
    parser.add_argument("--max_input_tokens", type=int, default=31500)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0,
                        help="Number of examples; 0 runs the complete parquet")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recall_stat", action="store_true")
    parser.add_argument("--strict_total_budget", action="store_true",
                        help="Select generated tokens and partial chunks within the same K")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.total_budget < args.chunk_size or args.total_budget % args.chunk_size:
        raise ValueError("total_budget must be a positive multiple of chunk_size")

    rows = pq.read_table(args.data_path).to_pylist()
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("dataset is empty")

    model_class = choose_model_class(args.model_name)
    model = model_class(
        model_name=args.model_name,
        batch_size=1,
        device=args.device,
        max_length=args.max_input_tokens + args.max_new_tokens + 8,
        attn_mode="shadowKV",
        dtype=torch.bfloat16,
        sparse_budget=args.total_budget,
        rank=args.rank,
        chunk_size=args.chunk_size,
    )

    if args.strict_total_budget:
        model.kv_cache = BudgetedShadowKVCache(
            model.config, batch_size=1, device=args.device, dtype=torch.bfloat16,
            max_length=args.max_input_tokens + args.max_new_tokens + 8,
            sparse_budget=args.total_budget, chunk_size=args.chunk_size, rank=args.rank,
        )

    # In upstream ShadowKV these pools are outside sparse_budget.  Disable them
    # The strict cache also removes free alignment/generated tokens. The
    # legacy path below is retained only for reproducing old results.
    model.kv_cache.local_chunk = 0
    model.kv_cache.outlier_chunk = 0
    if args.recall_stat:
        model.kv_cache.enable_recall_stats(args.total_budget)
    model.print_kv_stats()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    completed = {}
    policy = "shared_kv_all_history_v2" if args.strict_total_budget else "prompt_only_v1"
    if args.resume and os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if record.get("budget_policy", "prompt_only_v1") != policy:
                    raise ValueError("Cannot resume results from a different budget policy")
                completed[record["id"]] = record
    scores = [record["f1"] / 100 for record in completed.values()]
    mode = "a" if args.resume else "w"
    with open(args.output, mode, encoding="utf-8") as output:
        for index, row in enumerate(rows):
            row_id = str(row.get("_id", index))
            if row_id in completed:
                continue
            prompt = row["context"] + row["question"] + row["answer_prefix"]
            template_args = {}
            if "qwen3" in args.model_name.lower():
                template_args["enable_thinking"] = False
            prompt = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                **template_args,
            )
            input_ids = model.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
            if len(input_ids) > args.max_input_tokens:
                left = args.max_input_tokens // 2
                input_ids = torch.cat(
                    [input_ids[:left], input_ids[-(args.max_input_tokens-left):]]
                )
            input_ids = input_ids.unsqueeze(0).to(args.device)
            prediction = model.generate(
                input_ids,
                # ShadowKV.generate emits one token before its gen_len loop.
                # Pass N-1 so every method has the same N-token upper bound.
                gen_len=args.max_new_tokens - 1,
                top_p=1.0,
                temperature=0.0,
            )[0]
            effective_budget = (
                model.kv_cache.last_max_attention_tokens if args.strict_total_budget
                else int(model.kv_cache.sparse_end)
            )
            maximum_budget = args.total_budget if args.strict_total_budget else args.total_budget + args.chunk_size - 1
            if effective_budget > maximum_budget:
                raise RuntimeError(
                    f"effective budget {effective_budget} exceeds controlled bound"
                )
            score = max(qa_f1(prediction, answer) for answer in row["answers"])
            scores.append(score)
            record = {
                "id": row_id,
                "prediction": prediction,
                "answers": row["answers"],
                "f1": score * 100,
                "prompt_tokens": int(input_ids.shape[-1]),
                "requested_total_budget": args.total_budget,
                "effective_historical_budget": effective_budget,
                "budget_policy": policy,
                "selection_scope": "prompt_and_generated" if args.strict_total_budget else "prompt",
                "gqa_policy": "softmax_group_max",
            }
            if args.recall_stat:
                record.update(model.kv_cache.get_recall_stats(use_snapshot=True))
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[{index+1}/{len(rows)}] F1={score*100:.2f} "
                f"mean={100*sum(scores)/len(scores):.2f} "
                f"tokens={input_ids.shape[-1]} effective_K={effective_budget}",
                flush=True,
            )

    summary = {
        "method": "shadowkv_equal_total_budget",
        "samples": len(scores),
        "mean_f1": 100 * sum(scores) / len(scores),
        "requested_total_budget": args.total_budget,
        "maximum_effective_historical_budget": args.total_budget if args.strict_total_budget else args.total_budget + args.chunk_size - 1,
        "budget_policy": policy,
    }
    with open(args.output + ".summary.json", "w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
