import os, time
from requests.exceptions import ProxyError, SSLError
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import numpy as np
import random
import argparse
from accuracy.patch import parse_common_args, enable_attention_eval, get_config_output_affix
from accuracy.cluster_attention import cluster_reset
from accuracy.recall_stats import collect_model_recall_stats, reset_model_recall_stats


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser = parse_common_args(parser)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, help="task name", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_idx", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="Number of examples to run; 0 runs the full dataset")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Optional local parquet/json/jsonl dataset path")
    parser.add_argument("--output_root", type=str, default="pred")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if any(name in model_name for name in ("glm4", "intern", "llama3", "qwen3")):
        template_args = {}
        if "qwen3" in model_name:
            # Qwen3 otherwise starts a thinking trace, which consumes the short
            # answer budget used by LongBench QA tasks.
            template_args["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            **template_args,
        )
    return prompt


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    model_name,
):
    preds = []
    for json_obj in data:
        if args.cluster:
            cluster_reset(model)
        if args.recall_stat:
            reset_model_recall_stats(model)
        # The local parquet already stores the official prompt as context/question/prefix.
        # Avoid applying the LongBench template twice.
        if "question" in json_obj and "answer_prefix" in json_obj:
            prompt = json_obj["context"] + json_obj["question"] + json_obj["answer_prefix"]
        else:
            prompt = prompt_format.format(**json_obj)
        chat_applied = dataset not in [
            "trec",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]
        if chat_applied:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt",
                              add_special_tokens=not chat_applied).input_ids[0]
        if len(input_ids) > max_length:
            left = max_length // 2
            input_ids = torch.cat([input_ids[:left], input_ids[-(max_length-left):]])
        input_ids = input_ids.unsqueeze(0).to("cuda")

        if "glm4" in model_name:
            raise ValueError("This controlled runner currently supports the Llama models only")
        with torch.no_grad():
            output = model(input_ids=input_ids,
                           past_key_values=DynamicCache.from_legacy_cache(), use_cache=True)
            past_key_values = output.past_key_values
            logits = output.logits[:, -1, :]
            eos = model.generation_config.eos_token_id
            stop_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
            generated_content = []
            for step in range(max_gen):
                pred_token_idx = logits.argmax(dim=-1).unsqueeze(1)
                token = pred_token_idx.item()
                generated_content.append(token)
                if token in stop_ids or step + 1 == max_gen:
                    break
                output = model(input_ids=pred_token_idx, past_key_values=past_key_values,
                               use_cache=True)
                past_key_values = output.past_key_values
                logits = output.logits[:, -1, :]

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        # pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        record = {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
                "id": str(json_obj.get("_id", "")),
                "prompt_tokens": int(input_ids.shape[-1]),
                "budget_policy": "shared_kv_all_history_v2" if args.strict_total_budget else "legacy",
                "gqa_policy": args.gqa_policy,
            }
        if args.recall_stat:
            record.update(collect_model_recall_stats(model))
        preds.append(record)
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device):
    dtype = getattr(torch, args.dtype)
    if "intern" in model_name or "qwen" in model_name or "glm4" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=dtype,
            device_map="auto", low_cpu_mem_usage=True,
            attn_implementation="sdpa", use_cache=True
        ).to(device)
    elif "llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(
            path, torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True,
            attn_implementation="sdpa", use_cache=True
        )
    else:
        assert False
    model = model.eval()

    if args.quest or args.cluster:
        enable_attention_eval(model_name, model, args)

    return model, tokenizer

def load_model_with_retry(model_path, model_name, device, retries=3, delay=1):
    for attempt in range(retries):
        try:
            model, tokenizer = load_model_and_tokenizer(model_path, model_name, device)
            return model, tokenizer
        except (ProxyError, SSLError) as e:
            print(f"Attempt {attempt + 1} failed due to network error: {e}")
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                raise  # Re-raise the last exception if all retries fail

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    if args.strict_total_budget and (args.quest or args.cluster):
        if not any(name in args.model for name in ("llama", "qwen3")) or args.gqa_policy != "qavg":
            raise ValueError("strict accuracy requires Llama/Qwen3 with --gqa_policy qavg")
        if args.cluster and not 0 <= args.sink < args.token_budget:
            raise ValueError("sink must be smaller than the complete token budget")
        if args.quest and (args.token_budget < 3 * args.chunk_size or args.token_budget % args.chunk_size):
            raise ValueError("Quest budget must be a multiple of chunk_size and cover at least 3 pages")
    assert not (args.quest and args.cluster)     # cannot be enabled at same time
    if args.dist_t != "cosine":
        assert args.debug
    
    model2path = json.load(open("../config/model2path.json", "r"))
    model2maxlen = json.load(open("../config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    # define your model
    model, tokenizer = load_model_with_retry(
        model2path[model_name], model_name, device
    )
    max_length = model2maxlen[model_name]
    if args.task is not None:
        datasets = [args.task]
    else:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    if not os.path.exists("debug"):
        os.makedirs("debug")
    for dataset in datasets:
        if args.data_path:
            extension = os.path.splitext(args.data_path)[1].lstrip(".")
            loader = "json" if extension in ("json", "jsonl") else extension
            data = load_dataset(loader, data_files=args.data_path, split="train")
        elif args.e:
            data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
        else:
            data = load_dataset("THUDM/LongBench", f"{dataset}", split="test")

        if args.e:
            res_dir = "debug" if args.debug or args.data_idx is not None else "pred_e"
            if not os.path.exists(f"{res_dir}/{model_name}"):
                os.makedirs(f"{res_dir}/{model_name}")
            out_path = f"{res_dir}/{model_name}/{dataset}.jsonl"
            if args.quest:
                out_path = f"{res_dir}/{model_name}/{dataset}-{args.token_budget}.jsonl"
            else:
                out_path = f"{res_dir}/{model_name}/{dataset}.jsonl"
        else:
            res_dir = "debug" if args.debug or args.data_idx is not None else args.output_root
            if not os.path.exists(f"{res_dir}/{model_name}"):
                os.makedirs(f"{res_dir}/{model_name}")
            config_affix = get_config_output_affix(args)
            out_path = f"{res_dir}/{model_name}/{dataset}{config_affix}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        if args.debug:
            data = data.select(range(1))
        elif args.data_idx is not None:
            data = data.select(range(args.data_idx, args.data_idx+1))
        elif args.limit > 0:
            data = data.select(range(min(args.limit, len(data))))
        completed_ids = set()
        if args.resume and os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as source:
                completed_ids = {json.loads(line)["id"] for line in source if line.strip()}
        else:
            open(out_path, "w").close()
        pending = [row for row in data if str(row.get("_id", "")) not in completed_ids]
        with open(out_path, "a", encoding="utf-8") as output:
            for row in tqdm(pending, desc=f"{dataset} remaining"):
                pred = get_pred(
                    model,
                    tokenizer,
                    [row],
                    max_length,
                    max_gen,
                    prompt_format,
                    dataset,
                    model_name,
                )[0]
                json.dump(pred, output, ensure_ascii=False)
                output.write("\n")
                output.flush()
