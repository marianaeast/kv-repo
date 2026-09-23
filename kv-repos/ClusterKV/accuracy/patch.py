import inspect
import types
from transformers.models.llama.modeling_llama import LlamaAttention
try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
except ImportError:
    Qwen3Attention = None
from accuracy.quest_attention import forward_quest, forward_quest_glm
from accuracy.cluster_attention import forward_cluster, forward_cluster_glm, apply_cluster_config
from accuracy.recall_stats import new_recall_stats

global layer_id
layer_id = 32

def parse_common_args(parser):
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "llama3-8b-chat-4k",
            "llama3-8b-chat-8k",
            "llama3.1-8b-chat-4k",
            "llama3.1-8b-chat-8k",
            "llama3.1-8b-chat-32k",
            "qwen3-8b-chat-32k",
            "glm4-9b-chat-4k",
            "glm4-9b-chat-8k",
            "glm4-9b-chat-32k",
        ],
    )
    parser.add_argument("--token_budget", type=int, default=1024)

    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--quest", action="store_true", help="Enable Quest Attention")

    parser.add_argument("--sink", type=int, default=16)
    parser.add_argument("--cluster", action="store_true", help="Enable ClusterKV Attention")
    parser.add_argument("--head_sel", type=str, choices=["truc", "pad"], default="truc",
                        help="truncate or pad for different heads to make same selection budget")
    parser.add_argument("--balance", action="store_true", help="Use Balanced KMeans")
    parser.add_argument("--nlist", type=int, default=400, help="Number of clusters")
    parser.add_argument("--fit_iter", type=int, default=20, help="Max steps for clustering")
    parser.add_argument("--gqa_policy", type=str, choices=["qavg"], default=None)
    parser.add_argument("--strict_total_budget", action="store_true",
                        help="Include generated tokens in the shared KV-head budget")
    parser.add_argument("--dist_t", type=str, 
                        choices=["cosine", "inner_product", "l2", "l1", "euclidean", 
                                 "chebyshev", "canberra"], 
                        default="cosine", help="Distance for clustering")

    parser.add_argument("--cache_steps", type=int, default=0, 
                        help="Stat cache hit rate of recent steps")
    parser.add_argument("--topk_stat", action="store_true", help="Stat hit rate of TopK tokens")
    parser.add_argument(
        "--recall_stat",
        action="store_true",
        help="Store aggregate selection recall against exact token-level Top-K",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    return parser


def enable_attention_eval(model_name, model, args):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_attention_eval(model_name, module, args)

        global layer_id
        attention_types = (LlamaAttention,)
        if Qwen3Attention is not None:
            attention_types += (Qwen3Attention,)
        if isinstance(module, attention_types):
            fallback_layer_id = layer_id - 1
            layer_id = fallback_layer_id
            module.layer_id = getattr(module, "layer_idx", fallback_layer_id)
            module._bypasskv_model_type = getattr(module.config, "model_type", "llama")
            module._bypasskv_cache_argument = (
                "past_key_values"
                if "past_key_values" in inspect.signature(module.forward).parameters
                else "past_key_value"
            )
            module.num_heads = getattr(module, "num_heads", module.config.num_attention_heads)
            module.num_key_value_heads = getattr(
                module, "num_key_value_heads", module.config.num_key_value_heads
            )
            module.num_key_value_groups = module.num_heads // module.num_key_value_heads
            module.flash_forward = module.forward
            module.cache_steps = args.cache_steps
            module.token_budget = args.token_budget
            module.chunk_size = args.chunk_size
            module.cluster_cache = None
            module.gqa_policy = args.gqa_policy
            module.strict_total_budget = args.strict_total_budget
            if args.quest:
                module.forward = types.MethodType(forward_quest, module)
                module.gen = False
            elif args.cluster:
                module.forward = types.MethodType(forward_cluster, module)
                apply_cluster_config(module, args)
            module.topk_stat = True if args.topk_stat else False
            module.recall_stats = new_recall_stats() if args.recall_stat else None

        elif "glm4" in model_name and module.__class__.__name__ == "SelfAttention":
            module.flash_forward = module.forward
            module.cache_steps = args.cache_steps
            module.token_budget = args.token_budget
            module.chunk_size = args.chunk_size
            module.cluster_cache = None
            if args.quest:
                module.forward = types.MethodType(forward_quest_glm, module)
                module.gen = False
            elif args.cluster:
                module.forward = types.MethodType(forward_cluster_glm, module)
                apply_cluster_config(module, args)
            module.topk_stat = True if args.topk_stat else False
            module.recall_stats = new_recall_stats() if args.recall_stat else None


def get_config_output_affix(args):
    config_affix = ''
    if args.quest:
        if args.chunk_size == 16:
            config_affix = f"-{args.token_budget}"
        else:
            config_affix = f"-{args.token_budget}c{args.chunk_size}"
    elif args.cluster:
        km_id = ("h" + args.head_sel + ("b" if args.balance else "") 
                 + str(args.nlist) + f"fi{args.fit_iter}")
        if args.sink > 0:
            km_id += f"sink{args.sink}"
        config_affix = f"-{km_id}_{args.token_budget}"
    if (args.quest or args.cluster) and args.gqa_policy is not None:
        config_affix += f"-gqa{args.gqa_policy}"
    if (args.quest or args.cluster) and args.strict_total_budget:
        config_affix += "-totalv2"
    if args.recall_stat:
        config_affix += "-recall"
    return config_affix
