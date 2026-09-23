"""Exercise patched HF Llama forwards, including short-prompt budget crossing."""
import argparse
import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).parent / 'kv-repos/ClusterKV'))
from accuracy import patch
from accuracy.recall_stats import collect_model_recall_stats


@torch.inference_mode()
def main():
    torch.manual_seed(9)
    for method in ('naive', 'quest', 'cluster'):
        parser = patch.parse_common_args(argparse.ArgumentParser())
        args = parser.parse_args(['--model', 'llama3.1-8b-chat-32k',
            '--cluster' if method == 'cluster' else '--quest',
            '--token_budget', '128', '--chunk_size', '1' if method == 'naive' else '16',
            '--gqa_policy', 'qavg', '--strict_total_budget', '--recall_stat',
            '--nlist', '400', '--fit_iter', '2'])
        config = LlamaConfig(vocab_size=128, hidden_size=256, intermediate_size=512,
                             num_hidden_layers=2, num_attention_heads=8,
                             num_key_value_heads=2, max_position_embeddings=512)
        config._attn_implementation = 'sdpa'
        model = LlamaForCausalLM(config).to(device='cuda', dtype=torch.bfloat16).eval()
        patch.layer_id = config.num_hidden_layers
        patch.enable_attention_eval(args.model, model, args)
        for length in (120, 151):
            for layer in model.model.layers:
                layer.self_attn.key_centroids = None
            cache = DynamicCache()
            tokens = torch.randint(0, 128, (1, length), device='cuda')
            output = model(tokens, past_key_values=cache, use_cache=True)
            for step in range(16):
                output = model(output.logits[:, -1:].argmax(-1),
                               past_key_values=cache, use_cache=True)
                assert output.logits.isfinite().all()
            stats = collect_model_recall_stats(model)
            assert stats['mean_selected_tokens'] <= 128, stats
            if method == 'cluster':
                for layer in model.model.layers:
                    attn = layer.self_attn
                    assert attn.cluster_key_indices.shape[-1] == length + 16 - args.sink
                    assert (attn.cluster_key_size.sum(-1) == length + 16 - args.sink).all()
            print(method, 'prompt', length, stats, flush=True)
        del model, output, cache
        torch.cuda.empty_cache()
    print('PASS: actual HF forward, BF16, all layers, short/long prompt and dynamic clustering')


if __name__ == '__main__':
    main()
