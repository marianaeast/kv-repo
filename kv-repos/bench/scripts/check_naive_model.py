"""With budget >= context, naive must agree with dense model decode (FP16 tolerance)."""
import torch
from types import SimpleNamespace
from bench_breakdown import build, init_controller, DEFAULT_MODEL
from llama31_rope_patch import install as install_rope
from naive_attention import install as install_naive

@torch.inference_mode()
def main():
    device = torch.device('cuda:0')
    args = SimpleNamespace(method='full', model_path=DEFAULT_MODEL, nlist=200, niter=20,
                           token_budget=512, sink=16, window=320, window_nlist=8, offload=False)
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(DEFAULT_MODEL, device_map=device,
        torch_dtype=torch.float16, attn_implementation='sdpa').eval()
    data = torch.load('/home/zrd/bypasskv_repo/kv-repos/bench/data/ctx32k_pg19.pt', weights_only=True)
    ids = data['input_ids'][:256].unsqueeze(0).to(device)
    out = model(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    reference = [out.logits[:, -1].clone()]
    inputs = []
    for _ in range(3):
        token = reference[-1].argmax(-1, keepdim=True)
        inputs.append(token)
        out = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        reference.append(out.logits[:, -1].clone())
    del model, out, cache
    torch.cuda.empty_cache()
    model = build(args, device)
    install_rope(model.config, device, 512)
    install_naive(model, 512, 512, device)
    actual = [model(input_ids=ids).logits[:, -1].clone()]
    for token in inputs:
        actual.append(model(input_ids=token).logits[:, -1].clone())
    for step, (got, expected) in enumerate(zip(actual, reference)):
        print('max error before assertion', step, (got-expected).abs().max().item(), flush=True)
        torch.testing.assert_close(got, expected, atol=0.06, rtol=0.02)
        assert bool(torch.isfinite(got).all())
        print(f'step={step} max_logit_error={(got-expected).abs().max().item():.6f} PASS', flush=True)
    # Clearing the sequence must replace previous cache contents on a new prompt.
    model.clusterkv_clear()
    replay = model(input_ids=ids).logits[:, -1]
    torch.testing.assert_close(replay, actual[0], atol=0, rtol=0)
    print('Naive dense-limit, sequential append and prompt replay: PASS', flush=True)

if __name__ == '__main__':
    main()
