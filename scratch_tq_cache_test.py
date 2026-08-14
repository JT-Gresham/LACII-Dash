"""Standalone validation of the TurboQuant KV cache against the REAL transformers 5.x Cache API.
Run on om3nbox: ~/imenv/bin/python scratch_tq_cache_test.py
  (1) unit: multi-step update vs reference cat of raw K/V (rel error) + get_seq_length + crop
  (2) e2e: greedy generate Qwen2.5-0.5B-Instruct with DynamicCache vs TurboQuantCache (turbo3/turbo4)
"""
import sys, torch
from transformers.cache_utils import DynamicCache, DynamicLayer
import kv_quant as KQ

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEV} torch={torch.__version__}")


def make_cache(name):
    kb, vb = KQ.preset_bits(name)
    def qf(head_dim, device, dtype):
        return KQ.TurboQuantizer(torch, head_dim, key_bits=kb, value_bits=vb,
                                 device=device, dtype=dtype)
    return KQ.make_turboquant_cache(DynamicCache, DynamicLayer, qf)


def unit_test(name):
    print(f"\n=== UNIT: {name} ===")
    b, h, d = 1, 2, 64
    cache = make_cache(name)
    ref_k, ref_v = [], []
    torch.manual_seed(0)
    # prefill 10 + 6 decode steps of 1
    for step, q in enumerate([10] + [1] * 6):
        k = torch.randn(b, h, q, d, device=DEV, dtype=torch.bfloat16)
        v = torch.randn(b, h, q, d, device=DEV, dtype=torch.bfloat16)
        ref_k.append(k.float()); ref_v.append(v.float())
        ok, ov = cache.update(k, v, 0)
        full_k = torch.cat(ref_k, dim=-2); full_v = torch.cat(ref_v, dim=-2)
        assert ok.shape == full_k.shape, (ok.shape, full_k.shape)
        assert cache.get_seq_length() == full_k.shape[-2], cache.get_seq_length()
    rel_k = (ok.float() - full_k).norm() / full_k.norm()
    rel_v = (ov.float() - full_v).norm() / full_v.norm()
    print(f"  seq_len={cache.get_seq_length()} (expect 16)  rel_k={rel_k:.3f} rel_v={rel_v:.3f}")
    # crop to 12
    cache.layers[0].crop(12)
    print(f"  after crop(12): seq_len={cache.get_seq_length()} (expect 12)")
    assert cache.get_seq_length() == 12
    # one more decode after crop must still work + stay correct length
    k = torch.randn(b, h, 1, d, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(b, h, 1, d, device=DEV, dtype=torch.bfloat16)
    ok, ov = cache.update(k, v, 0)
    print(f"  post-crop decode: seq_len={cache.get_seq_length()} (expect 13) out={tuple(ok.shape)}")
    assert cache.get_seq_length() == 13
    print(f"  UNIT {name} OK")


def gen_test(model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"\n=== E2E generate: {model_dir} ===")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to(DEV).eval()
    msgs = [{"role": "user", "content": "In one sentence, what is the capital of France and why is it famous?"}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(DEV)

    def run(cache, label):
        torch.manual_seed(0)
        with torch.inference_mode():
            out = model.generate(ids, max_new_tokens=48, do_sample=False,
                                 past_key_values=cache, use_cache=True,
                                 pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        print(f"  [{label}] {txt!r}")
        return out[0, ids.shape[1]:].tolist()

    base = run(DynamicCache(), "bf16-KV (DynamicCache)")
    for name in ("turbo4", "turbo3"):
        toks = run(make_cache(name), name)
        n = min(len(base), len(toks))
        agree = sum(1 for i in range(n) if base[i] == toks[i])
        first_div = next((i for i in range(n) if base[i] != toks[i]), n)
        print(f"     -> {name}: {agree}/{n} tokens match baseline; first divergence @ {first_div}")


if __name__ == "__main__":
    unit_test("turbo3")
    unit_test("turbo4")
    md = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/sixoffive/infinitemodel/models/Qwen--Qwen2.5-0.5B-Instruct"
    gen_test(md)
    print("\nALL DONE")
