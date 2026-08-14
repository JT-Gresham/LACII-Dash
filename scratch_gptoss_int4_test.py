"""Standalone validation of the gpt-oss int4 expert recipe (#166) vs the REAL GptOssExperts.forward.
Confirms: transpose [E,H,2I]->[E,2I,H] (+ down [E,I,H]->[E,H,I]) before _pack4_3d, then the fused
op with interleaved clamped SwiGLU + biases reproduces the bf16 experts. Run on om3nbox (ROCm).
  ~/imenv/bin/python scratch_gptoss_int4_test.py
"""
import torch
import client as C
from transformers.models.gpt_oss import modeling_gpt_oss as M

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


class Cfg:
    def __init__(s):
        s.num_local_experts = 8
        s.hidden_size = 128
        s.intermediate_size = 256
        s.num_experts_per_tok = 2
        s._experts_implementation = "eager"


def main():
    cfg = Cfg()
    E, H, I, TK = cfg.num_local_experts, cfg.hidden_size, cfg.intermediate_size, cfg.num_experts_per_tok
    experts = M.GptOssExperts(cfg).to(DEV, dtype=torch.bfloat16)
    # random (but small) weights/biases so the reference is well-conditioned
    with torch.no_grad():
        experts.gate_up_proj.normal_(0, 0.05)
        experts.down_proj.normal_(0, 0.05)
        experts.gate_up_proj_bias.normal_(0, 0.02)
        experts.down_proj_bias.normal_(0, 0.02)

    T = 6
    hidden = torch.randn(T, H, device=DEV, dtype=torch.bfloat16) * 0.5
    router_indices = torch.stack([torch.randperm(E, device=DEV)[:TK] for _ in range(T)])   # [T,TK]
    routing_weights = torch.softmax(torch.randn(T, TK, device=DEV, dtype=torch.bfloat16), dim=-1)

    ref = experts.forward(hidden, router_indices, routing_weights).float()               # bf16 reference

    # --- int4 path: transpose-pack + fused op with gpt-oss activation + biases ---
    op = C._w4a16_moe_op()
    assert op is not None, "no fused MoE op (need triton/ROCm)"
    gu_t = experts.gate_up_proj.data.transpose(1, 2).contiguous()   # [E,H,2I] -> [E,2I,H]
    dn_t = experts.down_proj.data.transpose(1, 2).contiguous()      # [E,I,H]  -> [E,H,I]
    gu = C._pack4_3d(gu_t).to(DEV)
    dn = C._pack4_3d(dn_t).to(DEV)
    gub = experts.gate_up_proj_bias.data.to(DEV)
    dnb = experts.down_proj_bias.data.to(DEV)
    alpha, limit = experts.alpha, experts.limit

    eid = router_indices.reshape(-1)
    w = routing_weights.reshape(-1).to(hidden.dtype)
    xb = hidden.repeat_interleave(TK, dim=0)
    yb = op(xb, eid, gu.qweight, gu.scale, gu.zero, gu.group_size, gu.in_features)         # [B, 2I]
    # DIAG: pure-linear gate_up rel (isolates int4 packing from activation amplification)
    gu_ref = torch.stack([xb[i] @ experts.gate_up_proj.data[eid[i]] + gub[eid[i]] for i in range(xb.shape[0])]).float()
    yb_full = (yb + gub[eid]).float()
    print(f"  DIAG gate_up (pure int4 linear) rel={((yb_full-gu_ref).abs().mean()/(gu_ref.abs().mean()+1e-6)).item():.4f}")
    yb = yb + gub[eid]
    gate = yb[..., ::2].clamp(max=limit)
    up = yb[..., 1::2].clamp(min=-limit, max=limit)
    h = (up + 1) * (gate * torch.sigmoid(gate * alpha))                                    # [B, I]
    zb = op(h, eid, dn.qweight, dn.scale, dn.zero, dn.group_size, dn.in_features)          # [B, H]
    zb = (zb + dnb[eid]) * w[:, None]
    out = torch.zeros_like(hidden)
    tok = torch.arange(T, device=DEV).repeat_interleave(TK)
    out.index_add_(0, tok, zb.to(out.dtype))
    out = out.float()

    rel = ((out - ref).abs().mean() / (ref.abs().mean() + 1e-6)).item()
    relmax = ((out - ref).abs().max() / (ref.abs().max() + 1e-6)).item()
    print(f"gpt-oss int4 expert recipe: rel={rel:.4f} max={relmax:.4f}  (expect ~0.05 int4-vs-bf16)")
    print("PASS" if rel < 0.12 else "FAIL")


if __name__ == "__main__":
    main()
