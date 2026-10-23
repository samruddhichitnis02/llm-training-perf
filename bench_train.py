"""
bench_train.py  (llm-training-perf)

One file that does three jobs:

  1. Single run:   python bench_train.py --size small --precision bf16 --attn sdpa
  2. Ladder sweep: python bench_train.py --sweep --size small
                   Runs baseline -> bf16 -> sdpa -> fused Adam -> torch.compile
                   in one go and prints a before/after table.
  3. Calibration:  python bench_train.py --calibrate-only
                   Measures peak matmul speed (fp32, fp16, bf16, fp8) and
                   memory bandwidth. The performance model needs these numbers.

Results are saved as JSON in the results folder. The GPU model name is NOT
saved by default. Only the architecture tag (for example sm_86) is recorded,
so the numbers stay honest and tied to the hardware generation.

Works on Windows and Linux. torch.compile needs Triton, which is not
officially supported on Windows, so use --no-compile there.
"""

import argparse
import gc
import json
import math
import os
import platform
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

VOCAB = 32000

# Llama style sizes. d = model width, L = layers, H = heads, ffn = MLP width.
SIZES = {
    "tiny": dict(d=512, L=8, H=8, ffn=1408),       # about 60M params, smoke tests
    "small": dict(d=768, L=12, H=12, ffn=2048),    # about 135M params
    "medium": dict(d=1024, L=24, H=16, ffn=2816),  # about 375M params
    "large": dict(d=2048, L=16, H=16, ffn=5504),   # about 940M params, needs a big card
}


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.w.float()).to(x.dtype)


def rope_cache(T, head_dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(T).float()
    freqs = torch.outer(t, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None].to(q.dtype)
    sin = sin[None, None].to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, d, H, impl):
        super().__init__()
        self.H, self.hd, self.impl = H, d // H, impl
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, k, cos, sin)
        if self.impl == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Naive attention: builds the full T x T score matrix in memory.
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            att = att.masked_fill(mask, float("-inf"))
            att = torch.softmax(att.float(), dim=-1).to(q.dtype)
            y = att @ v
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.o(y)


class MLP(nn.Module):
    def __init__(self, d, ffn):
        super().__init__()
        self.w1 = nn.Linear(d, ffn, bias=False)
        self.w3 = nn.Linear(d, ffn, bias=False)
        self.w2 = nn.Linear(ffn, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d, H, ffn, impl):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = Attention(d, H, impl)
        self.n2 = RMSNorm(d)
        self.mlp = MLP(d, ffn)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class LM(nn.Module):
    def __init__(self, cfg, seq, attn_impl, use_ckpt):
        super().__init__()
        d, L, H, ffn = cfg["d"], cfg["L"], cfg["H"], cfg["ffn"]
        self.use_ckpt = use_ckpt
        self.emb = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList(Block(d, H, ffn, attn_impl) for _ in range(L))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        cos, sin = rope_cache(seq, d // H)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets):
        x = self.emb(idx)
        for blk in self.blocks:
            if self.use_ckpt and self.training:
                x = checkpoint(blk, x, self.cos, self.sin, use_reentrant=False)
            else:
                x = blk(x, self.cos, self.sin)
        logits = self.head(self.norm(x))
        return F.cross_entropy(logits.float().view(-1, VOCAB), targets.view(-1))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return total, total - model.emb.weight.numel()


# ----------------------------------------------------------------------------
# Calibration: how fast can this GPU really go?
# ----------------------------------------------------------------------------

def time_gpu(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def matmul_tflops(dtype, n=8192, iters=20):
    torch.backends.cuda.matmul.allow_tf32 = False
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    t = time_gpu(lambda: a @ b, iters)
    return 2 * n ** 3 / t / 1e12


def fp8_matmul_tflops(n=8192, iters=20):
    try:
        a = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
        sa = torch.tensor(1.0, device="cuda")
        sb = torch.tensor(1.0, device="cuda")
        t = time_gpu(lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb,
                                              out_dtype=torch.bfloat16), iters)
        return 2 * n ** 3 / t / 1e12
    except Exception as e:  # not every card or torch build supports this
        print(f"  fp8 matmul unavailable: {str(e)[:120]}")
        return None


def mem_bandwidth_gbs(mb=512, iters=20):
    n = mb * 1024 * 1024 // 4
    a = torch.empty(n, device="cuda")
    b = torch.empty_like(a)
    t = time_gpu(lambda: b.copy_(a), iters)
    return 2 * n * 4 / t / 1e9


_PEAK_CACHE = {}


def measured_peak(precision):
    if precision not in _PEAK_CACHE:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        _PEAK_CACHE[precision] = matmul_tflops(dtype)
    return _PEAK_CACHE[precision]


# ----------------------------------------------------------------------------
# One benchmark run
# ----------------------------------------------------------------------------

def bench(a, label, precision, attn, use_compile, fused, use_ckpt):
    dev = "cuda"
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = SIZES[a.size]
    res = dict(label=label, size=a.size, batch=a.batch, seq=a.seq,
               precision=precision, attn=attn, compile=use_compile,
               fused_adam=fused, ckpt=use_ckpt, status="ok")
    model = opt = run_model = None

    def push(name):
        if a.nvtx:
            torch.cuda.nvtx.range_push(name)

    def pop():
        if a.nvtx:
            torch.cuda.nvtx.range_pop()

    try:
        model = LM(cfg, a.seq, attn, use_ckpt).to(dev)
        total, n_mm = count_params(model)
        run_model = torch.compile(model) if use_compile else model
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=fused)
        x = torch.randint(0, VOCAB, (a.batch, a.seq), device=dev)
        y = torch.randint(0, VOCAB, (a.batch, a.seq), device=dev)
        use_bf16 = precision == "bf16"

        def step():
            push("forward")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss = run_model(x, y)
            pop()
            push("backward")
            loss.backward()
            pop()
            push("optimizer")
            opt.step()
            opt.zero_grad(set_to_none=True)
            pop()
            return loss

        for _ in range(a.warmup):
            loss = step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(a.steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        med = statistics.median(times)
        tok_s = a.batch * a.seq / med
        # Model FLOPs per token: 6 per matmul parameter, plus attention scores.
        # Recompute from checkpointing is NOT counted, which is the usual MFU rule.
        flops_per_tok = 6 * n_mm + 12 * cfg["L"] * cfg["d"] * a.seq
        tflops = tok_s * flops_per_tok / 1e12
        peak = measured_peak(precision)
        peak_bf16 = measured_peak("bf16")
        final_loss = float(loss.detach())

        res.update(
            params_total_m=round(total / 1e6, 1),
            params_matmul_m=round(n_mm / 1e6, 1),
            ms_per_step=round(med * 1000, 2),
            tokens_per_sec=round(tok_s),
            achieved_tflops=round(tflops, 2),
            measured_matmul_peak_tflops=round(peak, 1),
            mfu_vs_measured_pct=round(100 * tflops / peak, 1),
            mfu_vs_bf16_peak_pct=round(100 * tflops / peak_bf16, 1),
            peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
            final_loss=round(final_loss, 3),
            loss_finite=math.isfinite(final_loss),
        )
        if a.peak_tflops:
            res["mfu_vs_spec_pct"] = round(100 * tflops / a.peak_tflops, 1)

    except torch.cuda.OutOfMemoryError:
        res["status"] = "oom"
    except Exception as e:
        res["status"] = "error: " + str(e)[:200].replace("\n", " ")
    finally:
        del model, opt, run_model
        gc.collect()
        torch.cuda.empty_cache()
        if use_compile:
            try:
                torch._dynamo.reset()
            except Exception:
                pass
    return res


def print_table(rows):
    base = next((r["tokens_per_sec"] for r in rows if r["status"] == "ok"), None)
    print()
    print(f"{'config':26s} {'status':8s} {'ms/step':>9s} {'tok/s':>10s} "
          f"{'speedup':>8s} {'TFLOPs':>8s} {'MFU%':>6s} {'memGB':>7s}")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['label']:26s} {r['status'][:40]}")
            continue
        speedup = r["tokens_per_sec"] / base if base else 0.0
        print(f"{r['label']:26s} {'ok':8s} {r['ms_per_step']:9.1f} "
              f"{r['tokens_per_sec']:10d} {speedup:7.2f}x {r['achieved_tflops']:8.1f} "
              f"{r['mfu_vs_bf16_peak_pct']:6.1f} {r['peak_mem_gb']:7.2f}")
    print("\nspeedup is tokens per second against the first row.")
    print("MFU% is measured against the bf16 matmul peak this GPU actually reached,")
    print("the same reference for every row, so the rows can be compared.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=SIZES.keys(), default="small")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--attn", choices=["naive", "sdpa"], default="naive")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--fused-adam", action="store_true")
    p.add_argument("--ckpt", action="store_true", help="activation checkpointing")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--sweep", action="store_true", help="run the optimization ladder")
    p.add_argument("--no-compile", action="store_true", help="skip compile in the sweep")
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--peak-tflops", type=float, default=None,
                   help="optional spec sheet peak, adds an MFU vs spec column")
    p.add_argument("--machine-label", default="unlabeled")
    p.add_argument("--record-device-name", action="store_true")
    p.add_argument("--nvtx", action="store_true", help="add NVTX ranges for Nsight")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Check your driver and your PyTorch build.")
    os.makedirs(a.out, exist_ok=True)

    major, minor = torch.cuda.get_device_capability()
    env = dict(arch=f"sm_{major}{minor}", torch=torch.__version__,
               cuda=torch.version.cuda, os=platform.system(),
               machine_label=a.machine_label)
    if a.record_device_name:
        env["device_name"] = torch.cuda.get_device_name()
    print("env:", env)

    if a.calibrate_only:
        print("Calibrating (about a minute)...")
        cal = dict(
            matmul_tflops_fp32=round(matmul_tflops(torch.float32), 1),
            matmul_tflops_fp16=round(matmul_tflops(torch.float16), 1),
            matmul_tflops_bf16=round(matmul_tflops(torch.bfloat16), 1),
            memory_bandwidth_gbs=round(mem_bandwidth_gbs(), 1),
        )
        fp8 = fp8_matmul_tflops()
        if fp8:
            cal["matmul_tflops_fp8"] = round(fp8, 1)
        print(json.dumps(cal, indent=2))
        path = os.path.join(a.out, f"calibration_{a.machine_label}.json")
        with open(path, "w") as f:
            json.dump(dict(env=env, calibration=cal), f, indent=2)
        print("saved", path)
        return

    if a.sweep:
        ladder = [
            ("1_baseline_fp32", "fp32", "naive", False, False, False),
            ("2_bf16", "bf16", "naive", False, False, False),
            ("3_bf16_sdpa", "bf16", "sdpa", False, False, False),
            ("4_plus_fused_adam", "bf16", "sdpa", False, True, False),
        ]
        if not a.no_compile:
            ladder.append(("5_plus_compile", "bf16", "sdpa", True, True, False))
        rows = []
        for cfg in ladder:
            print(f"running {cfg[0]} ...")
            rows.append(bench(a, *cfg))
        print_table(rows)
        name = f"sweep_{a.machine_label}_{a.size}.json"
        data = dict(env=env, rows=rows)
    else:
        label = (f"{a.precision}_{a.attn}"
                 f"{'_compile' if a.compile else ''}"
                 f"{'_fused' if a.fused_adam else ''}"
                 f"{'_ckpt' if a.ckpt else ''}")
        rows = [bench(a, label, a.precision, a.attn, a.compile, a.fused_adam, a.ckpt)]
        print_table(rows)
        name = f"run_{a.machine_label}_{a.size}_{label}.json"
        data = dict(env=env, rows=rows)

    path = os.path.join(a.out, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()