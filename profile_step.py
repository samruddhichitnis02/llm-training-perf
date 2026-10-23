"""
profile_step.py  (llm-training-perf)

Runs a few training steps under the PyTorch profiler and answers one question:
where does the GPU time go?

  python profile_step.py --size small --batch 2 --seq 512 --machine-label dev

It prints:
  1. The top GPU kernels by time.
  2. A breakdown by bucket (matmul, attention, optimizer, memory copies, rest).
  3. How much of each step the GPU sat idle (a sign of launch overhead or CPU stalls).

It saves:
  profiles/<name>.json   full Chrome trace, open it at chrome://tracing or
                         ui.perfetto.dev (large file, ignored by git)
  results/profile_<name>.json   small summary you commit to the repo

Needs bench_train.py in the same folder. The default config is the best one
that needs no custom kernels (bf16, fused attention, fused Adam), so the
profile shows what is still worth fusing by hand.
"""

import argparse
import json
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile

from bench_train import LM, SIZES, VOCAB, count_params

BUCKETS = [
    ("optimizer", ("adam", "multi_tensor")),
    ("attention", ("flash", "fmha", "softmax", "attention")),
    ("matmul", ("gemm", "cutlass", "cublas", "xmma", "gemv")),
    ("memcpy_memset", ("memcpy", "memset")),
]


def bucket_of(name):
    n = name.lower()
    for bucket, keys in BUCKETS:
        if any(k in n for k in keys):
            return bucket
    return "elementwise_and_other"


def dev_time_us(e):
    """GPU time of one profiler row, in microseconds, across torch versions."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(e, attr, None)
        if v is not None:
            return v
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=SIZES.keys(), default="small")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--attn", choices=["naive", "sdpa"], default="sdpa")
    p.add_argument("--no-fused-adam", action="store_true")
    p.add_argument("--ckpt", action="store_true", help="activation checkpointing")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=3, help="steps to profile")
    p.add_argument("--top", type=int, default=15, help="kernels to print")
    p.add_argument("--machine-label", default="unlabeled")
    p.add_argument("--out", default="results")
    p.add_argument("--trace-dir", default="profiles")
    a = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Check your driver and your PyTorch build.")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.trace_dir, exist_ok=True)

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    fused = not a.no_fused_adam
    use_bf16 = a.precision == "bf16"

    model = LM(SIZES[a.size], a.seq, a.attn, a.ckpt).to(dev)
    total, n_mm = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=fused)
    x = torch.randint(0, VOCAB, (a.batch, a.seq), device=dev)
    y = torch.randint(0, VOCAB, (a.batch, a.seq), device=dev)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    for _ in range(a.warmup):
        step()
    torch.cuda.synchronize()

    # Clean step time measured without the profiler, so the idle estimate is fair.
    t0 = time.perf_counter()
    for _ in range(10):
        step()
    torch.cuda.synchronize()
    wall_per_step_us = (time.perf_counter() - t0) / 10 * 1e6

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(a.steps):
            step()
        torch.cuda.synchronize()

    name = (f"{a.machine_label}_{a.size}_{a.precision}_{a.attn}"
            f"{'_fused' if fused else ''}{'_ckpt' if a.ckpt else ''}")
    trace_path = os.path.join(a.trace_dir, f"trace_{name}.json")
    prof.export_chrome_trace(trace_path)

    # Keep only rows that are real GPU work, not CPU wrapper ops.
    rows = [e for e in prof.key_averages()
            if dev_time_us(e) > 0
            and not e.key.startswith(("aten::", "autograd::", "Optimizer"))]
    rows.sort(key=dev_time_us, reverse=True)
    busy_us = sum(dev_time_us(e) for e in rows)
    busy_per_step_us = busy_us / a.steps
    idle_pct = max(0.0, 100 * (1 - busy_per_step_us / wall_per_step_us))
    # On one stream, GPU busy time cannot exceed the real step time.
    inflated = busy_per_step_us > wall_per_step_us * 1.05

    kernels = []
    buckets = {}
    for e in rows:
        t = dev_time_us(e)
        b = bucket_of(e.key)
        buckets[b] = buckets.get(b, 0) + t
        kernels.append(dict(
            kernel=e.key,
            bucket=b,
            calls_per_step=round(e.count / a.steps, 1),
            ms_per_step=round(t / a.steps / 1000, 3),
            pct_of_gpu_time=round(100 * t / busy_us, 1),
        ))

    print(f"\nconfig: size={a.size} batch={a.batch} seq={a.seq} "
          f"{a.precision} {a.attn} fused_adam={fused} ckpt={a.ckpt}")
    print(f"step time {wall_per_step_us / 1000:.1f} ms (measured without the profiler), "
          f"GPU busy {busy_per_step_us / 1000:.1f} ms, "
          f"GPU idle about {idle_pct:.0f}%\n")
    if inflated:
        print("WARNING: GPU busy time is larger than the real step time, which cannot\n"
              "happen on one stream. The profiler inflated the kernel times in this run.\n"
              "Trust the shares (%) only. Ignore the ms values and the idle figure.\n")

    print(f"Top {a.top} GPU kernels")
    print(f"{'kernel':62s} {'calls':>6s} {'ms/step':>8s} {'%':>6s}")
    for k in kernels[:a.top]:
        short = k["kernel"] if len(k["kernel"]) <= 60 else k["kernel"][:57] + "..."
        print(f"{short:62s} {k['calls_per_step']:6.0f} "
              f"{k['ms_per_step']:8.3f} {k['pct_of_gpu_time']:6.1f}")

    print("\nBy bucket")
    bucket_rows = []
    for b, t in sorted(buckets.items(), key=lambda kv: kv[1], reverse=True):
        row = dict(bucket=b, ms_per_step=round(t / a.steps / 1000, 3),
                   pct_of_gpu_time=round(100 * t / busy_us, 1))
        bucket_rows.append(row)
        print(f"{b:26s} {row['ms_per_step']:8.3f} ms  {row['pct_of_gpu_time']:5.1f}%")

    major, minor = torch.cuda.get_device_capability()
    summary = dict(
        env=dict(arch=f"sm_{major}{minor}", torch=torch.__version__,
                 cuda=torch.version.cuda, machine_label=a.machine_label),
        config=dict(size=a.size, batch=a.batch, seq=a.seq, precision=a.precision,
                    attn=a.attn, fused_adam=fused, ckpt=a.ckpt,
                    params_total_m=round(total / 1e6, 1)),
        wall_ms_per_step=round(wall_per_step_us / 1000, 2),
        gpu_busy_ms_per_step=round(busy_per_step_us / 1000, 2),
        gpu_idle_pct=round(idle_pct, 1),
        timing_inflated=inflated,
        buckets=bucket_rows,
        top_kernels=kernels[:25],
    )
    summary_path = os.path.join(a.out, f"profile_{name}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved {summary_path}")
    print(f"saved {trace_path}  (open at ui.perfetto.dev)")


if __name__ == "__main__":
    main()