"""
Test: nn.Linear and nn.Conv1d GEMM
      with and without Ozaki Scheme II (GEMMul8 INT8 hook)

These ops use real GEMM:
  float32 → GEMMUL8_NUM_MOD_S_GEMM / GEMMUL8_FASTMODE_S_GEMM
  float64 → GEMMUL8_NUM_MOD_D_GEMM / GEMMUL8_FASTMODE_D_GEMM

Run without GEMMul8 (baseline):
    python test_ozaki_hook_RGEMM.py --save baseline_rgemm.pt

Run with GEMMul8 INT8 hook:
    LD_PRELOAD=/path/to/GEMMul8/lib/libgemmul8.so \
    GEMMUL8_BACKEND_GEMM=INT8 \
    GEMMUL8_NUM_MOD_D_GEMM=9 \
    GEMMUL8_FASTMODE_D_GEMM=1 \
    python test_ozaki_hook_RGEMM.py --dtype float64 --load baseline_rgemm.pt
"""

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn


# ── argument parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark nn.Linear & nn.Conv1d GEMM "
                    "with GEMMul8 Ozaki Scheme II hook.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = parser.add_argument_group("Problem dimensions")
    g.add_argument("--B",      type=int, default=1,       help="Batch size")
    g.add_argument("--P",      type=int, default=500_000, help="Number of particles / spatial points")
    g.add_argument("--in_ch",  type=int, default=2,      help="Input channels")
    g.add_argument("--out_ch", type=int, default=32,      help="Output channels")

    g2 = parser.add_argument_group("Precision")
    g2.add_argument("--dtype", type=str, default="float64",
                    choices=["float32", "float64"])

    g3 = parser.add_argument_group("Timing")
    g3.add_argument("--warmup", type=int, default=5)
    g3.add_argument("--iters",  type=int, default=10)

    g4 = parser.add_argument_group("Checkpoint")
    g4.add_argument("--save", type=str, default=None, metavar="FILE")
    g4.add_argument("--load", type=str, default=None, metavar="FILE")
    g4.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args


args = parse_args()

# Linear / Conv use real dtypes (S=float32, D=float64)
if args.dtype == "float64":
    data_type     = torch.float64
    _MOD_KEY      = "GEMMUL8_NUM_MOD_D_GEMM"
    _FASTMODE_KEY = "GEMMUL8_FASTMODE_D_GEMM"
else:
    data_type     = torch.float32
    _MOD_KEY      = "GEMMUL8_NUM_MOD_S_GEMM"
    _FASTMODE_KEY = "GEMMUL8_FASTMODE_S_GEMM"

B      = args.B
P      = args.P
in_ch  = args.in_ch
out_ch = args.out_ch


# ── device / GEMMul8 ──────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
maps   = open("/proc/self/maps").read()
OZAKI  = "libgemmul8" in maps

print(f"\nDevice  : {DEVICE}" + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
print(f"GEMMul8 : {'ACTIVE  moduli=' + os.environ.get(_MOD_KEY,'?') + '  fastmode=' + os.environ.get(_FASTMODE_KEY,'?') if OZAKI else 'OFF'}")
print(f"Config  : B={B}   P={P:,}  in_ch={in_ch}  out_ch={out_ch}  dtype={args.dtype}  seed={args.seed}")



def cuda_time_ms(fn):
    """Mean kernel time in ms via CUDA events."""
    if DEVICE == "cuda":
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        s.record()
        for _ in range(args.iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / args.iters   # elapsed_time returns ms
    else:
        import time
        for _ in range(args.warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            fn()
        return (time.perf_counter() - t0) / args.iters * 1e3


def param_mb(module):
    """Total parameter memory of an nn.Module in MB."""
    return sum(p.element_size() * p.nelement() for p in module.parameters()) / 1e6


def tensor_mb(t):
    return t.element_size() * t.nelement() / 1e6


def rel_err(a, b, dtype):
    """Max relative error"""
    a = a.to(dtype)
    b = b.to(dtype)
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-12)).item()


def get_err(key, tensor, saved, dtype=torch.float64):
    if not saved or key not in saved:
        return None
    return rel_err(tensor, saved[key].to(DEVICE), dtype)


def fmt_err(e):
    if e is None:
        return "—"
    tag = "OK" if e < 1e-5 else ("WARN" if e < 1e-3 else "FAIL")
    return f"{e:.3e} ({tag})"


def save_checkpoint(path, payload):
    meta = {
        "ozaki": OZAKI,
        "moduli":   os.environ.get(_MOD_KEY, "n/a"),
        "fastmode": os.environ.get(_FASTMODE_KEY, "n/a"),
        "B": B, "P": P, "in_ch": in_ch, "out_ch": out_ch,
        "dtype": args.dtype, "seed": args.seed,
    }
    torch.save({"meta": meta, "tensors": payload}, path)
    print(f"[saved] {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    meta = ckpt["meta"]
    print(f"[loaded] {path}  (saved: ozaki={meta['ozaki']} "
          f"moduli={meta['moduli']} fastmode={meta['fastmode']})")
    for key in ("B", "P", "in_ch", "out_ch", "dtype", "seed"):
        cur = {"B":B,"P":P,"in_ch":in_ch,"out_ch":out_ch,
               "dtype":args.dtype,"seed":args.seed}[key]
        if meta.get(key) != cur:
            print(f"  WARNING: {key} mismatch — saved={meta[key]} current={cur}")
    return ckpt["tensors"]



torch.manual_seed(args.seed)
x = torch.rand(B, in_ch, P, device=DEVICE, dtype=data_type)

saved_tensors      = load_checkpoint(args.load) if args.load else {}
checkpoint_payload = {}
rows               = []


# =============================================================================
# nn.Linear:  [B, in_ch, out_ch]
# nn.Conv1d:  [B, in_ch, out_ch] 
#   projection  = nn.Linear(in_ch, out_ch)   applied channel-wise
#   spectral    = nn.Conv1d(out_ch, out_ch, kernel_size=1)
#   inv_proj    = nn.Linear(out_ch,in_ch)
# =============================================================================

torch.manual_seed(args.seed)

proj1 = nn.Linear(in_ch, out_ch, bias=True).to(DEVICE, dtype=data_type)
conv1 = nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=True).to(DEVICE, dtype=data_type)
inv1  = nn.Linear(out_ch, in_ch, bias=True).to(DEVICE, dtype=data_type)

x1      = x.permute(0, 2, 1)               # [B, P, in_ch]
fwd1    = proj1(x1)                        # [B, P, out_ch]
fwd1_c  = fwd1.permute(0, 2, 1)            # [B, out_ch, P]
gemm1   = conv1(fwd1_c)                    # [B, out_ch, P]
gemm1_p = gemm1.permute(0, 2, 1)           # [B, P, out_ch]
out1    = inv1(gemm1_p)                    # [B, P, in_ch]

t_proj1 = cuda_time_ms(lambda: proj1(x1))
t_conv1 = cuda_time_ms(lambda: conv1(fwd1_c))
t_inv1  = cuda_time_ms(lambda: inv1(gemm1_p))

checkpoint_payload.update({
    "1d_proj": fwd1.cpu(), "1d_conv": gemm1.cpu(), "1d_inv": out1.cpu()
})
rows.append({
    "proj (ms)":      round(t_proj1, 3),
    "conv (ms)":      round(t_conv1, 3),
    "inv (ms)":       round(t_inv1,  3),
    "total (ms)":     round(t_proj1 + t_conv1 + t_inv1, 3),
    "proj params MB": round(param_mb(proj1), 3),
    "conv params MB": round(param_mb(conv1), 3),
    "inv params MB":  round(param_mb(inv1),  3),
    "x MB":           round(tensor_mb(x), 1),
    "rel-err proj":   fmt_err(get_err("1d_proj", fwd1,  saved_tensors, dtype)),
    "rel-err conv":   fmt_err(get_err("1d_conv", gemm1, saved_tensors, dtype)),
    "rel-err inv":    fmt_err(get_err("1d_inv",  out1,  saved_tensors, dtype)),
})


df = pd.DataFrame(rows)

if not saved_tensors:
    df = df.drop(columns=["rel-err proj", "rel-err conv", "rel-err inv"])

tag = (f"GEMMul8 ({_MOD_KEY}={os.environ.get(_MOD_KEY,'?')})"
       if OZAKI else "baseline (no GEMMul8)")

print(f"\n{'─' * 80}")
print(f"  Results — {tag}")
print(f"{'─' * 80}")
print(df.to_string())
print(f"{'─' * 80}\n")

if args.save:
    save_checkpoint(args.save, checkpoint_payload)