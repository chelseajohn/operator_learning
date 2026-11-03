
"""
Test: nn.Linear and nn.Conv1d with/without cuBLAS FP64 emulation.

Baseline:
    CUBLAS_EMULATE_DOUBLE_PRECISION=0 \
    python test_cublas_emulation_RGEMM.py --save baseline_fp64.pt

Emulation enabled:
    CUBLAS_EMULATE_DOUBLE_PRECISION=1 \
    CUBLAS_EMULATION_STRATEGY=eager \
    python test_cublas_emulation_RGEMM.py \
        --dtype float64 --load baseline_fp64.pt --profile

"""

import os
import argparse
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True

if "GH200" in torch.cuda.get_device_name(0):
    # torch.nn.conv1D with cuBLAS kernel is faster 
    # than cuDNN on GH200
    torch.backends.cudnn.deterministic = False
else:
    torch.backends.cudnn.deterministic = True

# torch.backends.cuda.matmul.fp32_precision = "bfx9"


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark Linear/Conv1d with cuBLAS FP emulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Problem dimensions")
    g.add_argument("--B", type=int, default=12)
    g.add_argument("--P", type=int, default=1000_000)
    g.add_argument("--in_ch", type=int, default=2)
    g.add_argument("--out_ch", type=int, default=32)

    g = p.add_argument_group("Precision")
    g.add_argument("--dtype", choices=["float32", "float64"],
                   default="float64")

    g = p.add_argument_group("Timing")
    g.add_argument("--warmup", type=int, default=5)
    g.add_argument("--iters", type=int, default=10)

    g = p.add_argument_group("Checkpoint")
    g.add_argument("--save", default=None)
    g.add_argument("--load", default=None)
    g.add_argument("--seed", type=int, default=42)

    p.add_argument("--profile", action="store_true",
                   help="Profile one invocation of each operation")

    return p.parse_args()


args = parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64 if args.dtype == "float64" else torch.float32

B, P = args.B, args.P
in_ch, out_ch = args.in_ch, args.out_ch


# ---------------------------------------------------------------------------
# Runtime diagnostics
# ---------------------------------------------------------------------------
def loaded_cublas_libraries():
    """Return libcublas-related shared libraries visible in /proc/self/maps."""
    try:
        with open("/proc/self/maps") as f:
            lines = f.readlines()
        return sorted({
            line.split()[-1]
            for line in lines
            if "/" in line and "libcublas" in line.lower()
        })
    except OSError:
        return []


def print_runtime():
    print("\n" + "=" * 78)
    print("Runtime configuration")
    print("=" * 78)
    print("Device                  :", DEVICE)
    if DEVICE == "cuda":
        print("GPU                     :", torch.cuda.get_device_name(0))
        print("Compute capability      :", torch.cuda.get_device_capability(0))
    print("PyTorch                 :", torch.__version__)
    print("PyTorch CUDA build      :", torch.version.cuda)
    print("Dtype                   :", args.dtype)
    if args.dtype == "float64":
        print("CUBLAS_EMULATE_DOUBLE_PRECISION:",
          os.getenv("CUBLAS_EMULATE_DOUBLE_PRECISION", "<unset>"))
    else:
        print("CUBLAS_EMULATE_SINGLE_PRECISION:",
          os.getenv("CUBLAS_EMULATE_SINGLE_PRECISION", "<unset>"))
    print("CUBLAS_EMULATION_STRATEGY:",
          os.getenv("CUBLAS_EMULATION_STRATEGY", "<unset>"))
    print("CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT:",
          os.getenv("CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT",
                    "<unset>"))

    libs = loaded_cublas_libraries()
    print("Loaded cuBLAS libraries:")
    if libs:
        for lib in libs:
            print("  ", lib)
    else:
        print("   No libcublas library found in process maps yet.")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Timing and numerical utilities
# ---------------------------------------------------------------------------
def time_ms(fn):
    if DEVICE == "cuda":
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(args.iters):
            fn()
        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end) / args.iters

    for _ in range(args.warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        fn()
    return (time.perf_counter() - t0) * 1000 / args.iters


def rel_err(a, b, dtype):
    """Max relative error"""
    a = a.to(dtype)
    b = b.to(dtype)
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-12)).item()



def save_checkpoint(path, outputs):
    meta = {
        "B": B,
        "P": P,
        "in_ch": in_ch,
        "out_ch": out_ch,
        "dtype": args.dtype,
        "seed": args.seed,
        "emulate_double_precision":
            os.getenv("CUBLAS_EMULATE_DOUBLE_PRECISION", "<unset>"),
        "emulate_single_precision":
            os.getenv("CUBLAS_EMULATE_SINGLE_PRECISION", "<unset>"),
        "emulation_strategy":
            os.getenv("CUBLAS_EMULATION_STRATEGY", "<unset>"),
        "fixedpoint_mantissa_bits":
            os.getenv("CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT",
                      "<unset>"),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
    }
    torch.save({
        "meta": meta,
        "tensors": {k: v.detach().cpu() for k, v in outputs.items()},
    }, path)
    print(f"\n[saved] {path}")


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]

    print(f"\n[loaded baseline] {path}")
    if args.dtype == "float64":
        precision = meta.get("emulate_double_precision")
    else:
        precision = meta.get("emulate_single_precision")
    print("Baseline emulation setting:", precision)
    print("Baseline strategy:", meta.get("emulation_strategy"))

    current = {
        "B": B, "P": P, "in_ch": in_ch, "out_ch": out_ch,
        "dtype": args.dtype, "seed": args.seed,
    }
    for key, val in current.items():
        if meta.get(key) != val:
            print(f"WARNING: {key} mismatch: baseline={meta.get(key)} "
                  f"current={val}")

    return ckpt["tensors"]


# ---------------------------------------------------------------------------
# Optional profiler
# ---------------------------------------------------------------------------
def profile_operation(name, fn):
    if not args.profile or DEVICE != "cuda":
        return

    print(f"\n--- Profiler: {name} ---")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        result = fn()
        torch.cuda.synchronize()

    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=25,
    ))

    # Kernel/API names can help identify the execution path.
    print("Relevant operation names:")
    for event in prof.key_averages():
        key = event.key.lower()
        if any(s in key for s in ("gemm", "cublas", "conv", "linear")):
            print(" ", event.key)

    del result


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def main():
    if DEVICE == "cuda":
        torch.cuda.init()

    print_runtime()

    torch.manual_seed(args.seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Input: [B, in_ch, P]
    x = torch.rand(B, in_ch, P, device=DEVICE, dtype=DTYPE)
    x_linear = x.permute(0, 2, 1)  # [B, P, in_ch]

    # Identical seeds ensure the same initial parameters in separate runs.
    torch.manual_seed(args.seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    proj = nn.Linear(in_ch, out_ch, bias=True).to(DEVICE, dtype=DTYPE)
    conv = nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=True).to(DEVICE, dtype=DTYPE)
    inv = nn.Linear(out_ch, in_ch, bias=True).to(DEVICE, dtype=DTYPE)

    # Forward pass
    with torch.no_grad():
        fwd = proj(x_linear)              # [B, P, out_ch]
        conv_input = fwd.permute(0, 2, 1) # [B, out_ch, P]
        conv_out = conv(conv_input)       # [B, out_ch, P]
        inv_input = conv_out.permute(0, 2, 1)
        output = inv(inv_input)            # [B, P, in_ch]

    torch.cuda.synchronize() if DEVICE == "cuda" else None

    # Profile each operation separately, outside the timed region.
    profile_operation("nn.Linear projection", lambda: proj(x_linear))
    profile_operation("nn.Conv1d kernel_size=1", lambda: conv(conv_input))
    profile_operation("nn.Linear inverse", lambda: inv(inv_input))

    # Timings
    t_proj = time_ms(lambda: proj(x_linear))
    t_conv = time_ms(lambda: conv(conv_input))
    t_inv = time_ms(lambda: inv(inv_input))

    outputs = {
        "linear_projection": fwd,
        "conv1d": conv_out,
        "linear_inverse": output,
    }

    baseline = load_checkpoint(args.load) if args.load else {}
    rows = []

    for key, tensor in outputs.items():
        ref = baseline.get(key)
        if ref is None:
            relative_err = None
        else:
            ref = ref.to(DEVICE)
            relative_err = rel_err(tensor, ref, DTYPE)

        rows.append({
            "operation": key,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "time_ms": None,
            "rel_error": relative_err,
        })

    timing = {
        "linear_projection": t_proj,
        "conv1d": t_conv,
        "linear_inverse": t_inv,
    }
    for row in rows:
        row["time_ms"] = round(timing[row["operation"]], 3)

    print("\n" + "=" * 100)
    print("Results")
    print("=" * 100)
    print(pd.DataFrame(rows).to_string(index=False))
    print("-" * 100)
    print(f"Total measured time: {t_proj + t_conv + t_inv:.3f} ms")
    print("=" * 100)

    if args.save:
        save_checkpoint(args.save, outputs)


if __name__ == "__main__":
    main()
