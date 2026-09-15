"""
Test: VandermondeTransformMatrixFree + Spectral GEMM
      with and without Ozaki Scheme II (GEMMul8 INT8 hook)

Run without GEMMul8 (saves baseline tensors):
    python test_ozaki_hook.py --save baseline.pt

Run with GEMMul8 INT8 hook (compares against saved baseline):
    LD_PRELOAD=/path/to/GEMMul8/lib/libgemmul8.so  \
    GEMMUL8_BACKEND_GEMM=INT8                        \
    GEMMUL8_NUM_MOD_Z_GEMM=13                        \
    GEMMUL8_FASTMODE_Z_GEMM=1                        \
    python test_ozaki_hook.py --load baseline.pt

"""

import os
import argparse
import pandas as pd
import torch
from operator_learning.data.transforms.vandermonde_matrix_free import (
    VandermondeTransformMatrixFree,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark VandermondeTransformMatrixFree + spectral GEMM "
                    "with GEMMul8 Ozaki Scheme II hook.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- problem dimensions ---
    g = parser.add_argument_group("Problem dimensions")
    g.add_argument("--B",   type=int, default=1,
                   help="Batch size")
    g.add_argument("--dv",  type=int, default=32,
                   help="Channel width (model hidden dim)")
    g.add_argument("--P",   type=int, default=500_000,
                   help="Number of particles")
    g.add_argument("--kX",  type=int, default=16,
                   help="Fourier modes in X  (weight shape has 2*kX modes)")
    g.add_argument("--kY",  type=int, default=None,
                   help="Fourier modes in Y  [default: same as --kX]")
    g.add_argument("--kZ",  type=int, default=None,
                   help="Fourier modes in Z  [default: same as --kX]")
    g.add_argument("--dim", type=int, default=None, choices=[1, 2, 3],
                   help="Spatial dimension to test. Default: run all three")

    # --- precision ---
    g2 = parser.add_argument_group("Precision")
    g2.add_argument("--dtype", type=str, default="float64",
                    choices=["float32", "float64"],
                    help="Real dtype for positions and field data")

    # --- timing ---
    g3 = parser.add_argument_group("Timing")
    g3.add_argument("--warmup", type=int, default=5,
                    help="CUDA event warmup iterations")
    g3.add_argument("--iters",  type=int, default=10,
                    help="CUDA event measurement iterations")

    # --- checkpoint ---
    g4 = parser.add_argument_group("Checkpoint")
    g4.add_argument("--save", type=str, default=None, metavar="FILE",
                    help="Save forward/GEMM/inverse outputs to FILE (e.g. baseline.pt). "
                         "Use this on the baseline (no GEMMul8) run.")
    g4.add_argument("--load", type=str, default=None, metavar="FILE",
                    help="Load a previously saved FILE and compute relative error "
                         "against the current run. Use this on the GEMMul8 run.")
    g4.add_argument("--seed", type=int, default=42,
                    help="RNG seed — must match between --save and --load runs")

    args = parser.parse_args()
    if args.kY is None:
        args.kY = args.kX
    if args.kZ is None:
        args.kZ = args.kX
    return args

args = parse_args()

if args.dtype == "float64":
    data_type  = torch.float64
    cdata_type = torch.cdouble
    _MOD_KEY      = "GEMMUL8_NUM_MOD_Z_GEMM"
    _FASTMODE_KEY = "GEMMUL8_FASTMODE_Z_GEMM"
else:
    data_type  = torch.float32
    cdata_type = torch.cfloat
    _MOD_KEY      = "GEMMUL8_NUM_MOD_C_GEMM"
    _FASTMODE_KEY = "GEMMUL8_FASTMODE_C_GEMM"

B    = args.B
dv   = args.dv
P    = args.P
kX   = args.kX
kY   = args.kY
kZ   = args.kZ
DIMS = [args.dim] if args.dim is not None else [1, 2, 3]

# ── device / GEMMul8 ──────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
maps   = open("/proc/self/maps").read()
OZAKI  = "libgemmul8" in maps

print(f"\nDevice  : {DEVICE}" + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
print(f"GEMMul8 : {'ACTIVE  moduli=' + os.environ.get(_MOD_KEY,'?') + '  fastmode=' + os.environ.get(_FASTMODE_KEY,'?') if OZAKI else 'OFF'}")
print(f"Config  : B={B}  dv={dv}  P={P:,}  kX={kX}  kY={kY}  kZ={kZ}  dtype={args.dtype}  seed={args.seed}")


def make_positions(B, P, device, dtype=torch.float32):
    return torch.rand(B, P, dtype=dtype, device=device)

def cuda_time_us(fn):
    """Mean kernel time in ms via CUDA events; falls back to Python time on CPU."""
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
        return s.elapsed_time(e) / args.iters 
    else:
        import time
        for _ in range(args.warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            fn()
        return (time.perf_counter() - t0) / args.iters * 1e6

def calc_mem(t):
    return t.element_size() * t.nelement() / 1e6

def rel_err(a, b, dtype, cdtype):
    """Max relative error"""
    a = a.to(cdtype) if a.is_complex() else a.to(dtype)
    b = b.to(cdtype) if b.is_complex() else b.to(dtype)
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-12)).item()

def section(title):
    print(f"\n{'─'*64}\n{title}\n{'─'*64}")

def save_checkpoint(path, payload: dict):
    """Save tensors + metadata to a .pt file."""
    meta = {
        "ozaki": OZAKI,
        "moduli": os.environ.get(_MOD_KEY, "n/a"),
        "fastmode": os.environ.get(_FASTMODE_KEY, "n/a"),
        "B": B, "dv": dv, "P": P, "kX": kX, "kY": kY, "kZ": kZ,
        "dtype": args.dtype, "seed": args.seed, "dims": DIMS,
    }
    torch.save({"meta": meta, "tensors": payload}, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"\n[saved] {path}  ({size_mb:.1f} MB)")

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    meta = ckpt["meta"]
    print(f"[loaded] {path}")
    print(f"saved with: ozaki={meta['ozaki']}  moduli={meta['moduli']}  "
          f"fastmode={meta['fastmode']}")
    # warn if config differs
    for key in ("B", "dv", "P", "kX", "kY", "kZ", "dtype", "seed", "dims"):
        cur = {"B":B,"dv":dv,"P":P,"kX":kX,"kY":kY,"kZ":kZ,
               "dtype":args.dtype,"seed":args.seed,"dims":DIMS}[key]
        if meta.get(key) != cur:
            print(f"WARNING: {key} mismatch — saved={meta[key]}  current={cur}")
    return ckpt["tensors"]

def get_err(key, tensor, saved, dtype=torch.float64, cdtype=torch.cdouble):
    if not saved or key not in saved:
        return None
    return rel_err(tensor, saved[key].to(DEVICE), dtype, cdtype)

def fmt_err(e):
    if e is None:
        return "—"
    tag = "OK" if e < 1e-5 else ("WARN" if e < 1e-3 else "FAIL")
    return f"{e:.3e} ({tag})"


# ── shared input (same seed → same x, positions, R for both runs) ─────────────
torch.manual_seed(args.seed)
x = torch.rand(B, dv, P, device=DEVICE, dtype=cdata_type)

# checkpoint accumulator
checkpoint_payload = {}
saved_tensors      = load_checkpoint(args.load) if args.load else {}

rows = []


# =============================================================================
# 1-D
# =============================================================================
if 1 in DIMS:
    section("1D  —  forward [B,dv,P]→[B,dv,2*kX]  |  spectral GEMM  |  inverse")

    torch.manual_seed(args.seed)   # reset so positions/R are identical across runs
    xpos = make_positions(B, P, DEVICE, dtype=data_type)
    mf1  = VandermondeTransformMatrixFree(
        x_positions=xpos, kX=kX, dim=1, device=DEVICE, dtype=data_type
    )
    fwd1 = mf1.forward(x)
    assert fwd1.shape == (B, dv, 2 * kX), \
        f"forward shape: got {tuple(fwd1.shape)}, expected {(B, dv, 2*kX)}"
    print(f"forward shape : {tuple(fwd1.shape)}")

    torch.manual_seed(args.seed + 1)
    R1 = torch.rand(dv, dv, 2 * kX, dtype=cdata_type, device=DEVICE)

    def spectral_mul1():
        return torch.einsum("bik,iok->bok", fwd1, R1)

    ref1 = spectral_mul1()
    assert ref1.shape == (B, dv, 2 * kX)
    print(f"spectral_mul shape: {tuple(ref1.shape)}")

    inv1 = mf1.inverse(fwd1)
    assert inv1.shape == (B, dv, P)
    print(f"inverse shape : {tuple(inv1.shape)}")

    t_fwd1 = cuda_time_us(lambda: mf1.forward(x))
    t_mul1 = cuda_time_us(spectral_mul1)
    t_inv1 = cuda_time_us(lambda: mf1.inverse(fwd1))

    print(f"\nMemory")
    print(f"  Input x         : {tuple(x.shape)}  {calc_mem(x):.1f} MB")
    print(f"  Fx              : {calc_mem(mf1.Fx):.1f} MB")
    print(f"  R1              : {calc_mem(R1):.1f} MB")

    print(f"\nTiming  ({'GEMMul8 ON' if OZAKI else 'baseline'})")
    print(f"  forward (NUFFT) : {t_fwd1:10.1f} ms   [einsum bcp,bkp->bck]")
    print(f"  spectral GEMM   : {t_mul1:10.1f} ms   [einsum bik,iok->bok]")
    print(f"  inverse (NUFFT) : {t_inv1:10.1f} ms   [einsum bck,bkp->bcp]")
    print(f"  total           : {t_fwd1+t_mul1+t_inv1:10.1f} ms")

    rows.append({
        "Dim":              "1D",
        "fwd NUFFT (ms)":  round(t_fwd1, 1),
        "GEMM (ms)":       round(t_mul1, 1),
        "inv NUFFT (ms)":  round(t_inv1, 1),
        "total (ms)":      round(t_fwd1 + t_mul1 + t_inv1, 1),
        "Fx/y/z (MB)":     round(calc_mem(mf1.Fx), 1),
        "R (MB)":          round(calc_mem(R1), 3),
        "rel-err fwd":     fmt_err(get_err("1d_fwd",  fwd1, saved_tensors, data_type, cdata_type)),
        "rel-err GEMM":    fmt_err(get_err("1d_gemm", ref1, saved_tensors, data_type, cdata_type)),
        "rel-err inv":     fmt_err(get_err("1d_inv",  inv1, saved_tensors, data_type, cdata_type))
    })

    # save / compare
    checkpoint_payload.update({
        "1d_fwd": fwd1.cpu(), "1d_gemm": ref1.cpu(), "1d_inv": inv1.cpu()
    })



# =============================================================================
# 2-D
# =============================================================================
if 2 in DIMS:
    section("2D  —  forward [B,dv,P]→[B,dv,(2*kX)*(2*kY)]  |  spectral GEMM  |  inverse")

    torch.manual_seed(args.seed + 10)
    xpos2 = make_positions(B, P, DEVICE, dtype=data_type)
    ypos2 = make_positions(B, P, DEVICE, dtype=data_type)
    mf2   = VandermondeTransformMatrixFree(
        x_positions=xpos2, kX=kX,
        y_positions=ypos2, kY=kY,
        dim=2, device=DEVICE, dtype=data_type,
    )

    fwd2 = mf2.forward(x)
    assert fwd2.shape == (B, dv, 4 * kX * kY), \
        f"forward shape: got {tuple(fwd2.shape)}, expected {(B, dv, 4*kX*kY)}"
    print(f"forward shape : {tuple(fwd2.shape)}")

    torch.manual_seed(args.seed + 11)
    R2      = torch.rand(dv, dv, 2*kX, 2*kY, dtype=cdata_type, device=DEVICE)
    fwd2_4d = fwd2.reshape(B, dv, 2*kX, 2*kY)

    def spectral_mul2():
        return torch.einsum("bixy,ioxy->boxy", fwd2_4d, R2)

    ref2 = spectral_mul2()
    assert ref2.shape == (B, dv, 2*kX, 2*kY)
    print(f"spectral_mul shape: {tuple(ref2.shape)}")

    inv2 = mf2.inverse(fwd2)
    assert inv2.shape == (B, dv, P)
    print(f"inverse shape : {tuple(inv2.shape)}")

    t_fwd2 = cuda_time_us(lambda: mf2.forward(x))
    t_mul2 = cuda_time_us(spectral_mul2)
    t_inv2 = cuda_time_us(lambda: mf2.inverse(fwd2))

    print(f"\nMemory")
    print(f"  Input x         : {tuple(x.shape)}  {calc_mem(x):.1f} MB")
    print(f"  Fx + Fy         : {calc_mem(mf2.Fx) + calc_mem(mf2.Fy):.1f} MB")
    print(f"  R2              : {calc_mem(R2):.1f} MB")

    print(f"\nTiming  ({'GEMMul8 ON' if OZAKI else 'baseline'})")
    print(f"  forward (NUFFT) : {t_fwd2:10.1f} ms   [einsum bcp,bkp,blp->bckl]")
    print(f"  spectral GEMM   : {t_mul2:10.1f} ms   [einsum bixy,ioxy->boxy]")
    print(f"  inverse (NUFFT) : {t_inv2:10.1f} ms   [einsum bckl,bkp,blp->bcp]")
    print(f"  total           : {t_fwd2+t_mul2+t_inv2:10.1f} ms")

    rows.append({
        "Dim":              "2D",
        "fwd NUFFT (ms)":  round(t_fwd2, 1),
        "GEMM (ms)":       round(t_mul2, 1),
        "inv NUFFT (ms)":  round(t_inv2, 1),
        "total (ms)":      round(t_fwd2 + t_mul2 + t_inv2, 1),
        "Fx/y/z (MB)":     round(calc_mem(mf2.Fx), 1),
        "R (MB)":          round(calc_mem(R2), 3),
        "rel-err fwd":     fmt_err(get_err("2d_fwd",  fwd2, saved_tensors, data_type, cdata_type)),
        "rel-err GEMM":    fmt_err(get_err("2d_gemm", ref2, saved_tensors, data_type, cdata_type)),
        "rel-err inv":     fmt_err(get_err("2d_inv",  inv2, saved_tensors, data_type, cdata_type)),
    })

    checkpoint_payload.update({
        "2d_fwd": fwd2.cpu(), "2d_gemm": ref2.cpu(), "2d_inv": inv2.cpu()
    })



# =============================================================================
# 3-D
# =============================================================================
if 3 in DIMS:
    section("3D  —  forward [B,dv,P]→[B,dv,(2*kX)*(2*kY)*(2*kZ)]  |  spectral GEMM  |  inverse")

    torch.manual_seed(args.seed + 20)
    xpos3 = make_positions(B, P, DEVICE, dtype=data_type)
    ypos3 = make_positions(B, P, DEVICE, dtype=data_type)
    zpos3 = make_positions(B, P, DEVICE, dtype=data_type)
    mf3   = VandermondeTransformMatrixFree(
        x_positions=xpos3, kX=kX,
        y_positions=ypos3, kY=kY,
        z_positions=zpos3, kZ=kZ,
        dim=3, device=DEVICE, dtype=data_type,
    )

    fwd3 = mf3.forward(x)
    assert fwd3.shape == (B, dv, 8 * kX * kY * kZ), \
        f"forward shape: got {tuple(fwd3.shape)}, expected {(B, dv, 8*kX*kY*kZ)}"
    print(f"forward shape : {tuple(fwd3.shape)}")

    torch.manual_seed(args.seed + 21)
    R3      = torch.rand(dv, dv, 2*kX, 2*kY, 2*kZ, dtype=cdata_type, device=DEVICE)
    fwd3_5d = fwd3.reshape(B, dv, 2*kX, 2*kY, 2*kZ)

    def spectral_mul3():
        return torch.einsum("bixyz,ioxyz->boxyz", fwd3_5d, R3)

    ref3 = spectral_mul3()
    assert ref3.shape == (B, dv, 2*kX, 2*kY, 2*kZ)
    print(f"spectral_mul shape: {tuple(ref3.shape)}")

    inv3 = mf3.inverse(fwd3)
    assert inv3.shape == (B, dv, P)
    print(f"inverse shape : {tuple(inv3.shape)}")

    t_fwd3 = cuda_time_us(lambda: mf3.forward(x))
    t_mul3 = cuda_time_us(spectral_mul3)
    t_inv3 = cuda_time_us(lambda: mf3.inverse(fwd3))

    print(f"\nMemory")
    print(f"  Input x         : {tuple(x.shape)}  {calc_mem(x):.1f} MB")
    print(f"  Fx + Fy + Fz    : {calc_mem(mf3.Fx) + calc_mem(mf3.Fy) + calc_mem(mf3.Fz):.1f} MB")
    print(f"  R3              : {calc_mem(R3):.1f} MB")

    print(f"\nTiming  ({'GEMMul8 ON' if OZAKI else 'baseline'})")
    print(f"  forward (NUFFT) : {t_fwd3:10.1f} ms   [einsum bcp,bkp,blp,bmp->bcklm]")
    print(f"  spectral GEMM   : {t_mul3:10.1f} ms   [einsum bixyz,ioxyz->boxyz]")
    print(f"  inverse (NUFFT) : {t_inv3:10.1f} ms   [einsum bcklm,bkp,blp,bmp->bcp]")
    print(f"  total           : {t_fwd3+t_mul3+t_inv3:10.1f} ms")

    rows.append({
        "Dim":              "3D",
        "fwd NUFFT (ms)":  round(t_fwd3, 1),
        "GEMM (ms)":       round(t_mul3, 1),
        "inv NUFFT (ms)":  round(t_inv3, 1),
        "total (ms)":      round(t_fwd3 + t_mul3 + t_inv3, 1),
        "Fx/y/z (MB)":     round(calc_mem(mf3.Fx), 1),
        "R (MB)":          round(calc_mem(R3), 3),
        "rel-err fwd":     fmt_err(get_err("3d_fwd",  fwd3, saved_tensors, data_type, cdata_type)),
        "rel-err GEMM":    fmt_err(get_err("3d_gemm", ref3, saved_tensors, data_type, cdata_type)),
        "rel-err inv":     fmt_err(get_err("3d_inv",  inv3, saved_tensors, data_type, cdata_type))
        })

    checkpoint_payload.update({
        "3d_fwd": fwd3.cpu(), "3d_gemm": ref3.cpu(), "3d_inv": inv3.cpu()
    })



# =============================================================================
# Save checkpoint (baseline run)
# =============================================================================
if args.save:
    save_checkpoint(args.save, checkpoint_payload)
print(f"Using routine for {_MOD_KEY}" )
df = pd.DataFrame(rows).set_index("Dim")
if not saved_tensors:
    df = df.drop(columns=["rel-err fwd", "rel-err GEMM", "rel-err inv"])
 
tag = (
    f"GEMMul8 (moduli={os.environ.get(_MOD_KEY, '?')})"
    if OZAKI else
    "baseline (no GEMMul8)"
)
print(f"\n{'─' * 60}")
print(f"  Results — {tag}")
print(f"{'─' * 60}")
print(df.to_string())
print(f"{'─' * 60}\n")

