"""Benchmark FP32/FP64 GPU penalty for simple and transcendental-heavy kernels.

This script is a small hardware diagnostic for users deciding whether to run
CuPy pairwise solves in `complex64` or `complex128`.

It reports:
- CUDA device metadata including `singleToDoublePrecisionPerfRatio`
- runtime ratios for a simple FMA-heavy kernel (float64 vs float32)
- runtime ratios for a transcendental-heavy kernel (`sin/cos/atan2/pow`)

The transcendental-heavy ratio is often a practical upper bound for kernels
that spend substantial time in angle/phase setup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyceles._optional import import_cupy


def _build_module(n: int) -> str:
    return f"""
extern "C" __global__ void trans_f32(const float* x, const float* y, float* out, int iters) {{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= {n}) return;
  float a = x[i];
  float b = y[i];
  float acc = 0.0f;
  for (int k = 0; k < iters; ++k) {{
    float t = atan2f(b + 0.001f * k, a + 0.002f * k);
    float s = sinf(t) + cosf(t);
    acc += powf(fabsf(s) + 1.0001f, 1.5f);
    a = fmaf(a, 1.00001f, 0.0001f);
    b = fmaf(b, 0.99999f, -0.0001f);
  }}
  out[i] = acc;
}}

extern "C" __global__ void trans_f64(const double* x, const double* y, double* out, int iters) {{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= {n}) return;
  double a = x[i];
  double b = y[i];
  double acc = 0.0;
  for (int k = 0; k < iters; ++k) {{
    double t = atan2(b + 0.001 * k, a + 0.002 * k);
    double s = sin(t) + cos(t);
    acc += pow(fabs(s) + 1.0001, 1.5);
    a = fma(a, 1.00001, 0.0001);
    b = fma(b, 0.99999, -0.0001);
  }}
  out[i] = acc;
}}

extern "C" __global__ void fma_f32(const float* x, const float* y, float* out, int iters) {{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= {n}) return;
  float a = x[i];
  float b = y[i];
  float acc = 0.0f;
  for (int k = 0; k < iters; ++k) {{
    acc = fmaf(a, b, acc);
    a = fmaf(a, 1.00001f, 0.0001f);
    b = fmaf(b, 0.99999f, -0.0001f);
  }}
  out[i] = acc;
}}

extern "C" __global__ void fma_f64(const double* x, const double* y, double* out, int iters) {{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= {n}) return;
  double a = x[i];
  double b = y[i];
  double acc = 0.0;
  for (int k = 0; k < iters; ++k) {{
    acc = fma(a, b, acc);
    a = fma(a, 1.00001, 0.0001);
    b = fma(b, 0.99999, -0.0001);
  }}
  out[i] = acc;
}}
"""


def _bench_kernel(kernel, grid, block, args, *, cupy, repeats: int) -> float:
    times: list[float] = []
    for _ in range(int(repeats)):
        start = cupy.cuda.Event()
        end = cupy.cuda.Event()
        start.record()
        kernel(grid, block, args)
        end.record()
        end.synchronize()
        times.append(float(cupy.cuda.get_elapsed_time(start, end)) / 1000.0)
    return float(np.mean(times))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure FP64/FP32 runtime ratio on simple and transcendental GPU kernels."
    )
    parser.add_argument("--n", type=int, default=4_000_000)
    parser.add_argument("--iters", type=int, default=128)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    cupy, _ = import_cupy()
    n = int(args.n)
    iters = int(args.iters)
    threads = int(args.threads)
    repeats = int(args.repeats)
    blocks = (n + threads - 1) // threads

    module = cupy.RawModule(code=_build_module(n))
    trans_f32 = module.get_function("trans_f32")
    trans_f64 = module.get_function("trans_f64")
    fma_f32 = module.get_function("fma_f32")
    fma_f64 = module.get_function("fma_f64")

    rng = cupy.random.RandomState(7)
    x32 = rng.random_sample(n, dtype=cupy.float32)
    y32 = rng.random_sample(n, dtype=cupy.float32)
    out32 = cupy.empty_like(x32)
    x64 = x32.astype(cupy.float64)
    y64 = y32.astype(cupy.float64)
    out64 = cupy.empty_like(x64)

    # Warmup
    trans_f32((blocks,), (threads,), (x32, y32, out32, np.int32(iters)))
    trans_f64((blocks,), (threads,), (x64, y64, out64, np.int32(iters)))
    fma_f32((blocks,), (threads,), (x32, y32, out32, np.int32(iters)))
    fma_f64((blocks,), (threads,), (x64, y64, out64, np.int32(iters)))
    cupy.cuda.Stream.null.synchronize()

    tf32 = _bench_kernel(
        trans_f32,
        (blocks,),
        (threads,),
        (x32, y32, out32, np.int32(iters)),
        cupy=cupy,
        repeats=repeats,
    )
    tf64 = _bench_kernel(
        trans_f64,
        (blocks,),
        (threads,),
        (x64, y64, out64, np.int32(iters)),
        cupy=cupy,
        repeats=repeats,
    )
    ff32 = _bench_kernel(
        fma_f32,
        (blocks,),
        (threads,),
        (x32, y32, out32, np.int32(iters)),
        cupy=cupy,
        repeats=repeats,
    )
    ff64 = _bench_kernel(
        fma_f64,
        (blocks,),
        (threads,),
        (x64, y64, out64, np.int32(iters)),
        cupy=cupy,
        repeats=repeats,
    )

    props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
    payload = {
        "device_name": props["name"].decode("utf-8", errors="replace"),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "single_to_double_precision_perf_ratio": int(
            props.get("singleToDoublePrecisionPerfRatio", -1)
        ),
        "n": n,
        "iters": iters,
        "threads": threads,
        "repeats": repeats,
        "trans_f32_s": tf32,
        "trans_f64_s": tf64,
        "trans_f64_over_f32": tf64 / tf32 if tf32 > 0 else float("nan"),
        "fma_f32_s": ff32,
        "fma_f64_s": ff64,
        "fma_f64_over_f32": ff64 / ff32 if ff32 > 0 else float("nan"),
    }

    print(json.dumps(payload, indent=2))
    if args.out_json is not None:
        out_path = Path(args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {out_path}")


if __name__ == "__main__":
    main()
