#!/usr/bin/env python3
"""Time generated kernels against each other on the machine running this.

Instruction counts say what a kernel costs to fetch, not what it costs to run;
on AVX-512 the two came apart by thirty percentage points in the first
measurement this was written for. So kernelstats guards against regressions and
this measures which of two ways of generating the same kernel is faster.

It compiles and runs the kernels, so it only works for an architecture the host
can execute. Run it on the target rather than cross-compiling: the answer for
an out-of-order core does not carry over to an in-order one.

    tools/benchmark.py --arch knl512 --compare none pipeline

Each kernel is timed in both variants, interleaved and repeated, with the
minimum taken, which is the usual way to keep a shared machine's noise out of
the result. Variants that differ only in block size can be compared the same
way by passing --block.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kernelstats import PRECISIONS, SHAPES, seed_of, write_mtx  # noqa: E402

from pypspamm.codegen.ccode import make_cfunc  # noqa: E402
from pypspamm.matmul import MatMul  # noqa: E402

# architectures the host can run, with the flags their kernels need
NATIVE_FLAGS = {
    "hsw": ["-mavx2", "-mfma"],
    "knl": ["-mavx512f", "-mavx512dq"],
}

HARNESS = r"""
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

long long pspamm_num_total_flops = 0;

#include "kernels.h"

template <typename T, typename F>
static double timeit(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB,
                     F f, int reps, int trials) {
  T *A, *B, *C;
  if (posix_memalign((void**)&A, 64, sizeof(T) * LDA * K) ||
      posix_memalign((void**)&B, 64, sizeof(T) * LDB * N) ||
      posix_memalign((void**)&C, 64, sizeof(T) * M * N)) {
    return -1.0;
  }
  /* small inputs, so that accumulating over many repetitions stays in range */
  for (unsigned i = 0; i < LDA * K; ++i) A[i] = (T)rand() / RAND_MAX * 1e-4;
  for (unsigned i = 0; i < LDB * N; ++i) B[i] = (T)rand() / RAND_MAX * 1e-4;
  for (unsigned i = 0; i < M * N; ++i) C[i] = 0;

  for (int i = 0; i < reps / 10; ++i) f(A, B, C, 1.0, 1.0, nullptr);

  double best = 1e30;
  for (int trial = 0; trial < trials; ++trial) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) f(A, B, C, 1.0, 1.0, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    best = std::min(best,
                    std::chrono::duration<double, std::nano>(t1 - t0).count() / reps);
  }
  free(A); free(B); free(C);
  return best;
}

template <typename T, typename F1, typename F2>
static void run(const char* name, unsigned M, unsigned N, unsigned K,
                unsigned LDA, unsigned LDB, F1 f1, F2 f2, int reps, int trials) {
  double a = timeit<T>(M, N, K, LDA, LDB, f1, reps, trials);
  double b = timeit<T>(M, N, K, LDA, LDB, f2, reps, trials);
  /* once more, so that a machine drifting under us shows up as disagreement */
  a = std::min(a, timeit<T>(M, N, K, LDA, LDB, f1, reps, trials));
  printf("RESULT %s %.4f %.4f\n", name, a, b);
}

int main() {
#include "cases.inc"
  return 0;
}
"""


def kernel_source(tag, arch, precision, shape, variant, mtxdir):
    """Generate one kernel and report its source and its instruction count."""

    name, m, n, k, sparse, density = shape
    lda, ldb, amtx, bmtx = m, k, "", ""
    if sparse == "a":
        lda = 0
        amtx = os.path.join(mtxdir, f"{name}_a.mtx")
        if not os.path.exists(amtx):
            write_mtx(amtx, m, k, density, seed_of(name, "a"))
    if sparse == "b":
        ldb = 0
        bmtx = os.path.join(mtxdir, f"{name}_b.mtx")
        if not os.path.exists(bmtx):
            write_mtx(bmtx, k, n, density, seed_of(name, "b"))

    matmul = MatMul(
        m=m,
        n=n,
        k=k,
        lda=lda,
        ldb=ldb,
        ldc=m,
        alpha="1.0",
        beta="1.0",
        mtx_filename="",
        amtx_filename=amtx,
        bmtx_filename=bmtx,
        arch=arch,
        precision=precision,
        output_funcname=tag,
        **variant,
    )
    body = matmul.make()
    text = make_cfunc(
        tag,
        matmul.generator.get_template(),
        body,
        matmul.flop,
        matmul.starting_regs,
        matmul.generator.get_precision(),
    )
    instructions = sum(1 for i in body.flatten() if i.stmtname() != "label")
    return text, instructions, (m, n, k, lda or m, ldb or k)


def build(workdir, arch, labels, variants, reps, trials):
    sources, cases, counts = [], [], {}
    with tempfile.TemporaryDirectory() as mtxdir:
        for precision in PRECISIONS:
            ctype = "double" if precision == "d" else "float"
            for shape in SHAPES:
                tag = f"{arch}_{precision}_{shape[0]}"
                built = {}
                for label, variant in zip(labels, variants):
                    try:
                        built[label] = kernel_source(
                            f"{tag}_{label}", arch, precision, shape, variant, mtxdir
                        )
                    except Exception:  # noqa: BLE001 - a shape an arch cannot express
                        break
                if len(built) != len(labels):
                    continue
                for label in labels:
                    sources.append(built[label][0])
                counts[tag] = [built[label][1] for label in labels]
                m, n, k, lda, ldb = built[labels[0]][2]
                names = ", ".join(f"{tag}_{label}" for label in labels)
                cases.append(
                    f'  run<{ctype}>("{tag}", {m}, {n}, {k}, {lda}, {ldb}, '
                    f"{names}, {reps}, {trials});"
                )

    with open(os.path.join(workdir, "kernels.h"), "w") as f:
        f.write("\n".join(sources))
    with open(os.path.join(workdir, "cases.inc"), "w") as f:
        f.write("\n".join(cases) + "\n")
    with open(os.path.join(workdir, "harness.cpp"), "w") as f:
        f.write(HARNESS)
    return counts


def flags_for(arch):
    for prefix, flags in NATIVE_FLAGS.items():
        if arch.startswith(prefix):
            return flags
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="knl512", help="architecture to time")
    parser.add_argument(
        "--compare",
        nargs=2,
        default=["none", "pipeline"],
        help="two scheduling levels to compare (default: none pipeline)",
    )
    parser.add_argument(
        "--block",
        nargs=2,
        default=None,
        metavar="BMxBNxBK",
        help="compare two block sizes instead of two scheduling levels",
    )
    parser.add_argument("--reps", type=int, default=20000)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--json", help="also write the results here")
    args = parser.parse_args()

    flags = flags_for(args.arch)
    if flags is None:
        print(
            f"{args.arch} kernels cannot be run on this host; "
            f"run this on the target machine",
            file=sys.stderr,
        )
        return 2

    if args.block:
        labels = ["a", "b"]
        variants = []
        for spec in args.block:
            bm, bn, bk = (int(x) for x in spec.lower().split("x"))
            variants.append({"bm": bm, "bn": bn, "bk": bk})
        titles = args.block
    else:
        labels = args.compare
        variants = [{"scheduling": level} for level in args.compare]
        titles = args.compare

    with tempfile.TemporaryDirectory() as workdir:
        counts = build(workdir, args.arch, labels, variants, args.reps, args.trials)
        binary = os.path.join(workdir, "bench")
        compile_command = [args.cxx, "-O2", *flags, "harness.cpp", "-o", binary]
        result = subprocess.run(
            compile_command, cwd=workdir, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(result.stderr[-2000:], file=sys.stderr)
            return 1
        run = subprocess.run([binary], capture_output=True, text=True)
        if run.returncode != 0:
            print(run.stderr[-2000:], file=sys.stderr)
            return 1

    rows = []
    for line in run.stdout.splitlines():
        match = re.match(r"RESULT (\S+) (\S+) (\S+)", line)
        if match:
            tag, first, second = (
                match.group(1),
                float(match.group(2)),
                float(match.group(3)),
            )
            rows.append((tag, first, second, counts.get(tag, [0, 0])))

    print(
        f"{'kernel':34s} {titles[0]:>9s} {titles[1]:>9s} {'time':>8s} {'instrs':>16s}"
    )
    for tag, first, second, count in rows:
        delta = (second / first - 1) * 100 if first > 0 else 0.0
        growth = (count[1] / count[0] - 1) * 100 if count[0] else 0.0
        print(
            f"{tag:34s} {first:9.1f} {second:9.1f} {delta:+7.1f}% "
            f"{count[0]:6d} ->{count[1]:6d} ({growth:+.0f}%)"
        )
    if not rows:
        print("no kernel could be generated both ways for this architecture")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "arch": args.arch,
                    "variants": titles,
                    "unit": "nanoseconds per call",
                    "kernels": {
                        tag: {
                            "time": [first, second],
                            "instructions": count,
                        }
                        for tag, first, second, count in rows
                    },
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
