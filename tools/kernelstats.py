#!/usr/bin/env python3
"""Static code size statistics for generated PSpaMM kernels.

Counts the instructions PSpaMM emits for a fixed set of kernels, for every
supported architecture. The counts are derived from the syntax tree, not from
the printed assembly, and are therefore independent of the host and of any
toolchain: the numbers are identical on every machine and for every ISA,
including those that can only be executed under emulation.

Two uses:

* ``emit`` writes a baseline file.
* ``compare`` checks a fresh run against that baseline and reports every kernel
  whose instruction count moved. It exits non-zero if any count grew by more
  than the given tolerance, which makes it usable as a CI gate.

Loop bodies are counted as they are emitted, i.e. once per unroll step, so the
totals track the instruction cache footprint of a kernel rather than its
dynamic instruction count.
"""

import argparse
import json
import os
import random
import sys
import tempfile
import zlib
from collections import Counter

from pypspamm.matmul import MatMul

ARCHS = [
    "hsw256",
    "knl512",
    "arm128",
    "arm_sve512",
    "rvv512",
    "lsx128",
]

# (name, m, n, k, sparse operand, density)
# The shapes follow the ADER-DG operators PSpaMM is used for: a few dense
# blocks, and sparse global matrices at the sizes that occur for convergence
# orders 3 to 6.
SHAPES = [
    ("dense_small", 8, 8, 8, None, 0.0),
    ("dense_o4", 20, 9, 20, None, 0.0),
    ("dense_o6", 56, 9, 56, None, 0.0),
    ("dense_tall", 56, 3, 20, None, 0.0),
    ("dense_odd", 23, 31, 13, None, 0.0),
    ("sparse_a_o4", 20, 9, 20, "a", 0.2),
    ("sparse_a_o6", 56, 9, 56, "a", 0.1),
    ("sparse_b_o4", 20, 9, 20, "b", 0.2),
    ("sparse_b_o6", 56, 9, 56, "b", 0.1),
    ("sparse_b_narrow", 56, 9, 9, "b", 0.35),
]

PRECISIONS = ["s", "d"]


def seed_of(name, operand):
    """A seed that is stable across processes, unlike hash() on a string."""
    return zlib.crc32(f"{name}/{operand}".encode())


def write_mtx(path, rows, cols, density, seed):
    """Write a deterministic sparse pattern in MatrixMarket coordinate format."""
    rng = random.Random(seed)
    entries = []
    for j in range(1, cols + 1):
        for i in range(1, rows + 1):
            if rng.random() < density:
                entries.append((i, j, rng.uniform(0.5, 1.5)))
    if not entries:
        entries.append((1, 1, 1.0))
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{rows} {cols} {len(entries)}\n")
        for i, j, v in entries:
            f.write(f"{i} {j} {v!r}\n")


def count_statements(asm):
    """Count emitted statements by kind, as they appear in the output."""
    counts = Counter()
    for instr in asm.flatten():
        counts[instr.stmtname()] += 1
    return counts


def kernel_stats(arch, precision, shape, mtxdir, scheduling="none"):
    name, m, n, k, sparse, density = shape

    amtx = ""
    bmtx = ""
    lda, ldb, ldc = m, k, m

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
        ldc=ldc,
        alpha="1.0",
        beta="1.0",
        mtx_filename="",
        amtx_filename=amtx,
        bmtx_filename=bmtx,
        arch=arch,
        precision=precision,
        output_funcname=f"{name}_{arch}_{precision}",
        scheduling=scheduling,
    )

    counts = count_statements(matmul.make())
    total = sum(counts.values()) - counts.get("label", 0)

    return {
        "arch": arch,
        "precision": precision,
        "kernel": name,
        "block": [matmul.bm, matmul.bn, matmul.bk],
        "instructions": total,
        "by_kind": dict(sorted(counts.items())),
    }


def collect(archs, scheduling="none", verbose=False):
    results = []
    with tempfile.TemporaryDirectory() as mtxdir:
        for arch in archs:
            for precision in PRECISIONS:
                for shape in SHAPES:
                    try:
                        entry = kernel_stats(arch, precision, shape, mtxdir, scheduling)
                    except Exception as e:  # noqa: BLE001 - report and continue
                        entry = {
                            "arch": arch,
                            "precision": precision,
                            "kernel": shape[0],
                            "error": f"{type(e).__name__}: {e}",
                        }
                    if verbose:
                        print(
                            f"{entry['arch']:12s} {entry['precision']} "
                            f"{entry['kernel']:18s} "
                            f"{entry.get('instructions', entry.get('error'))}"
                        )
                    results.append(entry)
    return results


def key_of(entry):
    return f"{entry['arch']}/{entry['precision']}/{entry['kernel']}"


def cmd_emit(args):
    results = collect(args.arch, args.scheduling, verbose=args.verbose)
    payload = {"version": 1, "kernels": {key_of(e): e for e in results}}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"wrote {len(results)} entries to {args.output}")
    return 0


def cmd_compare(args):
    with open(args.baseline) as f:
        baseline = json.load(f)["kernels"]

    results = {key_of(e): e for e in collect(args.arch, args.scheduling)}

    regressions = []
    improvements = []
    missing = []
    added = []

    for key, entry in sorted(results.items()):
        if key not in baseline:
            added.append(key)
            continue
        old = baseline[key].get("instructions")
        new = entry.get("instructions")
        if old is None or new is None:
            if old != new:
                regressions.append((key, old, new))
            continue
        if new > old * (1.0 + args.tolerance):
            regressions.append((key, old, new))
        elif new < old:
            improvements.append((key, old, new))

    for key in sorted(baseline):
        if key not in results:
            missing.append(key)

    def report(title, items):
        if not items:
            return
        print(f"\n{title}:")
        for key, old, new in items:
            delta = "n/a" if old is None or new is None else f"{new - old:+d}"
            print(f"  {key:44s} {old} -> {new} ({delta})")

    report("regressions", regressions)
    report("improvements", improvements)
    if added:
        print("\nnot in baseline:\n  " + "\n  ".join(added))
    if missing:
        print("\nmissing from this run:\n  " + "\n  ".join(missing))
    if not (regressions or improvements or added or missing):
        print(f"unchanged: {len(results)} kernels")

    return 1 if regressions else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch",
        action="append",
        choices=ARCHS,
        help="restrict to one architecture (repeatable, default: all)",
    )
    parser.add_argument(
        "--scheduling",
        choices=["none", "peephole", "pipeline"],
        default="none",
        help="scheduling level to generate with (default: none)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    emit = sub.add_parser("emit", help="write a baseline file")
    emit.add_argument(
        "-o", "--output", default="-", help="output file, or - for stdout"
    )
    emit.add_argument("-v", "--verbose", action="store_true")
    emit.set_defaults(func=cmd_emit)

    compare = sub.add_parser("compare", help="check against a baseline file")
    compare.add_argument("baseline")
    compare.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="relative growth accepted without failing (default: 0)",
    )
    compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    if not args.arch:
        args.arch = ARCHS
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
