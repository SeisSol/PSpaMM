#!/usr/bin/env python3
"""Report why register assignment runs out, per kernel and per pool.

Replays the linear scan that assigns registers and, where it runs out, reports
the block shape, whether the k loop is unrolled, and the live ranges of the
values holding the pool at that moment: how far apart their first and last use
are, and how often they are used. Short ranges mean too many values live in a
small window; long ranges mean values held across stretches where they are not
needed.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernelstats as ks

from pypspamm.codegen.sugar import block
from pypspamm.codegen.virtual import VirtualRegister, usagePass
from pypspamm.matmul import MatMul


def build_tree(mm):
    A_ptr, B_ptr, C_ptr = mm.A.start(), mm.B.start(), mm.C.start()
    asm = block("kernel")
    mm.blockloop(asm, A_ptr, B_ptr, C_ptr, None)
    asm = mm.optimize(asm)
    usagePass(asm)
    return asm


def scan(pool, asm):
    """Replay the linear scan and report where and why it runs out."""
    order = list(asm.flatten())
    index = {id(i): n for n, i in enumerate(order)}
    unlive, held = list(pool.registers), []
    peak = 0
    for instr in order:
        mine = sorted(
            (
                v
                for v in instr.regs()
                if isinstance(v, VirtualRegister) and v.pool is pool
            ),
            key=lambda v: v.serial,
        )
        for v in mine:
            if v.lastUsage() is instr and v in held:
                unlive.append(v.register)
                held.remove(v)
        for v in mine:
            if v.firstUsage() is instr:
                if not unlive:
                    return (
                        peak,
                        [
                            (
                                index.get(id(x.firstUsage()), -1),
                                index.get(id(x.lastUsage()), -1),
                                len(x.usage),
                            )
                            for x in held
                        ],
                        len(order),
                    )
                v.register = unlive.pop(0)
                held.append(v)
                peak = max(peak, len(held))
                if v.lastUsage() is instr:
                    unlive.append(v.register)
                    held.remove(v)
    return peak, None, len(order)


rows = []
for arch in ["hsw256", "knl512", "arm128", "arm_sve512", "rvv512", "lsx128"]:
    for prec in ("s", "d"):
        for shape in ks.SHAPES:
            name, m, n, k, sp, dens = shape
            with tempfile.TemporaryDirectory() as d:
                lda, ldb, amtx, bmtx = m, k, "", ""
                if sp == "a":
                    lda = 0
                    amtx = d + "/a.mtx"
                    ks.write_mtx(amtx, m, k, dens, ks.seed_of(name, "a"))
                if sp == "b":
                    ldb = 0
                    bmtx = d + "/b.mtx"
                    ks.write_mtx(bmtx, k, n, dens, ks.seed_of(name, "b"))
                try:
                    mm = MatMul(
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
                        precision=prec,
                        output_funcname="p",
                        scheduling="pipeline",
                    )
                    mm.make()
                    continue
                except (AssertionError, NotImplementedError):
                    pass
                try:
                    mm = MatMul(
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
                        precision=prec,
                        output_funcname="p",
                        scheduling="pipeline",
                    )
                    asm = build_tree(mm)
                except (AssertionError, NotImplementedError):
                    continue
                for pool, label in (
                    (mm.A_pool, "A"),
                    (mm.B_pool, "B"),
                    (mm.C_pool, "C"),
                ):
                    if not pool.registers:
                        continue
                    peak, stuck, total = scan(pool, asm)
                    if stuck is not None:
                        spans = [b - a for a, b, _ in stuck if a >= 0 and b >= 0]
                        uses = [u for _, _, u in stuck]
                        rows.append(
                            (
                                f"{arch}/{prec}/{name}",
                                label,
                                len(pool.registers),
                                f"{mm.bm}x{mm.bn}x{mm.bk}",
                                mm.unroll_m or mm.unroll_n,
                                total,
                                max(spans) if spans else 0,
                                sum(spans) // len(spans) if spans else 0,
                                min(uses) if uses else 0,
                            )
                        )
                        break

print(
    f"{'kernel':34s} {'pool':5s} {'sz':3s} {'block':10s} {'unroll':6s} {'instrs':7s} {'maxspan':8s} {'avgspan':8s} {'minuses'}"
)
for r in rows:
    print(
        f"{r[0]:34s} {r[1]:5s} {r[2]:<3d} {r[3]:10s} {str(r[4]):6s} {r[5]:<7d} {r[6]:<8d} {r[7]:<8d} {r[8]}"
    )
print("\nfailing kernels:", len(rows))
