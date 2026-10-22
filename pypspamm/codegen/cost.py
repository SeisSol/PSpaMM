"""What a kernel costs to run, estimated from its statements.

The estimate counts the instructions a kernel executes, the body of a loop as
often as the loop runs, by the unit they occupy: loads, stores, arithmetic and
permutations. Each unit takes so many instructions per cycle on the target
(Throughput, in target.py), and the busiest one bounds the time. Latency, the
front end and the caches are left out, so the estimate tells two kernels of
the same multiplication apart rather than predicting the time of either.
"""

from collections import Counter

from pypspamm.codegen.ast import (
    AddStmt,
    BcstStmt,
    Block,
    FmaStmt,
    LoadStmt,
    MovStmt,
    MulStmt,
    PermuteStmt,
    PrefetchStmt,
    StoreStmt,
)
from pypspamm.codegen.forms import Loop
from pypspamm.codegen.operands import AsmType, MemoryAddress, Register
from pypspamm.codegen.target import Throughput

SCALAR_TYPES = (AsmType.i8, AsmType.i16, AsmType.i32, AsmType.i64)


def _memory(operand) -> bool:
    return isinstance(operand, MemoryAddress)


def _pointer(operand) -> bool:
    """A scalar register where a vector instruction takes an operand: the
    address of a value it broadcasts from memory."""

    return isinstance(operand, Register) and operand.typeinfo == AsmType.i64


def _vector(operand) -> bool:
    return isinstance(operand, Register) and operand.typeinfo not in SCALAR_TYPES


def count(stmt) -> Counter:
    """How many instructions of each kind a statement executes: load, store,
    arithmetic, shuffle, and vector for any other instruction on vectors."""

    if isinstance(stmt, Loop):
        # the body as it is, not the contents, which number a label each time
        # they are taken
        inner = count(stmt.body_contents)
        return Counter(
            {kind: number * stmt.final_val for kind, number in inner.items()}
        )
    if isinstance(stmt, Block):
        total = Counter()
        for content in stmt.contents:
            total += count(content)
        return total

    counts = Counter()
    if isinstance(stmt, FmaStmt):
        counts["arithmetic"] += 1
        if _memory(stmt.bcast_src) or _pointer(stmt.mult_src):
            counts["load"] += 1
    elif isinstance(stmt, MulStmt):
        counts["arithmetic"] += 1
        if _memory(stmt.src) or _pointer(stmt.mult_src) or _memory(stmt.mult_src):
            counts["load"] += 1
    elif isinstance(stmt, AddStmt):
        if _vector(stmt.dest):
            counts["arithmetic"] += 1
    elif isinstance(stmt, PermuteStmt):
        counts["shuffle"] += 1
        if _memory(stmt.control) or _memory(stmt.table) or _memory(stmt.table2):
            counts["load"] += 1
    elif isinstance(stmt, BcstStmt):
        counts["load" if _memory(stmt.bcast_src) else "shuffle"] += 1
    elif isinstance(stmt, (MovStmt, LoadStmt, StoreStmt)):
        # a move between registers or of a constant costs none of the units
        if _memory(stmt.src):
            counts["load"] += 1
        if _memory(stmt.dest):
            counts["store"] += 1
    elif isinstance(stmt, PrefetchStmt):
        counts["load"] += 1
    return counts


def cycles(counts: Counter, throughput: Throughput) -> float:
    """The cycles the busiest unit needs for the instructions counted."""

    vector = counts["arithmetic"] + counts["shuffle"] + counts["vector"]
    return max(
        counts["load"] / throughput.loads,
        counts["store"] / throughput.stores,
        counts["arithmetic"] / throughput.arithmetic,
        counts["shuffle"] / throughput.shuffles,
        vector / throughput.vector,
    )


def estimate(stmt, throughput: Throughput) -> float:
    return cycles(count(stmt), throughput)
