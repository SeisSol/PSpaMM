"""What a code generator's target offers, stated once per architecture.

These are the facts that the generators, the block size search and the
register assignment all need and that used to be spelled out separately in
each of them: how large the vector register file is, whether there are masks,
and which register numbers an instruction may reach.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional


class BroadcastForm(Enum):
    """How the second operand of the multiplication reaches it."""

    #: the multiplication takes it straight from memory and costs no register
    MEMORY = auto()
    #: one vector register per value
    ELEMENT = auto()
    #: one vector register per lane group, with the multiplication naming the
    #: lane, so that several values share a register
    INDEXED = auto()
    #: a scalar register, from a file of its own
    SCALAR = auto()


@dataclass(frozen=True)
class TargetDescription:
    #: how many vector registers the kernel may use
    vector_registers: int
    #: how many mask or predicate registers exist; zero if the target has none
    mask_registers: int = 0
    #: whether alpha and beta have to be broadcast into a vector register
    #: before they can be used, rather than being an operand of the
    #: multiplication itself
    scalar_broadcast: bool = True
    #: for an instruction that indexes a lane of a register operand, the
    #: number of low registers that operand may come from, per scalar size in
    #: bytes. Empty when the target has no such instruction or does not
    #: restrict it.
    indexed_operand_registers: Dict[int, int] = field(default_factory=dict)
    #: how the second operand of the multiplication reaches it
    broadcast: BroadcastForm = BroadcastForm.ELEMENT
    #: size of the scalar register file, for a target that keeps the operand
    #: there, and how many of those registers are spoken for otherwise
    scalar_registers: int = 0
    reserved_scalar_registers: int = 0

    @property
    def has_masks(self) -> bool:
        return self.mask_registers > 0

    def indexed_limit(self, scalar_bytes: int) -> int:
        """How many registers the indexed operand of an FMA may come from."""

        return self.indexed_operand_registers.get(scalar_bytes, self.vector_registers)

    def broadcast_registers(
        self, bn: int, bk: int, scalar_bytes: Optional[int] = None
    ) -> int:
        """How many vector registers a bn by bk block of the operand occupies."""

        if self.broadcast in (BroadcastForm.MEMORY, BroadcastForm.SCALAR):
            return 0
        if self.broadcast is BroadcastForm.ELEMENT:
            return bn * bk
        assert scalar_bytes is not None, "the indexed form needs the scalar size"
        lane = 16 // scalar_bytes
        grouped = -(bk // -lane)
        if bn * grouped <= self.indexed_limit(scalar_bytes):
            return bn * grouped
        return bn * bk

    def fits(
        self, bn: int, bk: int, vm: int, scalar_bytes: Optional[int] = None
    ) -> bool:
        """Whether a block of this size has room for its accumulators and operands."""

        used = (bn + bk) * vm + self.broadcast_registers(bn, bk, scalar_bytes)
        if used > self.vector_registers:
            return False
        if self.broadcast is BroadcastForm.SCALAR:
            return bn * bk + self.reserved_scalar_registers <= self.scalar_registers
        return True


TARGETS = {
    # 16 ymm registers; alpha and beta are broadcast explicitly
    "hsw": TargetDescription(vector_registers=16),
    # 32 zmm registers, k1 to k7 as masks; the multiplication takes its
    # broadcast operand straight from memory, so no explicit broadcast
    "knl": TargetDescription(
        vector_registers=32,
        mask_registers=7,
        scalar_broadcast=False,
        broadcast=BroadcastForm.MEMORY,
    ),
    # 32 v registers, no masks
    "arm": TargetDescription(vector_registers=32, broadcast=BroadcastForm.INDEXED),
    # 32 z registers, p0 to p7 as predicates. The indexed form of FMLA takes
    # its second operand from z0 to z15 for doubles and from z0 to z7 for
    # anything narrower.
    "arm_sve": TargetDescription(
        vector_registers=32,
        mask_registers=8,
        indexed_operand_registers={8: 16, 4: 8, 2: 8},
        broadcast=BroadcastForm.INDEXED,
    ),
    # 32 v registers; the multiplication takes a scalar operand directly
    "rvv": TargetDescription(
        vector_registers=32,
        scalar_broadcast=False,
        broadcast=BroadcastForm.SCALAR,
        scalar_registers=32,
        reserved_scalar_registers=2,
    ),
    # 32 v registers, no masks
    "lsx": TargetDescription(vector_registers=32),
}
