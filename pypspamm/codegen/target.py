"""What a code generator's target offers, stated once per architecture.

These are the facts that the generators, the block size search and the
register assignment all need and that used to be spelled out separately in
each of them: how large the vector register file is, whether there are masks,
and which register numbers an instruction may reach.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Tuple


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
    #: whether the generator can lay out room for the operands of more than
    #: one iteration, as a rotated loop needs
    operand_copies_supported: bool = False
    #: largest byte offset a memory instruction can encode, and the granularity
    #: that offset has to be a multiple of, by the number of registers the
    #: instruction transfers at once
    memory_offsets: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    #: for a target that counts the offset in whole vector registers rather
    #: than in bytes, the largest such multiple. Zero says the instructions
    #: take no immediate offset at all.
    vector_offset_steps: Optional[int] = None
    #: whether the original base is used whenever it reaches, rather than
    #: staying on the scratch register once that is in use
    prefer_original_base: bool = False
    #: range of the immediate an addition can carry, which decides whether a
    #: scratch base can be advanced by a difference or has to be recomputed
    #: from the original base
    scalar_immediate: Optional[Tuple[int, int]] = None

    @property
    def has_masks(self) -> bool:
        return self.mask_registers > 0

    def indexed_limit(self, scalar_bytes: int) -> int:
        """How many registers the indexed operand of an FMA may come from."""

        return self.indexed_operand_registers.get(scalar_bytes, self.vector_registers)

    def memory_offset_limit(self, fused: int = 1, vector_bytes: int = 0):
        """Largest byte offset of a memory instruction, and its granularity.

        None where the encoding puts no useful limit on the offset, which is
        the case for a target with a wide displacement field.
        """

        if self.vector_offset_steps is not None:
            return self.vector_offset_steps * vector_bytes, max(vector_bytes, 1)
        if fused in self.memory_offsets:
            return self.memory_offsets[fused]
        return None

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
        self,
        bn: int,
        bk: int,
        vm: int,
        scalar_bytes: Optional[int] = None,
        copies: int = 1,
    ) -> bool:
        """Whether a block of this size has room for its accumulators and operands.

        A rotated loop holds the operands of two iterations at once, which is
        what copies counts. The accumulators are live across the whole loop
        either way and are counted once.
        """

        operands = copies * (bk * vm + self.broadcast_registers(bn, bk, scalar_bytes))
        if bn * vm + operands > self.vector_registers:
            return False
        if self.broadcast is BroadcastForm.SCALAR:
            return (
                copies * bn * bk + self.reserved_scalar_registers
                <= self.scalar_registers
            )
        return True


TARGETS = {
    # 16 ymm registers; alpha and beta are broadcast explicitly
    "hsw": TargetDescription(vector_registers=16, operand_copies_supported=True),
    # 32 zmm registers, k1 to k7 as masks; the multiplication takes its
    # broadcast operand straight from memory, so no explicit broadcast
    "knl": TargetDescription(
        vector_registers=32,
        mask_registers=7,
        scalar_broadcast=False,
        broadcast=BroadcastForm.MEMORY,
        operand_copies_supported=True,
    ),
    # 32 v registers, no masks
    "arm": TargetDescription(
        vector_registers=32,
        broadcast=BroadcastForm.INDEXED,
        # one register reaches 65520 bytes, a pair 1008, and the three and four
        # register forms only the fixed step they advance by
        memory_offsets={1: (65520, 16), 2: (1008, 16), 3: (48, 24), 4: (64, 32)},
        prefer_original_base=True,
    ),
    # 32 z registers, p0 to p7 as predicates. The indexed form of FMLA takes
    # its second operand from z0 to z15 for doubles and from z0 to z7 for
    # anything narrower.
    "arm_sve": TargetDescription(
        vector_registers=32,
        mask_registers=8,
        indexed_operand_registers={8: 16, 4: 8, 2: 8},
        broadcast=BroadcastForm.INDEXED,
        # ld1d and st1d encode the offset as a multiple of the vector length
        vector_offset_steps=7,
        prefer_original_base=True,
    ),
    # 32 v registers; the multiplication takes a scalar operand directly
    "rvv": TargetDescription(
        vector_registers=32,
        scalar_broadcast=False,
        broadcast=BroadcastForm.SCALAR,
        scalar_registers=32,
        reserved_scalar_registers=2,
        # the vector loads take no immediate offset
        vector_offset_steps=0,
        # addi carries a signed twelve bit immediate
        scalar_immediate=(-2048, 2047),
    ),
    # 32 v registers, no masks
    "lsx": TargetDescription(
        vector_registers=32,
        # a twelve bit signed immediate, on the memory instructions and on the
        # addition alike
        memory_offsets={1: (2047, 1)},
        scalar_immediate=(-2048, 2047),
        prefer_original_base=True,
    ),
}
