"""Reaching an address a memory instruction cannot encode on its own.

Memory instructions carry only a small immediate offset, and how far it reaches
is a property of the target. Where a wanted address lies out of reach, the
difference is added into a scratch register once and the accesses that follow
are made relative to that register instead.
"""

from pypspamm.codegen.sugar import add


class ScratchBase:
    """A register standing in for a base address that is out of reach.

    Holds the offset the register currently carries relative to the original
    base, so that a run of nearby accesses shares one addition.
    """

    def __init__(
        self,
        register,
        scale: int = 1,
        immediate=None,
        prefer_original: bool = False,
    ):
        self.register = register
        #: what an encoded displacement counts, in bytes
        self.scale = scale
        #: range of the immediate an addition carries, or None where it is wide
        #: enough not to matter
        self.immediate = immediate
        #: whether the original base is used whenever it reaches, rather than
        #: staying on the scratch register once that is in use
        self.prefer_original = prefer_original
        #: offset currently held, or None while the register is unused
        self.held = None

    @property
    def offset(self) -> int:
        return self.held or 0

    def fits(self, offset, limit, granularity):
        return 0 <= offset <= limit and offset % granularity == 0

    def reaches(self, offset) -> bool:
        """Whether an addition can carry this offset as an immediate."""

        if self.immediate is None:
            return True
        low, high = self.immediate
        return low <= offset <= high

    def place(self, asm, addr, limit, granularity, force=False, comment=""):
        """Rewrite an address so the instruction can encode it.

        Uses the base already in hand where it reaches, and otherwise brings
        the scratch register onto the address, advancing it by the difference
        where the addition carries that and recomputing it from the original
        base where it does not. Reports whether it had to do the latter.
        """

        original = addr.base
        absolute = addr.disp

        def use(base, offset):
            addr.base = base
            addr.disp = offset // self.scale

        if not force:
            if (self.prefer_original or self.held is None) and self.fits(
                absolute, limit, granularity
            ):
                use(original, absolute)
                return False
            if self.held is not None and self.fits(
                absolute - self.held, limit, granularity
            ):
                use(self.register, absolute - self.held)
                return False

        if self.held is not None and self.reaches(absolute - self.held):
            asm.add(add(absolute - self.held, self.register, comment))
        else:
            asm.add(add(absolute, self.register, comment, original))
        self.held = absolute
        use(self.register, 0)
        return True
