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

    def __init__(self, register):
        self.register = register
        #: offset currently held, or None while the register is unused
        self.held = None

    def fits(self, offset, limit, granularity):
        return offset <= limit and offset % granularity == 0

    def place(self, asm, addr, limit, granularity):
        """Rewrite an address so the instruction can encode it.

        Adds to the scratch register where the offset is out of reach, and
        rewrites the address to be relative to it.
        """

        if self.held is None:
            offset = addr.disp
        else:
            offset = addr.disp - self.held
            addr.disp = offset
            addr.base = self.register

        if self.fits(offset, limit, granularity):
            return

        if self.held is None:
            asm.add(add(offset, self.register, "", addr.base))
            self.held = offset
        else:
            asm.add(add(offset, self.register, ""))
            self.held += offset
        addr.disp = 0
        addr.base = self.register
