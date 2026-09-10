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

    def __init__(self, register, scale: int = 1, eager: bool = False):
        self.register = register
        #: what an encoded displacement counts, in bytes
        self.scale = scale
        #: whether every address is rewritten to be relative to the scratch
        #: register once it is in use, or only those that need it. Rewriting
        #: eagerly loses sight of the original base, so a refresh has to add
        #: the difference; rewriting lazily can recompute from the original.
        self.eager = eager
        #: offset currently held, or None while the register is unused
        self.held = None
        #: the base an address is expressed against
        self.base = None

    @property
    def offset(self) -> int:
        return self.held or 0

    def fits(self, offset, limit, granularity):
        return offset <= limit and offset % granularity == 0

    def place(self, asm, addr, limit, granularity, force=False, comment=""):
        """Rewrite an address so the instruction can encode it.

        Adds to the scratch register where the offset is out of reach, and
        rewrites the address to be relative to it. A caller with its own reason
        to refresh the register says so with force.
        """

        rebased = self.eager and self.held is not None
        if rebased:
            addr.disp -= self.held
            addr.base = self.register
        offset = addr.disp if rebased else addr.disp - self.offset

        if force or not self.fits(offset, limit, granularity):
            if rebased:
                asm.add(add(offset, self.register, comment))
                self.held += offset
                addr.disp = 0
            else:
                asm.add(add(addr.disp, self.register, comment, addr.base))
                self.held = addr.disp
                if self.eager:
                    addr.disp = 0
            addr.base = self.register
            self.base = self.register

        if self.eager:
            return

        if self.base is None:
            self.base = addr.base
        addr.base = self.base
        addr.disp = (addr.disp - self.offset) // self.scale
