from dataclasses import dataclass
from enum import Enum


class Precision(Enum):
    """A scalar type, with the letter it goes by on the command line.

    The letter is part of the value so that half and bfloat16 stay distinct
    members: they agree on size and on the C type they are passed as, and an
    enum folds members with equal values into one.
    """

    DOUBLE = ("d", "double", 8)
    SINGLE = ("s", "float", 4)
    HALF = ("h", "uint16_t", 2)
    BFLOAT16 = ("bf16", "uint16_t", 2)

    def __init__(self, code: str, ctype: str, size: int):
        self.code = code
        self._ctype = ctype
        self._size = size

    @classmethod
    def fromCode(cls, code: str) -> "Precision":
        code = code.lower()
        for precision in cls:
            if precision.code == code:
                return precision
        raise ValueError(
            f"{code}: unknown precision, expected one of "
            f"{', '.join(p.code for p in cls)}"
        )

    @classmethod
    def getCType(cls, precision: "Precision") -> str:
        return precision._ctype

    def ctype(self) -> str:
        return self._ctype

    def size(self) -> int:
        return self._size

    def __repr__(self):
        return self._ctype

    def __str__(self):
        return self._ctype


@dataclass(frozen=True)
class Types:
    """The scalar type of each operand of a multiplication.

    They are all the same today. Keeping them apart is what a target with a
    separate accumulator needs: the matrix engines take a narrow input type and
    accumulate in a wider one, and the emulated paths need to say which of the
    two a value is in.
    """

    a: Precision
    b: Precision
    c: Precision
    #: the type of alpha and beta
    scalar: Precision
    #: the type intermediate sums are kept in
    accumulator: Precision

    @classmethod
    def uniform(cls, precision: Precision) -> "Types":
        accumulator = precision
        if precision in (Precision.HALF, Precision.BFLOAT16):
            # no hardware sums these in their own type
            accumulator = Precision.SINGLE
        return cls(precision, precision, precision, precision, accumulator)

    @property
    def mixed(self) -> bool:
        return len({self.a, self.b, self.c, self.accumulator}) > 1
