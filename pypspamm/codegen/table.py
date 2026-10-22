"""Data a kernel carries itself: the values of an operand known when it is generated.

A kernel reads such an operand from a table of its own, which the C function
defines ahead of the assembly and points the operand at (see make_cfunc). The
entries are written as the bit patterns of the scalar type, so that the table
holds exactly the values given, whatever the C compiler would make of a decimal
literal, and so that index vectors can share it with the values.
"""

from typing import List, Sequence

import numpy as np

from pypspamm.codegen.precision import Precision
from pypspamm.cursors.matrix import Matrix

#: the unsigned C type of the width of each scalar type, and the suffix of its
#: literals
_WORDS = {
    8: ("unsigned long long", "ULL"),
    4: ("unsigned int", "U"),
    2: ("unsigned short", ""),
}


def bits(value: float, precision: Precision) -> int:
    """The bit pattern of a value rounded to the scalar type."""

    if precision == Precision.DOUBLE:
        return int(np.float64(value).view(np.uint64))
    if precision == Precision.SINGLE:
        return int(np.float32(value).view(np.uint32))
    if precision == Precision.HALF:
        return int(np.float16(value).view(np.uint16))
    if precision == Precision.BFLOAT16:
        # the upper half of the single precision pattern, rounded to nearest even
        word = int(np.float32(value).view(np.uint32))
        rounding = 0x7FFF + ((word >> 16) & 1)
        return ((word + rounding) >> 16) & 0xFFFF
    raise NotImplementedError(f"no bit patterns for {precision}")


def table_prelude(
    operand: str, words: Sequence[int], precision: Precision, alignment: int = 64
) -> str:
    """C code that defines a table of words of the scalar width and points the
    operand at it.

    The operand is the name of the pointer parameter the kernel reads; it keeps
    its type, and what the caller passes for it is not read any more.
    """

    ctype, suffix = _WORDS[precision.size()]
    digits = 2 * precision.size()
    name = f"pspamm_{operand}_table"
    lines = []
    for start in range(0, len(words), 4):
        chunk = words[start : start + 4]
        lines.append(
            "    " + ", ".join(f"0x{word:0{digits}x}{suffix}" for word in chunk) + ","
        )
    body = "\n".join(lines)
    return (
        f"  static const {ctype} {name}[{max(len(words), 1)}] "
        f"__attribute__((aligned({alignment}))) = {{\n{body}\n  }};\n"
        f"  {operand} = (const {precision.ctype()}*){name};\n"
    )


def vector_tiles(pattern: Matrix, rows: int, bm: int, v_size: int) -> Matrix:
    """The pattern widened to whole vector tiles.

    A blocked kernel holds a column of an operand block in vector registers,
    v_size rows each, counted from the start of every block of bm rows. Where a
    tile of a column has an entry, the kernel loads the whole tile, so the
    table has to hold all of its rows; the ones without a value hold zero.
    """

    cols = pattern.cols
    widened = np.zeros((pattern.rows, cols), dtype=bool)
    nonzero = np.asarray(pattern) != 0
    for start in range(0, rows, bm):
        end = min(start + bm, rows)
        for tile in range(start, end, v_size):
            stop = min(tile + v_size, end)
            occupied = nonzero[tile:stop, :].any(axis=0)
            widened[tile:stop, occupied] = True
    return Matrix(widened)


def values_at(offsets: Matrix, values: Matrix, length: int) -> List[float]:
    """The values in the order a cursor reads them, zero where nothing is."""

    table = [0.0] * length
    offs = np.asarray(offsets)
    vals = np.asarray(values)
    for (row, col), offset in np.ndenumerate(offs):
        if offset >= 0 and row < vals.shape[0] and col < vals.shape[1]:
            table[offset] = float(vals[row, col])
    return table
