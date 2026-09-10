"""Walking the cells a microkernel block actually has to touch.

Every generator loops over the same three indices and skips the cells whose
operands are zero. What differs is the order the indices are nested in and
whether the sparsity of A is decided here or further in.
"""

from pypspamm.cursors import Coords


def cells(
    A,
    B,
    A_ptr,
    B_ptr,
    to_A_block,
    to_B_block,
    Vm,
    bn,
    bk,
    v_size,
    order="mnk",
    require_a=True,
):
    """Yield the cells of a block that carry a contribution.

    Each is a tuple of the three indices and the coordinates of the cell in A
    and in B. order names the nesting, outermost index first, with m standing
    for the vector row, n for the column of the block and k for the depth.
    """

    ranges = {"m": range(Vm), "n": range(bn), "k": range(bk)}
    outer, middle, inner = (ranges[name] for name in order)

    for first in outer:
        for second in middle:
            for third in inner:
                index = dict(zip(order, (first, second, third)))
                Vmi, bni, bki = index["m"], index["n"], index["k"]
                to_bcell = Coords(down=bki, right=bni)
                to_acell = Coords(down=Vmi * v_size, right=bki)
                if not B.has_nonzero_cell(B_ptr, to_B_block, to_bcell):
                    continue
                if require_a and not A.has_nonzero_cell(A_ptr, to_A_block, to_acell):
                    continue
                yield Vmi, bni, bki, to_acell, to_bcell
