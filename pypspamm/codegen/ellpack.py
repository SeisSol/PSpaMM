"""A sparse A whose values are known, laid out for kernels that gather from B.

The rows of A are cut into slices of one vector each, as the rows of C are
held in vector registers: sliced ELLPACK with the vector length as the slice
height and without sorting the rows, so that a slice is exactly what one vector
of C holds. A kernel keeps a column of B in vector registers and works through
a slice in steps. A step picks, for every lane, one entry of that lane's row,
gathers the rows of B those entries need from the registers by one permutation
and multiplies with their values, which the kernel carries in a table (see
table.py). A lane whose row has nothing left idles with the value zero.

A permutation reads from at most two registers, so a step is restricted to the
entries that lie in two registers of B. Which two, is chosen step by step: the
pair that the most lanes still have an entry in. A row then needs at least as
many steps as it has entries, and the slice as many as its longest row, as in
ELLPACK; where the entries of a slot are spread over more than two registers,
it takes more.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Step:
    #: the registers of B the step reads, one or two of them
    sources: Tuple[int, ...]
    #: per lane, the row of B the lane reads, or None for a lane that idles
    columns: Tuple[Optional[int], ...]
    #: per lane, the value of A the lane multiplies with, zero where it idles
    values: Tuple[float, ...]

    def index(self, v_size: int) -> List[int]:
        """Per lane, where its row of B is in the two sources together.

        The first source holds positions 0 to v_size - 1, the second the ones
        above. An idle lane reads the first position; it multiplies with zero.
        """

        result = []
        for column in self.columns:
            if column is None:
                result.append(0)
            else:
                register, position = divmod(column, v_size)
                result.append(position + v_size * self.sources.index(register))
        return result

    def in_place(self, v_size: int) -> bool:
        """Whether every lane reads its own position of one register, so that
        the register serves as it is, without a permutation."""

        if len(self.sources) != 1:
            return False
        base = self.sources[0] * v_size
        return all(
            column is None or column == base + lane
            for lane, column in enumerate(self.columns)
        )


@dataclass(frozen=True)
class Slice:
    #: the first row of A and of C in the slice
    start: int
    #: how many rows it has, the vector length but at the end of A
    rows: int
    steps: Tuple[Step, ...]


class Ellpack:
    """The slices of a sparse A for vectors of v_size entries.

    values is A as a dense array, m by k; its nonzero entries are the entries.
    """

    def __init__(self, values, v_size: int):
        values = np.asarray(values, dtype=np.float64)
        self.m, self.k = values.shape
        self.v_size = v_size
        #: how many vector registers a column of B fills
        self.registers = -(self.k // -v_size)
        self.slices = tuple(
            self._slice(values, start, min(start + v_size, self.m))
            for start in range(0, self.m, v_size)
        )

    def _slice(self, values, start: int, stop: int) -> Slice:
        v = self.v_size
        # what is left of every row: (column, value), by column
        left = [
            [(int(c), float(values[row, c])) for c in np.nonzero(values[row])[0]]
            for row in range(start, stop)
        ]
        steps = []
        while any(left):
            registers = sorted({c // v for row in left for c, _ in row})
            candidates = [(r,) for r in registers] + list(combinations(registers, 2))

            def covered(sources):
                return sum(1 for row in left if any(c // v in sources for c, _ in row))

            # the pair the most lanes have an entry in; among equals the one
            # first in order, and a single register where it covers as much
            best = max(candidates, key=lambda s: (covered(s), -len(s)))
            columns = []
            numbers = []
            for row in left:
                pick = next((e for e in row if e[0] // v in best), None)
                if pick is None:
                    columns.append(None)
                    numbers.append(0.0)
                else:
                    row.remove(pick)
                    columns.append(pick[0])
                    numbers.append(pick[1])
            # a pair of which one register ended up unread is a single register
            used = tuple(
                r for r in best if any(c is not None and c // v == r for c in columns)
            )
            columns += [None] * (v - len(columns))
            numbers += [0.0] * (v - len(numbers))
            steps.append(Step(used, tuple(columns), tuple(numbers)))
        return Slice(start, stop - start, tuple(steps))

    @property
    def steps(self) -> int:
        return sum(len(s.steps) for s in self.slices)

    def permutations(self) -> int:
        return sum(
            1 for s in self.slices for step in s.steps if not step.in_place(self.v_size)
        )
