# Need a native Python matrix type.
# Lists of lists are too cumbersome, and scipy does not understand typing.
# Also don't want to introduce a hard dependence on scipy if not necessary.

from typing import Generic, List, Tuple, TypeVar, Union, overload

import numpy as np
from scipy.io import mmread

T = TypeVar("T")


class Matrix(Generic[T]):
    """A two-dimensional array with matrix indexing semantics.

    Indexing yields either a scalar or another two-dimensional Matrix:
    ``m[i, :]`` is a row vector of shape ``(1, cols)`` and ``m[:, j]`` a column
    vector of shape ``(rows, 1)``.
    """

    def __init__(self, data):
        if isinstance(data, Matrix):
            underlying = np.array(data._underlying)
        else:
            underlying = np.asarray(data)
        if underlying.ndim == 1:
            underlying = underlying.reshape(1, -1)
        assert underlying.ndim == 2, "Matrix requires two-dimensional data"
        self._underlying = underlying
        self.shape = self._underlying.shape
        self.rows = self.shape[0]
        self.cols = self.shape[1]

    @classmethod
    def full(cls, rows: int, cols: int, initial_value: T):
        """Create a brand new matrix of given size"""
        return cls(np.full((rows, cols), initial_value))

    def __array__(self, dtype=None, copy=None):
        if dtype is None:
            return self._underlying
        return self._underlying.astype(dtype)

    def __repr__(self):
        col_str = []
        for ri in range(self.rows):
            row_str = []
            for ci in range(self.cols):
                row_str.append(str(self._underlying[ri, ci]).rjust(8))
            col_str.append("".join(row_str))
        return "\n".join(col_str)

    def __eq__(self, other):
        return (self._underlying == other._underlying).all()

    @overload
    def __getitem__(self, t: Tuple[slice, slice]) -> "Matrix[T]":
        pass

    @overload
    def __getitem__(self, t: Tuple[int, int]) -> T:
        pass

    def __getitem__(self, t) -> Union[T, "Matrix[T]"]:
        result = self._underlying[t]
        if not isinstance(result, np.ndarray):
            return result
        if result.ndim == 0:
            return result[()]
        if result.ndim == 1:
            # a single integer index collapses one axis; restore it, so that the
            # result stays a row or a column vector
            row_indexed = isinstance(t, tuple) and isinstance(t[0], (int, np.integer))
            result = result.reshape((1, -1) if row_indexed else (-1, 1))
        return Matrix(result)

    def __setitem__(self, cell: Tuple[int, int], value: T):
        if isinstance(value, Matrix):
            value = value._underlying.reshape(np.shape(self._underlying[cell]))
        self._underlying[cell] = value

    def __or__(self, other):
        return Matrix(self._underlying | other._underlying)

    def __and__(self, other):
        return Matrix(self._underlying & other._underlying)

    def any(self, axis=None, out=None):
        return self._underlying.any(axis, out)

    def nnz(self, axis=None) -> Union[int, List[int]]:
        if axis is None:
            return sum(
                self[r, c] != 0 for r in range(self.rows) for c in range(self.cols)
            )
        if axis == 1:
            return [
                sum(self[r, c] != 0 for r in range(self.rows)) for c in range(self.cols)
            ]
        if axis == 0:
            return [
                sum(self[r, c] != 0 for c in range(self.cols)) for r in range(self.rows)
            ]

    @classmethod
    def load(cls, filename) -> "Matrix[float]":
        m = mmread(filename)
        if not isinstance(m, np.ndarray):
            m = m.toarray()
        return Matrix(m.astype(np.float64))
