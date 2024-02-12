

# Need a native Python matrix type.
# Lists of lists are too cumbersome, and scipy does not understand typing.
# Also don't want to introduce a hard dependence on scipy if not necessary.

from typing import TypeVar, Generic, Union, Tuple, List, overload, Any
from scipy.sparse import csc_matrix
from scipy.io import mmread, mmwrite
import numpy as np
import random

T = TypeVar('T')
class Matrix(Generic[T]):

    def __init__(self, data):
        if isinstance(data, Matrix):
            self._underlying = np.matrix(data._underlying)
        else:
            self._underlying = np.matrix(data)
        self.shape = self._underlying.shape
        self.rows = self.shape[0]
        self.cols = self.shape[1]

    @classmethod
    def full(cls, rows:int, cols:int, initial_value:T):
        """Create a brand new matrix of given size"""
        return cls(np.full((rows,cols), initial_value))

    def __repr__(self):
        col_str = []
        for ri in range(self.rows):
            row_str = []
            for ci in range(self.cols):
                row_str.append(str(self._underlying[ri,ci]).rjust(8))
            col_str.append("".join(row_str))
        return "\n".join(col_str)

    def __eq__(self, other):
        return (self._underlying == other._underlying).all()

    @overload
    def __getitem__(self, t: Tuple[slice,slice]) -> "Matrix[T]":
        pass

    @overload
    def __getitem__(self, t: Tuple[int,int]) -> T:
        pass

    def __getitem__(self, t) -> Union[T, "Matrix[T]"]:
        result = self._underlying[t]
        if isinstance(result, np.matrix):
            return Matrix(result)
        else:
            return result

    def __setitem__(self, cell:Tuple[int,int], value:T):
        self._underlying[cell] = value

    def __or__(self, other):
        return Matrix(self._underlying | other._underlying)

    def __and__(self, other):
        return Matrix(self._underlying & other._underlying)

    def any(self, axis=None, out=None):
        return self._underlying.any(axis, out)

    def nnz(self) -> int:
        return sum(self[r,c] != 0 for r in range(self.rows)
                                  for c in range(self.cols))

    @classmethod
    def load_pattern(cls, filename) -> "Matrix[bool]":
        m = mmread(filename)
        m = m.astype(np.bool)
        m = m.todense()
        return Matrix(m)

    @classmethod
    def load(cls, filename) -> "Matrix[float]":
        m = mmread(filename)
        m = m.astype(np.float64)
        if not isinstance(m, np.ndarray):
            m = m.todense()
        return Matrix(m)
