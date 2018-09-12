from cursors.matrix import Matrix
from cursors.coords import Coords

from codegen.operands import *
from codegen.ast import AsmStmt, Command

from typing import NamedTuple, List, Tuple



class BlockInfo(NamedTuple):
    br: int                 # Cell rows in block
    bc: int                 # Cell cols in block
    pattern_index: int      # Pattern location in index
    pattern: Matrix[bool]   # The pattern itself



class CursorLocation:
    current_block: Coords  # Absolute coords of current block
    current_cell: Coords   # Relative?

    def __init__(self,
                 current_block: Coords = Coords(absolute=True),
                 current_cell: Coords = Coords(absolute=False)
                ) -> None:
        assert(current_cell.absolute == False)
        self.current_block = current_block
        self.current_cell = current_cell


class Cursor:
    name: str
    base_ptr: Register
    index_ptr: Register
    r: int
    c: int
    br: int
    bc: int

    @property
    def Br(self) -> int:
        return self.r // self.br

    @property
    def Bc(self) -> int:
        return self.c // self.bc

    @property
    def brf(self) -> int:
        return self.r % self.br

    @property
    def bcf(self) -> int:
        return self.c % self.bc

    def move(self,
             src: CursorLocation,
             dest_block: Coords
            ) -> Tuple[AsmStmt, CursorLocation]:
        raise NotImplementedError()

    def look(self,
             src: CursorLocation,
             dest_block: Coords,
             dest_cell: Coords
            ) -> Tuple[MemoryAddress, str]:
        raise NotImplementedError()

    def start_location(self, dest_block: Coords = Coords(absolute=True)) -> CursorLocation:
        raise NotImplementedError()

    def get_block(self, src: CursorLocation=None, dest_block: Coords=None) -> BlockInfo:
        raise NotImplementedError()


class CursorMovement(Command):
    matrix: Cursor

class CursorLookup(MemoryAddress):
    matrix: Cursor
    src: CursorLocation
    dest: CursorLocation





