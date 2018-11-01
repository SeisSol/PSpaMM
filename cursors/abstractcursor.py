from cursors.matrix import Matrix
from cursors.coords import Coords

from codegen.operands import *
from codegen.ast import AsmStmt, Command

from typing import List, Tuple

from collections import namedtuple

BlockInfo = namedtuple("Blockinfo", ("br bc pattern_index pattern"))

class CursorLocation:
    current_block = None  # Absolute coords of current block
    current_cell = None   # Relative?

    def __init__(self,
                 current_block = Coords(absolute=True),
                 current_cell = Coords(absolute=False)
                ) -> None:
        assert(current_cell.absolute == False)
        self.current_block = current_block
        self.current_cell = current_cell


class Cursor:
    name = None
    base_ptr = None
    index_ptr = None
    r = None
    c = None
    br = None
    bc = None

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
    matrix = None

class CursorLookup(MemoryAddress):
    matrix = None
    src = None
    dest = None





