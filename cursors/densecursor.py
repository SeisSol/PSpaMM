from typing import List, Tuple, cast

from codegen.operands import *
from cursors import *


class DenseCursorDef(TiledCursorDef):
    def __init__(self,
                 name: str,
                 base_ptr: Register,
                 rows:int,
                 cols:int,
                 ld: int,
                 block_rows: int,
                 block_cols: int,
                 scalar_bytes:int = 8,
                 vector_width:int = 8) -> None:

        pattern = Matrix.full(block_rows, block_cols, True)
        TiledCursorDef.__init__(self, name, base_ptr, rows, cols, pattern, scalar_bytes, vector_width)
        self.ld = ld

    def offset(self,
               src_block: Coords,
               dest_block: Coords,
               dest_cell: Coords
              ) -> int:
        # TODO: Why not make offset compute the 1D distance
        # from current pointer to desired logical cell instead?

        assert(src_block.absolute == True)
        assert(dest_cell.absolute == False)
        if not dest_block.absolute:
            dest_block += src_block

        Bri, Bci = dest_block.down, dest_block.right
        bri, bci = dest_cell.down, dest_cell.right

        return (Bci*self.bc + bci) * self.ld + Bri*self.br + bri

