from cursors.abstractcursor import *
from cursors.matrix import Matrix
from cursors.coords import Coords

from codegen.sugar import *
from typing import cast # TODO: Use function overloads correctly now that I know how

class BlockCursorDef(CursorDef):

    blocks: Matrix[int]
    patterns: List[Matrix[bool]]
    offsets: Matrix[int]

    def __init__(self,
                 name: str,
                 base_ptr: Register,
                 rows: int,
                 cols: int,
                 block_rows: int,
                 block_cols: int,
                 blocks: Matrix[int],
                 patterns: List[Matrix[bool]]) -> None:

        self.name = name
        self.base_ptr = base_ptr
        self.index_ptr = None
        self.scale = 1
        self.scalar_bytes = 8
        self.r = rows
        self.c = cols
        self.br = block_rows
        self.bc = block_cols  # The reported blocksizes are the truth
        self.blocks = blocks
        self.patterns = patterns

        # The block patterns that are passed in need to conform with the truth,
        # not the other way around.
        topleftblock = blocks[0,0]
        topleftblock = cast(int, topleftblock)
        brp, bcp = patterns[topleftblock].shape
        Brp, Bcp = blocks.shape

        assert(brp==self.br)
        assert(bcp==self.bc)
        assert(Brp==self.Br)
        assert(Bcp==self.Bc)

        # This enforces a constant blocksize (excluding fringes), which
        # makes a lot of things easier
        x = 0
        offsets = Matrix.full(rows+1, cols+1, -1)
        for Bci in range(self.Bc):        # Iterate over blocks of columns
            for Bri in range(self.Br):    # Iterate over blocks of rows
                index = cast(int, blocks[Bri, Bci])
                pattern = patterns[index]                    # Pattern for current block
                for bci in range(self.bc):                        # Iterate over cols inside block
                    for bri in range(self.br):                    # Iterate over rows inside block
                        if pattern[bri,bci]:
                            offsets[Bri*self.br + bri, Bci*self.bc + bci] = x
                            x += 1
        # TODO: Handle fringes correctly.
        self.offsets = offsets



    def offset(self,
               src_loc: CursorLocation,
               dest_loc: CursorLocation
              ) -> int:

        src_block = src_loc.current_block
        src_cell = src_loc.current_cell
        dest_block = dest_loc.current_block
        dest_cell = dest_loc.current_cell

        if not dest_block.absolute:
            dest_block += src_block

        assert(src_block.absolute)
        assert(not src_cell.absolute)
        assert(not dest_cell.absolute)

        src_cell += Coords(src_block.down*self.br, src_block.right*self.bc, True)
        dest_cell += Coords(dest_block.down*self.br, dest_block.right*self.bc, True)

        src_offset = self.offsets[src_cell.down, src_cell.right]
        dest_offset = self.offsets[dest_cell.down, dest_cell.right]

        if (src_offset == -1 or dest_offset == -1):
            raise Exception("Cursor location does not exist in memory!")

        return dest_offset - src_offset


    def move(self,
             src_loc: CursorLocation,
             dest_block: Coords
            ) -> Tuple[AsmStmt, CursorLocation]:

        comment = f"Move {self.name} to {str(dest_block)}"
        if self.index_ptr is None:
            ptr_to_move = self.base_ptr
        else:
            ptr_to_move = self.index_ptr

        if dest_block.absolute:
            dest_loc = self.start_location(dest_block)
        else:
            dest_loc = self.start_location(dest_block + src_loc.current_block)

        offset_bytes = self.offset(src_loc, dest_loc) * self.scalar_bytes
        
        return add(offset_bytes, ptr_to_move, comment), dest_loc


    def look(self,
             src_loc: CursorLocation,
             dest_block: Coords,
             dest_cell: Coords
            ) -> Tuple[MemoryAddress, str]:

        src_offset_abs = self.offset(src.current_block, Coords(), src.current_cell)
        dest_offset_abs = self.offset(src.current_block, dest_block, dest_cell)
        rel_offset = self.scalar_bytes * (dest_offset_abs - src_offset_abs)
        comment = f"{self.name}[{dest_block.down},{dest_block.right}][{dest_cell.down},{dest_cell.right}]"

        addr = architecture.operands.mem(self.base_ptr, self.index_ptr, self.scale, rel_offset)
        
        return (addr, comment)


    def get_block(self, src: CursorLocation=None, dest_block: Coords=None) -> BlockInfo:

        if src is None: # Have dest_block but no src
            assert(dest_block is not None)
            assert(dest_block.absolute == True)
            block_abs = dest_block

        elif dest_block is None: # Have src but no dest_block
            assert(src.current_block.absolute == True)
            block_abs = src.current_block

        elif dest_block.absolute: # Have src and absolute dest_block
            block_abs = dest_block

        else: # Have both src and relative dest_block
            assert(src.current_block.absolute == True)
            block_abs = dest_block + src.current_block


        br = self.br if block_abs.down < self.Br else self.brf   #TODO: Verify these
        bc = self.bc if block_abs.right < self.Bc else self.bcf
        index = self.blocks[block_abs.down, block_abs.right]
        index = cast(int, index)  # TODO: Overload functions correctly
        pattern = self.patterns[index][0:br, 0:bc]
        pattern = cast(Matrix[bool], pattern)
        return BlockInfo(br, bc, index, pattern)


    def has_nonzero_cell(self,
                         src_loc: CursorLocation,
                         dest_block: Coords,
                         dest_cell: Coords
                        ) -> bool:

        assert(not dest_cell.absolute)
        if not dest_block.absolute:
            dest_block += src_loc.current_block

        dest_cell += Coords(dest_block.down*self.br, dest_block.right*self.bc, True)
        return self.offsets[dest_cell.down, dest_cell.right] != -1


    def has_nonzero_block(self, src: CursorLocation, dest_block: Coords) -> bool:
        nonzero = False
        br,bc,idx,pat = self.get_block(src, dest_block)
        for bci in range(bc):
            for bri in range(br):
                if pat[bri,bci]:
                    nonzero = True
        return nonzero


    def start_location(self, dest_block: Coords = Coords(absolute=True)) -> CursorLocation:

        assert(dest_block.absolute == True)
        br,bc,idx,pat = self.get_block(dest_block=dest_block)
        for bci in range(bc):
            for bri in range(br):
                if pat[bri,bci]:
                    return CursorLocation(dest_block, Coords(down=bri, right=bci, absolute=False))

        raise Exception(f"Block {dest_block} has no starting location because it is empty!")


    def start(self) -> CursorLocation:

        Br, Bc = self.blocks.shape
        for Bci in range(Bc):
            for Bri in range(Br):
                target_block = Coords(down=Bri, right=Bci, absolute=True)
                if self.has_nonzero_block(None, target_block):
                    return self.start_location(target_block)
        raise Exception("Matrix is completely empty!")



    def _bounds_check(self, abs_cells: Coords) -> None:
        ri,ci = abs_cells.down, abs_cells.right
        r, c = self.offsets.shape
        if ri >= r or ci >= c or ri < 0 or ci < 0:
            raise Exception(f"Entry {ri},{ci} outside matrix!")

    def pattern(self) -> Matrix[bool]:
        return Matrix(self.offsets._underlying != -1)



def minicursor(name: str, base_ptr: Register, pattern: Matrix[bool]):
    rows, cols = pattern.shape
    cursor = BlockCursorDef(name, base_ptr, rows, cols, rows, cols, Matrix([[0]]), [pattern])
    return cursor


