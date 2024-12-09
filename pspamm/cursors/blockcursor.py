from pspamm.cursors.abstractcursor import *
from pspamm.cursors.matrix import Matrix
from pspamm.cursors.coords import Coords

from pspamm.codegen.sugar import *
from typing import cast

class BlockCursor(Cursor):

    blocks = None
    patterns = None
    offsets = None

    def __init__(self,
                 name: str,
                 base_ptr: Register,
                 rows: int,
                 cols: int,
                 ld: int,
                 block_rows: int,
                 block_cols: int,
                 scalar_bytes:int,
                 blocks: Matrix[int],
                 patterns: List[Matrix[bool]],
                 mtx_overhead) -> None:

        self.name = name
        self.base_ptr = base_ptr
        self.scalar_bytes = scalar_bytes
        self.r = rows
        self.c = cols
        self.ld = ld
        self.br = block_rows
        self.bc = block_cols
        self.blocks = blocks
        self.patterns = patterns

        self.offsets = Matrix.full(rows, cols, -1)
        x = 0
        for i in range(self.c):
            for j in range(self.r):
                Bci = i // self.bc
                Bri = j // self.br
                index = cast(int, blocks[Bri, Bci])
                pattern = patterns[index]   
                if pattern[j % self.br,i % self.bc]:
                    self.offsets[j, i] = x
                    x += 1
            if ld != 0:
                x += self.ld - self.r
            x += mtx_overhead[i]

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

        return dest_offset


    def move(self,
             src_loc: CursorLocation,
             dest_block: Coords
            ) -> Tuple[AsmStmt, CursorLocation]:

        comment = f"Move {self.name} to {str(dest_block)}"

        if dest_block.absolute:
            dest_loc = self.start_location(dest_block)
        else:
            dest_loc = self.start_location(dest_block + src_loc.current_block)

        offset_bytes = self.offset(src_loc, dest_loc) * self.scalar_bytes
        
        return add(offset_bytes, self.base_ptr, comment), dest_loc


    def look(self,
             src_loc: CursorLocation,
             dest_block: Coords,
             dest_cell: Coords
            ) -> Tuple[MemoryAddress, str]:

        dest_loc = CursorLocation(dest_block, dest_cell)
        offset_bytes = self.offset(src_loc, dest_loc) * self.scalar_bytes
        comment = "{}[{},{}][{},{}]".format(self.name,dest_block.down,dest_block.right,dest_cell.down,dest_cell.right)

        addr = pspamm.architecture.operands.mem(self.base_ptr, offset_bytes)
        
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

        raise Exception("Block {} has no starting location because it is empty!".format(dest_block))


    def start(self) -> CursorLocation:

        Br, Bc = self.blocks.shape
        for Bci in range(Bc):
            for Bri in range(Br):
                target_block = Coords(down=Bri, right=Bci, absolute=True)
                if self.has_nonzero_block(None, target_block):
                    return self.start_location(target_block)
        raise Exception("Matrix is completely empty!")


def sparse_mask(A_regs: Matrix[Register],
           A: Cursor,
           A_ptr: CursorLocation,
           A_block_offset: Coords,
           B: Cursor,
           B_ptr: CursorLocation,
           B_block_offset: Coords,
           v_size: int,
           has_mask: bool = False
          ) -> Matrix[bool]:

    Vr, Vc = A_regs.shape
    mask = Matrix.full(Vr, Vc, False)
    A_br, A_bc, A_idx, A_pat = A.get_block(A_ptr, A_block_offset)
    B_br, B_bc, B_idx, B_pat = B.get_block(B_ptr, B_block_offset)

    if not has_mask:
        assert (Vr * v_size == A_br)    # bm must tile m exactly for now in non-mask-supporting ISAs
    assert(Vc >= A_bc)                  # Matrix block must fit in register block
    assert(A_bc == B_br)                # Matrix blocks are compatible

    # Mask out registers not used in current block, including zero-rows of B and A
    for Vci in range(A_bc):
        if B_pat[Vci,:].any(axis=1):
            mask[:,Vci] = A_pat[:,Vci]

    return mask
