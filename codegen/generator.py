from cursors import *
from codegen.ast import *

class AbstractGenerator:
  
    def get_v_size(self):
        raise NotImplementedError()

    def get_template(self):
        raise NotImplementedError()

    def make_reg_blocks(self, bm:int, bn:int, bk:int, v_size:int, nnz:int, m:int, n:int, k:int):
        raise NotImplementedError()

    def move_register_block(self,
                            cursor: Cursor,
                            cursor_ptr: CursorLocation,
                            block_offset: Coords,
                            registers: Matrix[Register],
                            v_size: int,
                            additional_regs,
                            mask: Matrix[bool] = None,
                            store: bool = False
                            ) -> Block:

        raise NotImplementedError()

    def make_zero_block(self, registers: Matrix[Register], additional_regs) -> Block:
        raise NotImplementedError()

    def make_microkernel(self,
                         A: Cursor,
                         B: Cursor,
                         A_ptr: CursorLocation,
                         B_ptr: CursorLocation,
                         A_regs: Matrix[Register],
                         B_regs,
                         C_regs: Matrix[Register],
                         v_size:int,
                         additional_regs,
                         to_A_block: Coords = Coords(),
                         to_B_block: Coords = Coords()
                         ) -> Block:
        raise NotImplementedError()

    def init_prefetching(self, prefetching):
        raise NotImplementedError()
