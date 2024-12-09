from pspamm.cursors import *
from pspamm.codegen.ast import *
from pspamm.codegen.precision import *
from abc import ABC, abstractmethod

class AbstractGenerator(ABC):
    def __init__(self, precision: Precision):
      self.precision = precision

    def get_precision(self):
      return self.precision
    
    def set_sparse(self):
        pass

    # taken from https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    def ceil_div(self, n, d):
        return -(n // -d)

    @abstractmethod
    def init_mask(self, bm, v_size, tempreg, maskreg):
        pass

    @abstractmethod
    def use_broadcast(self):
        pass

    @abstractmethod
    def has_masks(self):
        pass

    @abstractmethod
    def get_v_size(self):
        pass

    @abstractmethod
    def get_template(self):
        pass

    @abstractmethod
    def make_reg_blocks(self, bm:int, bn:int, bk:int, v_size:int, nnz:int, m:int, n:int, k:int):
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def make_zero_block(self, registers: Matrix[Register], additional_regs) -> Block:
        pass

    @abstractmethod
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
                         to_B_block: Coords = Coords(),
                         sub: bool = False
                         ) -> Block:
        pass

    @abstractmethod
    def init_prefetching(self, prefetching):
        pass
