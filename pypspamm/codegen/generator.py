from abc import ABC, abstractmethod

from pypspamm.codegen.analysis import used_registers
from pypspamm.codegen.arguments import Argument
from pypspamm.codegen.ast import *
from pypspamm.codegen.precision import *
from pypspamm.codegen.sugar import ld
from pypspamm.codegen.target import TargetDescription
from pypspamm.cursors import *


class AbstractGenerator(ABC):
    #: what the target offers; set by every concrete generator
    target: TargetDescription = None
    #: whether alpha and beta go in as pointers to them rather than as values
    scalars_by_pointer = False

    def __init__(self, precision: Precision):
        self.precision = precision

    def get_precision(self):
        return self.precision

    def set_sparse(self):
        pass

    # taken from https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    def ceil_div(self, n, d):
        return -(n // -d)

    def arguments(self, starting_regs, prefetch):
        """The arguments of a kernel and the registers it reads them in, the
        first of starting_regs (see codegen/arguments.py): A, B and C go in as
        they are, alpha and beta as values from memory or as pointers to them
        (scalars_by_pointer), prefetch where the kernel prefetches."""

        arguments = [
            Argument("A", starting_regs[0], "A", "const {real_type}*"),
            Argument("B", starting_regs[1], "B", "const {real_type}*"),
            Argument("C", starting_regs[2], "C", "{real_type}*"),
        ]
        for name, register in (("alpha", starting_regs[3]), ("beta", starting_regs[4])):
            if self.scalars_by_pointer:
                arguments.append(Argument(name, register, f"&{name}", "{real_type}*"))
            else:
                arguments.append(Argument(name, register))
        if prefetch:
            arguments.append(
                Argument("prefetch", starting_regs[5], "prefetch", "const {real_type}*")
            )
        return arguments

    def load_argument(self, argument):
        """The statement that loads an argument the C function does not bind to
        its register, from its input operand."""

        return ld(
            InputOperand(argument.name, "m", argument.name), argument.register, False
        )

    def take_arguments(self, starting_regs, prefetch, body):
        """How a kernel of these statements takes its arguments: the loads of
        the values it reads, to go ahead of the statements; the arguments the
        C function binds to their registers; and the registers of
        starting_regs that hold no argument, which the kernel keeps for
        itself. An argument whose register the statements do not use is left
        out."""

        used = used_registers(body)
        arguments = self.arguments(starting_regs, prefetch)
        read = [argument for argument in arguments if argument.register in used]
        loads = [
            self.load_argument(argument) for argument in read if not argument.bound
        ]
        bound = [argument for argument in read if argument.bound]
        taken = {argument.register for argument in arguments}
        reserved = [register for register in starting_regs if register not in taken]
        return loads, bound, reserved

    @abstractmethod
    def init_mask(self, bm, v_size, tempreg, maskreg):
        pass

    def use_broadcast(self):
        return self.target.scalar_broadcast

    def has_masks(self):
        return self.target.has_masks

    @abstractmethod
    def get_v_size(self):
        pass

    @abstractmethod
    def get_template(self):
        pass

    @abstractmethod
    def make_reg_blocks(
        self,
        bm: int,
        bn: int,
        bk: int,
        v_size: int,
        nnz: int,
        m: int,
        n: int,
        k: int,
        copies: int = 1,
    ):
        pass

    @abstractmethod
    def move_register_block(
        self,
        cursor: Cursor,
        cursor_ptr: CursorLocation,
        block_offset: Coords,
        registers: Matrix[Register],
        v_size: int,
        additional_regs,
        mask: Matrix[bool] = None,
        store: bool = False,
    ) -> Block:
        pass

    @abstractmethod
    def make_zero_block(self, registers: Matrix[Register], additional_regs) -> Block:
        pass

    @abstractmethod
    def make_microkernel(
        self,
        A: Cursor,
        B: Cursor,
        A_ptr: CursorLocation,
        B_ptr: CursorLocation,
        A_regs: Matrix[Register],
        B_regs,
        C_regs: Matrix[Register],
        v_size: int,
        additional_regs,
        to_A_block: Coords = Coords(),
        to_B_block: Coords = Coords(),
        sub: bool = False,
    ) -> Block:
        pass
