from typing import List, Set

from pypspamm.codegen.forms import Loop
from pypspamm.codegen.sugar import *
from pypspamm.codegen.visitor import Visitor


def used_registers(stmt) -> Set[Register]:
    """The registers a statement uses, those of the statements of a block or a
    loop included. A loop is looked into through its body and its counter: its
    contents number a label each time they are taken, which would change the
    labels of the code printed afterwards."""

    if isinstance(stmt, Loop):
        return {stmt.iteration_var} | used_registers(stmt.body_contents)
    if isinstance(stmt, Block):
        used = set()
        for content in stmt.contents:
            used |= used_registers(content)
        return used
    return stmt.regs()


class Analyzer:
    def __init__(self, reserved: List[Register] = None):
        self.clobbered_registers = set(reserved)
        self.input_operands = set()

    def collect(self, block: Block):
        for instr in block.flatten():
            self.clobbered_registers |= instr.regs()
            self.input_operands |= instr.args()
