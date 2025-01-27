from pspamm.codegen.visitor import Visitor
from pspamm.codegen.sugar import *

from typing import List, Set

class Analyzer:
    def __init__(self, starting_regs: List[Register] = None):
        self.clobbered_registers = set(starting_regs)
        self.input_operands = set()

    def collect(self, block: Block):
        for instr in block.flatten():
            self.clobbered_registers |= instr.regs()
            self.input_operands |= instr.args()
