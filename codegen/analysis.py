from codegen.visitor import Visitor
from codegen.sugar import *

from typing import List, Set

class Analyzer(Visitor):

    def __init__(self, starting_regs: List[Register] = None):
        self.clobbered_registers = set(starting_regs)
        self.stack = []

    def visitFma(self, stmt: FmaStmt):
        self.clobbered_registers.add(stmt.add_dest)

    def visitMul(self, stmt: FmaStmt):
        self.clobbered_registers.add(stmt.dest)

    def visitBcst(self, stmt: FmaStmt):
        self.clobbered_registers.add(stmt.dest)

    def visitAdd(self, stmt: AddStmt):
        self.clobbered_registers.add(stmt.dest)

    def visitLabel(self, stmt: LabelStmt):
        pass

    def visitJump(self, stmt: JumpStmt):
        pass

    def visitMov(self, stmt: MovStmt):
        if isinstance(stmt.dest, Register):
            self.clobbered_registers.add(stmt.dest)

    def visitLea(self, stmt: MovStmt):
            self.clobbered_registers.add(stmt.dest)

    def visitStore(self, stmt: MovStmt):
        if isinstance(stmt.dest, Register):
            self.clobbered_registers.add(stmt.dest)

    def visitLoad(self, stmt: MovStmt):
        if isinstance(stmt.dest, Register):
            self.clobbered_registers.add(stmt.dest)

    def visitPrefetch(self, stmt: PrefetchStmt):
        self.clobbered_registers.add(stmt.dest.base)

    def visitCmp(self, stmt: CmpStmt):
        pass

    def visitBlock(self, block: Block):
        self.stack.append(block)
        for stmt in block.contents:
            stmt.accept(self)
        self.stack.pop()



