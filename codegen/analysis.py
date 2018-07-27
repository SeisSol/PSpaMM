from codegen.visitors import Visitor
from codegen.sugar import *

from typing import List, Set

class Analyzer(Visitor):

    clobbered_registers: Set[Register]
    basic_blocks: BlockBuilder
    current_block: BlockBuilder
    flat: BlockBuilder
    stack: List[Block]

    def __init__(self):
        self.clobbered_registers = set()
        self.basic_blocks = block("Basic block representation")
        self.current_block = block("New basic block")
        self.basic_blocks.add(self.current_block)
        self.flat = block("Flattened representation")
        self.stack = []

    def visitFma(self, stmt: FmaStmt):
        self.clobbered_registers.add(stmt.add_dest)
        self.flat.add(stmt)
        self.current_block.add(stmt)

    def visitAdd(self, stmt: AddStmt):
        self.clobbered_registers.add(stmt.dest)
        self.flat.add(stmt)
        self.current_block.add(stmt)

    def visitLabel(self, stmt: LabelStmt):
        b = block("New basic block", stmt)
        self.current_block = b
        self.basic_blocks.add(b)
        self.flat.add(stmt)

    def visitJump(self, stmt: JumpStmt):
        self.current_block.add(stmt)
        b = block("New basic block")
        self.current_block = b
        self.basic_blocks.add(b)
        self.flat.add(stmt)

    def visitMov(self, stmt: MovStmt):
        s = f"mov {stmt.src.ugly} -> {stmt.dest.ugly}"
        if isinstance(stmt.dest, Register):
            self.clobbered_registers.add(stmt.dest)
        self.flat.add(stmt)
        self.current_block.add(stmt)

    def visitStore(self, stmt: MovStmt):
        s = f"str {stmt.src.ugly} -> {stmt.dest.ugly}"
        if isinstance(stmt.dest, Register):
            self.clobbered_registers.add(stmt.dest)
        self.flat.add(stmt)
        self.current_block.add(stmt)

    def visitLoad(self, stmt: MovStmt):
        s = f"ldr {stmt.src.ugly} -> {stmt.dest.ugly}"
        if isinstance(stmt.dest, Register):
            self.clobbered_registers.add(stmt.dest)
        self.flat.add(stmt)
        self.current_block.add(stmt)

    def visitCmp(self, stmt: CmpStmt):
        s = f"cmp {stmt.rhs.ugly} <=> {stmt.lhs.ugly}"
        self.flat.add(stmt)
        self.current_block.add(stmt)

    def visitBlock(self, block: Block):
        self.stack.append(block)
        for stmt in block.contents:
            stmt.accept(self)
        self.stack.pop()



