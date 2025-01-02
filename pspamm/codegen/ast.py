
from typing import List, TYPE_CHECKING
from pspamm.codegen.operands import *

if TYPE_CHECKING:
    from pspamm.codegen.arm.visitors import Visitor


class AsmStmt:
    comment = None

    def accept(self, visitor: "Visitor"):
        raise Exception("AsmStmt is supposed to be abstract")
    
    def reg_in_candidate(self):
        return ()
    
    def reg_out_candidate(self):
        return ()
    
    def regs_in(self):
        return set(reg for regc in self.reg_in_candidate() if regc is not None for reg in regc.registers() if isinstance(reg, Register))

    def regs_out(self):
        return set(reg for regc in self.reg_out_candidate() if regc is not None for reg in regc.registers() if isinstance(reg, Register))
    
    def regs(self):
        return self.regs_in() | self.regs_out()
    
    def args_in(self):
        return set(reg for reg in self.reg_in_candidate() if reg is not None and isinstance(reg, InputOperand))

    def args_out(self):
        return set(reg for reg in self.reg_out_candidate() if reg is not None and isinstance(reg, InputOperand))
    
    def barrier(self):
        return False
    
    def args(self):
        return self.args_in() | self.args_out()
    
    def normalize(self):
        yield self
    
    def flatten(self):
        yield self
    
    def stmtname(self):
        return '???'
    
    def __str__(self):
        inregs = ', '.join(reg.ugly for reg in self.regs_in())
        outregs = ', '.join(reg.ugly for reg in self.regs_out())
        return f'{self.stmtname()} {inregs} -> {outregs}'

class GenericStmt(AsmStmt):
    operation = None

    def accept(self, visitor: "Visitor"):
        visitor.visitStmt(self)


class MovStmt(AsmStmt):
    src = None
    dest = None
    typ = None
    aligned = False
    pred = None
    expand = False
    temp = None

    def accept(self, visitor: "Visitor"):
        visitor.visitMov(self)
    
    def reg_in_candidate(self):
        return (self.src,self.temp,self.pred)
    
    def reg_out_candidate(self):
        return (self.dest,)
    
    def stmtname(self):
        return 'mov'

class LeaStmt(AsmStmt):
    src = None
    dest = None
    offset = None
    typ = None
    aligned = False
    pred = None

    def accept(self, visitor: "Visitor"):
        visitor.visitLea(self)
    
    def reg_in_candidate(self):
        return (self.src,self.pred)
    
    def reg_out_candidate(self):
        return (self.dest,)
    
    def stmtname(self):
        return 'lea'

class LoadStmt(AsmStmt):
    src = None
    dest = None
    typ = None
    aligned = False
    dest2 = None
    # used in arm_sve:
    pred = None
    is_B = None
    scalar_offs = False
    add_reg = None

    def accept(self, visitor: "Visitor"):
        visitor.visitLoad(self)
    
    def reg_in_candidate(self):
        return (self.src,self.pred,self.add_reg)
    
    def reg_out_candidate(self):
        return (self.dest, self.dest2)
    
    def stmtname(self):
        return 'load'

class StoreStmt(AsmStmt):
    src = None
    dest = None
    typ = None
    aligned = False
    src2 = None
    # used in arm_sve:
    pred = None
    scalar_offs = False
    add_reg = None

    def accept(self, visitor: "Visitor"):
        visitor.visitStore(self)
    
    def reg_in_candidate(self):
        return (self.src, self.src2, self.pred, self.add_reg)
    
    def reg_out_candidate(self):
        return (self.dest,)
    
    def stmtname(self):
        return 'store'

class PrefetchStmt(AsmStmt):
    dest = None
    # used in arm_sve:
    pred = None
    precision = None
    access_type = None

    def accept(self, visitor: "Visitor"):
        visitor.visitPrefetch(self)
    
    def stmtname(self):
        return 'prefetch'


class FmaStmt(AsmStmt):
    bcast_src = None
    mult_src = None
    add_dest = None
    bcast = None
    pred = None
    sub = False

    def accept(self, visitor: "Visitor"):
        visitor.visitFma(self)
    
    def reg_in_candidate(self):
        return (self.add_dest, self.bcast_src, self.mult_src, self.pred)
    
    def reg_out_candidate(self):
        return (self.add_dest,)
    
    def stmtname(self):
        return 'fma'

class MulStmt(AsmStmt):
    src = None
    mult_src = None
    dest = None
    pred = None

    def accept(self, visitor: "Visitor"):
        visitor.visitMul(self)
    
    def reg_in_candidate(self):
        return (self.mult_src,self.src,self.pred)
    
    def reg_out_candidate(self):
        return (self.dest,)
    
    def stmtname(self):
        return 'mul'

class BcstStmt(AsmStmt):
    bcast_src = None
    dest = None
    pred = None

    def accept(self, visitor: "Visitor"):
        visitor.visitBcst(self)
    
    def reg_in_candidate(self):
        return (self.bcast_src,self.pred,)
    
    def reg_out_candidate(self):
        return (self.dest,)
    
    def stmtname(self):
        return 'broadcast'

class AddStmt(AsmStmt):
    src = None
    dest = None
    typ = None
    additional = None
    pred = None

    def accept(self, visitor: "Visitor"):
        visitor.visitAdd(self)

    def reg_in_candidate(self):
        if self.additional is not None:
            return (self.src,self.additional,self.pred)
        else:
            return (self.src,self.dest,self.pred)
    
    def reg_out_candidate(self):
        return (self.dest,)
    
    def stmtname(self):
        return 'add'

class CmpStmt(AsmStmt):
    lhs = None
    rhs = None
    pred = None

    def accept(self, visitor: "Visitor"):
        visitor.visitCmp(self)
    
    def reg_in_candidate(self):
        return (self.lhs,self.rhs,self.pred)
    
    def stmtname(self):
        return 'cmp'

class LabelStmt(AsmStmt):
    label = None

    def accept(self, visitor: "Visitor"):
        visitor.visitLabel(self)
    
    def __str__(self):
        return f'Label: {self.label.ugly}'


class JumpStmt(AsmStmt):
    destination = None
    cmpreg = None

    def accept(self, visitor: "Visitor"):
        visitor.visitJump(self)
    
    def reg_in_candidate(self):
        return (self.cmpreg,)

    def stmtname(self):
        return 'branch'

class DataStmt(AsmStmt):
    value = None
    asmType = None

    def accept(self, visitor: "Visitor"):
        visitor.visitData(self)

class RVSetVLStmt(AsmStmt):
    actual = None
    requested = None

    def accept(self, visitor: "Visitor"):
        visitor.visitRVSetVLStmt(self)
    
    def reg_in_candidate(self):
        return (self.requested,)
    
    def reg_out_candidate(self):
        return (self.actual,)
    
    def barrier(self):
        return True

class Block(AsmStmt):
    contents = []

    def accept(self, visitor: "Visitor"):
        visitor.visitBlock(self)
    
    def normalize(self):
        return (subcontent for content in self.contents for subcontent in content.normalize())
    
    def flatten(self):
        return (subcontent for content in self.contents for subcontent in content.flatten())
    
    def regs_in(self):
        regs = set()
        for instr in self.contents:
            regs |= instr.regs_in()
        return regs
    
    def regs_out(self):
        regs = set()
        for instr in self.contents:
            regs |= instr.regs_out()
        return regs
    
    def __str__(self):
        return 'block {\n' + '\n'.join(str(content) for content in self.contents) + '\n}'

class Command(AsmStmt):
    name = None

    def accept(self, visitor: "Visitor"):
        visitor.visitCommand(self)

    def make(self, e) -> Block:
        raise NotImplementedError()
