
from typing import List, TYPE_CHECKING
from pspamm.codegen.operands import Operand, Label, Register, AsmType, MemoryAddress

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
    
    def reg_in(self):
        return set(reg for reg in self.reg_in_candidate() if isinstance(reg, Register))

    def reg_out(self):
        return set(reg for reg in self.reg_out_candidate() if isinstance(reg, Register))


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
        return (src,)
    
    def reg_out_candidate(self):
        return (dest,temp)

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
        return (src,)
    
    def reg_out_candidate(self):
        return (dest,)

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
        return (src,)
    
    def reg_out_candidate(self):
        return (dest,dest2)

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
        return (src,src2)
    
    def reg_out_candidate(self):
        return (dest)

class PrefetchStmt(AsmStmt):
    dest = None
    # used in arm_sve:
    pred = None
    precision = None
    access_type = None

    def accept(self, visitor: "Visitor"):
        visitor.visitPrefetch(self)


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
        return (bcast_src,mult_src)
    
    def reg_out_candidate(self):
        return (add_dest,)

class MulStmt(AsmStmt):
    src = None
    mult_src = None
    dest = None
    pred = None

    def accept(self, visitor: "Visitor"):
        visitor.visitMul(self)
    
    def reg_in_candidate(self):
        return (mult_src,src)
    
    def reg_out_candidate(self):
        return (dest,)

class BcstStmt(AsmStmt):
    bcast_src = None
    dest = None
    pred = None

    def accept(self, visitor: "Visitor"):
        visitor.visitBcst(self)
    
    def reg_in_candidate(self):
        return (bcast_src,)
    
    def reg_out_candidate(self):
        return (dest,)

class AddStmt(AsmStmt):
    src = None
    dest = None
    typ = None
    additional = None
    pred = None

    def accept(self, visitor: "Visitor"):
        visitor.visitAdd(self)

    def reg_in_candidate(self):
        return (src,additional)
    
    def reg_out_candidate(self):
        return (dest,)

class CmpStmt(AsmStmt):
    lhs = None
    rhs = None
    pred = None

    def accept(self, visitor: "Visitor"):
        visitor.visitCmp(self)
    
    def reg_in_candidate(self):
        return (lhs,rhs)
    
    def reg_out_candidate(self):
        return ()

class LabelStmt(AsmStmt):
    label = None

    def accept(self, visitor: "Visitor"):
        visitor.visitLabel(self)


class JumpStmt(AsmStmt):
    destination = None

    def accept(self, visitor: "Visitor"):
        visitor.visitJump(self)

class DataStmt(AsmStmt):
    value = None
    asmType = None

    def accept(self, visitor: "Visitor"):
        visitor.visitData(self)


class Block(AsmStmt):
    contents = []

    def accept(self, visitor: "Visitor"):
        visitor.visitBlock(self)

class ControlBlock(AsmStmt):
    contents = []

class Command(AsmStmt):
    name = None

    def accept(self, visitor: "Visitor"):
        visitor.visitCommand(self)

    def make(self, e) -> Block:
        raise NotImplementedError()

class VirtualRegister(Register):
    pass
