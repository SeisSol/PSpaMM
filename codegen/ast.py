
from typing import List, TYPE_CHECKING
from codegen.operands import Operand, Label, Register, AsmType, MemoryAddress

if TYPE_CHECKING:
    from codegen.arm.visitors import Visitor


class AsmStmt:
    comment = None
    implied_inputs = []
    implied_outputs = []

    def accept(self, visitor: "Visitor"):
        raise Exception("AsmStmt is supposed to be abstract!")


class GenericStmt(AsmStmt):
    operation = None
    inputs = None
    output = None

    def accept(self, visitor: "Visitor"):
        visitor.visitStmt(self)


class MovStmt(AsmStmt):
    src = None
    dest = None
    typ = None
    aligned = False

    def accept(self, visitor: "Visitor"):
        visitor.visitMov(self)

class LeaStmt(AsmStmt):
    src = None
    dest = None
    offset = None
    typ = None
    aligned = False

    def accept(self, visitor: "Visitor"):
        visitor.visitLea(self)

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

    def accept(self, visitor: "Visitor"):
        visitor.visitFma(self)

class MulStmt(AsmStmt):
    src = None
    mult_src = None
    dest = None
    pred = None

    def accept(self, visitor: "Visitor"):
        visitor.visitMul(self)

class BcstStmt(AsmStmt):
    bcast_src = None
    dest = None

    def accept(self, visitor: "Visitor"):
        visitor.visitBcst(self)

class AddStmt(AsmStmt):
    src = None
    dest = None
    typ = None
    additional = None

    def accept(self, visitor: "Visitor"):
        visitor.visitAdd(self)

class CmpStmt(AsmStmt):
    lhs = None
    rhs = None

    def accept(self, visitor: "Visitor"):
        visitor.visitCmp(self)

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


class Command(AsmStmt):
    name = None

    def accept(self, visitor: "Visitor"):
        visitor.visitCommand(self)

    def make(self, e) -> Block:
        raise NotImplementedError()




