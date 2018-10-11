
from typing import List, TYPE_CHECKING
from codegen.operands import Operand, Label, Register, AsmType, MemoryAddress

if TYPE_CHECKING:
    from codegen.arm.visitors import Visitor


class AsmStmt:
    comment: str = None
    implied_inputs: List[Register] = []
    implied_outputs: List[Register] = []

    def accept(self, visitor: "Visitor"):
        raise Exception("AsmStmt is supposed to be abstract!")


class GenericStmt(AsmStmt):
    operation: str
    inputs: List[Operand]
    output: Operand

    def accept(self, visitor: "Visitor"):
        visitor.visitStmt(self)


class MovStmt(AsmStmt):
    src: Operand
    dest: Operand
    typ: AsmType
    aligned: bool = False

    def accept(self, visitor: "Visitor"):
        visitor.visitMov(self)

class LoadStmt(AsmStmt):
    src: Operand
    dest: Operand
    typ: AsmType
    aligned: bool = False
    dest2: Operand = None

    def accept(self, visitor: "Visitor"):
        visitor.visitLoad(self)

class StoreStmt(AsmStmt):
    src: Operand
    dest: Operand
    typ: AsmType
    aligned: bool = False
    src2: Operand = None

    def accept(self, visitor: "Visitor"):
        visitor.visitStore(self)

class PrefetchStmt(AsmStmt):
    dest: Operand

    def accept(self, visitor: "Visitor"):
        visitor.visitPrefetch(self)


class FmaStmt(AsmStmt):
    bcast_src: Register
    mult_src: Register
    add_dest: Register

    def accept(self, visitor: "Visitor"):
        visitor.visitFma(self)


class AddStmt(AsmStmt):
    src: Operand = None
    dest: Register = None
    typ: AsmType = None
    additional: Register = None

    def accept(self, visitor: "Visitor"):
        visitor.visitAdd(self)

class CmpStmt(AsmStmt):
    lhs: Operand
    rhs: Operand

    def accept(self, visitor: "Visitor"):
        visitor.visitCmp(self)

class LabelStmt(AsmStmt):
    label: Label = None

    def accept(self, visitor: "Visitor"):
        visitor.visitLabel(self)


class JumpStmt(AsmStmt):
    destination: Operand = None

    def accept(self, visitor: "Visitor"):
        visitor.visitJump(self)

class DataStmt(AsmStmt):
    value: Operand
    asmType: AsmType

    def accept(self, visitor: "Visitor"):
        visitor.visitData(self)


class Block(AsmStmt):
    contents : List[AsmStmt] = []

    def accept(self, visitor: "Visitor"):
        visitor.visitBlock(self)


class Command(AsmStmt):
    name: str

    def accept(self, visitor: "Visitor"):
        visitor.visitCommand(self)

    def make(self, e) -> Block:
        raise NotImplementedError()




