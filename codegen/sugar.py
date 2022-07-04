from typing import Union

from codegen.ast import *
from codegen.operands import *

import architecture

# Convenient statement constructors
def add(src: Union[Operand, int], dest: Register, comment: str = None, additional: Register = None):
    stmt = AddStmt()
    stmt.src = src if isinstance(src, Operand) else architecture.operands.c(src)
    stmt.dest = dest
    stmt.comment = comment
    stmt.additional = additional
    return stmt

def label(name: str):
    stmt = LabelStmt()
    stmt.label = architecture.operands.l(name)
    return stmt

def fma(bcast_src: Register, mult_src: Register, add_dest: Register, comment: str = None, bcast: bool = True, pred: Register = None):
    stmt = FmaStmt()
    stmt.bcast_src = bcast_src
    stmt.mult_src = mult_src
    stmt.add_dest = add_dest
    stmt.comment = comment
    stmt.bcast = bcast
    # used in arm_sve:
    stmt.pred = pred
    return stmt

def mul(src: Register, mult_src: Register, dest: Register, comment: str = None, pred: Register = None):
    stmt = MulStmt()
    stmt.src = src
    stmt.mult_src = mult_src
    stmt.dest = dest
    stmt.comment = comment
    # used in arm_sve:
    stmt.pred = pred
    return stmt

def bcst(bcast_src: Register, dest: Register, comment: str = None):
    stmt = BcstStmt()
    stmt.bcast_src = bcast_src
    stmt.dest = dest
    stmt.comment = comment
    return stmt

def cmp(lhs: Union[Operand, int], rhs: Union[Operand, int]):
    stmt = CmpStmt()
    stmt.lhs = lhs if isinstance(lhs, Operand) else architecture.operands.c(lhs)
    stmt.rhs = rhs if isinstance(rhs, Operand) else architecture.operands.c(rhs)
    return stmt

def jump(label: str, backwards=True):
    stmt = JumpStmt()
    stmt.destination = architecture.operands.l(label)
    return stmt

def mov(src: Union[Operand, int], dest: Operand, vector: bool, comment:str = None):
    stmt = MovStmt()
    stmt.src = src if isinstance(src, Operand) else architecture.operands.c(src)
    stmt.dest = dest
    stmt.comment = comment
    if vector:
        stmt.aligned = True
        stmt.typ = AsmType.f64x8
    else:
        stmt.aligned = False
        stmt.typ = AsmType.i64
    return stmt

def lea(src: Register, dest: Operand, offset: int, comment:str = None):
    stmt = LeaStmt()
    stmt.src = src
    stmt.dest = dest
    stmt.offset = offset
    stmt.comment = comment
    return stmt

def ld(src: Union[Operand, int], dest: Operand, vector: bool, comment:str = None, dest2: Operand = None, pred: Register = None, is_B: bool = False, scalar_offs: bool = False, add_reg: AsmType.i64 = None):
    stmt = LoadStmt()
    stmt.src = src if isinstance(src, Operand) else architecture.operands.c(src)
    stmt.dest = dest
    stmt.dest2 = dest2
    stmt.comment = comment
    # used in arm_sve:
    stmt.pred = pred
    stmt.is_B = is_B
    stmt.scalar_offs = scalar_offs
    stmt.add_reg = add_reg

    if vector:
        stmt.aligned = True
        stmt.typ = AsmType.f64x8
    else:
        stmt.aligned = False
        stmt.typ = AsmType.i64
    return stmt

def st(src: Union[Operand, int], dest: Operand, vector: bool, comment:str = None, src2: Operand = None, pred: Register = None, scalar_offs: bool = False, add_reg: AsmType.i64 = None):
    stmt = StoreStmt()
    stmt.src = src if isinstance(src, Operand) else architecture.operands.c(src)
    stmt.src2 = src2
    stmt.dest = dest
    stmt.comment = comment
    # used in arm_sve:
    stmt.pred = pred
    stmt.scalar_offs = scalar_offs
    stmt.add_reg = add_reg

    if vector:
        stmt.aligned = True
        stmt.typ = AsmType.f64x8
    else:
        stmt.aligned = False
        stmt.typ = AsmType.i64
    return stmt

def prefetch(dest: Operand, comment:str = None):
    stmt = PrefetchStmt()
    stmt.dest = dest
    stmt.comment = comment
    return stmt

def data(value: Union[Operand, int], asmType=AsmType.i64):
    stmt = DataStmt()
    stmt.value = value if isinstance(value, Operand) else architecture.operands.c(value)
    stmt.asmType = asmType
    return stmt


# Fluent interface
class BlockBuilder(Block):

    def __init__(self, description: str, parent: "BlockBuilder" = None) -> None:
        self.parent = parent
        self.comment = description
        self.contents = []

    def add(self, stmt: AsmStmt):
        self.contents.append(stmt)
        return self

    def open(self, description: str):
        b = BlockBuilder(description, parent=self)
        self.contents.append(b)
        return b

    def close(self):
        return self.parent

    def body(self, *args):
        self.contents = list(args)
        return self



# S-expression interface
def block(description: str, *args: AsmStmt):
    b = BlockBuilder(description)
    b.contents = list(args)
    return b