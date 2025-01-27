from typing import Union

from pspamm.codegen.ast import *
from pspamm.codegen.operands import *

import pspamm.architecture

# Convenient statement constructors
def add(src: Union[Operand, int], dest: Register, comment: str = None, additional: Register = None):
    stmt = AddStmt()
    stmt.src = src if isinstance(src, Operand) else pspamm.architecture.operands.c(src)
    stmt.dest = dest
    stmt.comment = comment
    stmt.additional = additional
    return stmt

def label(name: str):
    stmt = LabelStmt()
    stmt.label = pspamm.architecture.operands.l(name)
    return stmt

def fma(bcast_src: Register, mult_src: Register, add_dest: Register, comment: str = None, bcast: Union[int, None] = None, pred: Register = None, sub=False):
    stmt = FmaStmt()
    stmt.bcast_src = bcast_src
    stmt.mult_src = mult_src
    stmt.add_dest = add_dest
    stmt.comment = comment
    stmt.bcast = bcast
    # used in arm_sve:
    stmt.pred = pred
    stmt.sub = sub
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
    stmt.lhs = lhs if isinstance(lhs, Operand) else pspamm.architecture.operands.c(lhs)
    stmt.rhs = rhs if isinstance(rhs, Operand) else pspamm.architecture.operands.c(rhs)
    return stmt

def jump(label: str, cmpreg = None, backwards=True):
    stmt = JumpStmt()
    stmt.destination = pspamm.architecture.operands.l(label)
    stmt.cmpreg = cmpreg
    return stmt

def mov(src: Union[Operand, int], dest: Operand, vector: bool, comment:str = None, pred = None, expand=None, temp=None):
    stmt = MovStmt()
    stmt.src = src if isinstance(src, Operand) else pspamm.architecture.operands.c(src)
    stmt.dest = dest
    stmt.comment = comment
    stmt.pred = pred
    stmt.expand = expand
    stmt.temp = temp
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

def ld(src: Union[Operand, int], dest: Operand, vector: bool, comment:str = None, dest2: Operand = None, pred: Register = None, is_B: bool = False, scalar_offs: bool = False, add_reg: AsmType.i64 = None, sub128: bool = False, expand=None, dest3: Operand = None, dest4: Operand = None):
    stmt = LoadStmt()
    stmt.src = src if isinstance(src, Operand) else pspamm.architecture.operands.c(src)
    stmt.dest = dest
    stmt.dest2 = dest2
    stmt.dest3 = dest3
    stmt.dest4 = dest4
    stmt.comment = comment
    # used in arm_sve:
    stmt.pred = pred
    stmt.is_B = is_B
    stmt.scalar_offs = scalar_offs
    stmt.add_reg = add_reg
    stmt.expand = expand

    if vector:
        stmt.aligned = True
        if sub128:
            stmt.typ = AsmType.f64x2
        else:
            stmt.typ = AsmType.f64x8
    else:
        stmt.aligned = False
        stmt.typ = AsmType.i64
    return stmt

def st(src: Union[Operand, int], dest: Operand, vector: bool, comment:str = None, src2: Operand = None, pred: Register = None, scalar_offs: bool = False, add_reg: AsmType.i64 = None, expand=None, src3: Operand=None, src4: Operand=None):
    stmt = StoreStmt()
    stmt.src = src if isinstance(src, Operand) else pspamm.architecture.operands.c(src)
    stmt.src2 = src2
    stmt.src3 = src3
    stmt.src4 = src4
    stmt.dest = dest
    stmt.comment = comment
    # used in arm_sve:
    stmt.pred = pred
    stmt.scalar_offs = scalar_offs
    stmt.add_reg = add_reg
    stmt.expand = expand

    if vector:
        stmt.aligned = True
        stmt.typ = AsmType.f64x8
    else:
        stmt.aligned = False
        stmt.typ = AsmType.i64
    return stmt

def prefetch(dest: Operand, comment:str = None, pred: Register = None, precision: str = None, access_type: str = None, closeness: str = None, temporality: str = None):
    stmt = PrefetchStmt()
    stmt.dest = dest
    stmt.comment = comment
    # used in arm_sve:
    stmt.pred = pred
    stmt.precision = precision
    stmt.access_type = access_type
    stmt.closeness = closeness
    stmt.temporality = temporality
    return stmt

def data(value: Union[Operand, int], asmType=AsmType.i64):
    stmt = DataStmt()
    stmt.value = value if isinstance(value, Operand) else pspamm.architecture.operands.c(value)
    stmt.asmType = asmType
    return stmt

def rvsetvl(actual: Register, requested: Union[Register, int]):
    stmt = RVSetVLStmt()
    stmt.actual = actual
    stmt.requested = requested if isinstance(requested, Operand) else pspamm.architecture.operands.c(requested)
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
