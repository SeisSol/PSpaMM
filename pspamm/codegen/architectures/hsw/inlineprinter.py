from typing import List
from pspamm.codegen.ast import *
from pspamm.codegen.visitor import Visitor
from pspamm.codegen.operands import *
from pspamm.codegen.precision import *


class InlinePrinter(Visitor):

    show_comments = False
    indent = "  "
    depth = 0
    lmargin = 0
    rmargin = 70
    vpadding = False
    output = None
    stack = None


    def __init__(self, precision: Precision):
        self.output = []
        self.stack = []
        assert precision in [Precision.SINGLE, Precision.DOUBLE]
        self.precision = 'd' if precision == Precision.DOUBLE else 's'

    def show(self):
        print("\n".join(self.output))


    def addLine(self, stmt: str, comment: str):

        line = " "*self.lmargin + self.indent*self.depth

        if stmt is not None and comment is not None and self.show_comments:
            stmt = '"' + stmt + '\\r\\n"'
            line += stmt.ljust(self.rmargin) + "// " + comment

        elif stmt is not None:
            line += '"' + stmt + '\\r\\n"'

        elif stmt is None and comment is not None:
            line += "// " + comment

        self.output.append(line)



    def visitFma(self, stmt: FmaStmt):
        b = stmt.bcast_src.ugly
        m = stmt.mult_src.ugly
        a = stmt.add_dest.ugly

        # no broadcasting supported inside the instruction (unlike AVX-512)
        s = "vfmadd231p{} {}, {}, {}".format(self.precision, b,m,a)
        self.addLine(s, stmt.comment)

    def visitMul(self, stmt: MulStmt):
        b = stmt.src.ugly
        m = stmt.mult_src.ugly
        a = stmt.dest.ugly
        s = "vmulp{} {}, {}, {}".format(self.precision, b,m,a)
        self.addLine(s, stmt.comment)

    def visitBcst(self, stmt: BcstStmt):
        b = stmt.bcast_src.ugly
        a = stmt.dest.ugly
        # check if we broadcast a general register
        if isinstance(stmt.bcast_scr, Regsiter):
            # reformat bcast_src to be a memory address
            b = "0({})".format(b)
        s = "vbroadcasts{} {}, {}".format(self.precision, b, a)
        self.addLine(s, stmt.comment)

    def visitAdd(self, stmt: AddStmt):
        s = "addq {}, {}".format(stmt.src.ugly,stmt.dest.ugly)
        self.addLine(s, stmt.comment)

    def visitLabel(self, stmt: LabelStmt):
        s = "{}:".format(stmt.label.ugly)
        self.addLine(s, stmt.comment)

    def visitCmp(self, stmt: CmpStmt):
        s = "cmp {}, {}".format(stmt.lhs.ugly,stmt.rhs.ugly)
        self.addLine(s, stmt.comment)

    def visitJump(self, stmt: JumpStmt):
        s = "jl {}".format(stmt.destination.ugly)
        self.addLine(s, stmt.comment)

    def visitMov(self, stmt: MovStmt):
        if isinstance(stmt.src, Label):
            src_str = "$" + stmt.src.ugly
        else:
            src_str = stmt.src.ugly

        if stmt.typ == AsmType.i64:
            s = "movq {}, {}".format(src_str,stmt.dest.ugly)
        elif stmt.typ == AsmType.f64x8 and stmt.aligned:
            if isinstance(stmt.src, Constant) and stmt.src.value == 0:
                s = "vpxor {}, {}, {}".format(stmt.dest.ugly,stmt.dest.ugly,stmt.dest.ugly)
            else:
                s = "vmovup{} {}, {}".format(self.precision, src_str,stmt.dest.ugly)
        else:
            raise NotImplementedError()
        self.addLine(s, stmt.comment)

    def visitLea(self, stmt: LeaStmt):
        s = "leaq {}({}), {}".format(stmt.offset,stmt.src.ugly,stmt.dest.ugly)
        self.addLine(s, stmt.comment)

    def visitPrefetch(self, stmt: PrefetchStmt):
        s = "prefetcht1 {}".format(stmt.dest.ugly)
        self.addLine(s, stmt.comment)

    def visitBlock(self, block: Block):
        self.stack.append(block)
        self.depth += 1
        if self.show_comments and block.comment != '':
            self.addLine(None, block.comment)
        for stmt in block.contents:
            stmt.accept(self)
        self.depth -= 1
        self.stack.pop()


def render(s: AsmStmt):
    p = InlinePrinter()
    s.accept(p)
    return "\n".join(p.output)
