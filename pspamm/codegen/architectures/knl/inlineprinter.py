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
        assert precision in (Precision.BFLOAT16, Precision.HALF, Precision.SINGLE, Precision.DOUBLE)
        self.precision = precision
        self.psuffix = {
            Precision.DOUBLE: 'd',
            Precision.SINGLE: 's',
            Precision.HALF: 'h',
            Precision.BFLOAT16: 'h'
        }[precision]
        self.alupsuffix = {
            Precision.DOUBLE: 'pd',
            Precision.SINGLE: 'ps',
            Precision.HALF: 'ph',
            Precision.BFLOAT16: 'nepbf16'
        }[precision]
        self.broadcast_multiplier = {
            Precision.DOUBLE: 2,
            Precision.SINGLE: 4,
            Precision.HALF: 8,
            Precision.BFLOAT16: 8
        }[precision]

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

    def maskformat(self, pred):
        if pred is None:
            return ''
        elif pred.zero:
            return f'{{{pred.register.ugly}}}{{z}}'
        else:
            return f'{{{pred.register.ugly}}}'

    def visitFma(self, stmt: FmaStmt):
        mask = self.maskformat(stmt.pred)
        b = stmt.bcast_src.ugly
        m = stmt.mult_src.ugly
        a = stmt.add_dest.ugly
        regsize = stmt.add_dest.size() // 16
        extent = regsize * self.broadcast_multiplier
        if stmt.bcast:
            s = f"vfmadd231{self.alupsuffix} {b}%{{1to{extent}%}} {mask}, {m}, {a}"
        else:
            if stmt.mult_src.typeinfo == AsmType.i64:
                # in this case, m is a Register that points to alpha; manually format to be a memory address
                s = f"vfmadd231{self.alupsuffix} 0({m})%{{1to{extent}%}} {mask}, {b}, {a}"
            else:
                s = f"vfmadd231{self.alupsuffix} {b} {mask}, {m}, {a}"
        self.addLine(s, stmt.comment)

    def visitMul(self, stmt: MulStmt):
        mask = self.maskformat(stmt.pred)
        b = stmt.src.ugly
        m = stmt.mult_src.ugly
        a = stmt.dest.ugly
        regsize = stmt.dest.size() // 16
        extent = regsize * self.broadcast_multiplier
        if stmt.mult_src.typeinfo == AsmType.i64:
            # in this case, m is a Register that points to alpha/beta; manually format to be a memory address
            s = f"vmul{self.alupsuffix} 0({m})%{{1to{extent}%}} {mask}, {b}, {a}"
        else:
            s = f"vmul{self.alupsuffix} {b} {mask}, {m}, {a}"
        self.addLine(s, stmt.comment)

    def visitBcst(self, stmt: BcstStmt):
        mask = self.maskformat(stmt.pred)
        b = stmt.bcast_src.ugly
        a = stmt.dest.ugly
        regsize = stmt.dest.size()
        if self.precision == Precision.HALF or self.precision == Precision.BFLOAT16:
            instruction = 'vpbroadcastw'
        elif self.precision == Precision.DOUBLE and regsize == 16:
            instruction = 'vmovddup'
        else:
            instruction = f"vbroadcasts{self.psuffix}"
        s = f"{instruction} {b} {mask}, {a}"
        self.addLine(s, stmt.comment)

    def visitAdd(self, stmt: AddStmt):
        # only used for scalar addition right now
        s = f"addq {stmt.src.ugly}, {stmt.dest.ugly}"
        self.addLine(s, stmt.comment)

    def visitLabel(self, stmt: LabelStmt):
        s = f"{stmt.label.ugly}:"
        self.addLine(s, stmt.comment)

    def visitCmp(self, stmt: CmpStmt):
        mask = self.maskformat(stmt.pred)
        s = f"cmp {stmt.lhs.ugly} {mask}, {stmt.rhs.ugly}"
        self.addLine(s, stmt.comment)

    def visitJump(self, stmt: JumpStmt):
        s = f"jl {stmt.destination.ugly}"
        self.addLine(s, stmt.comment)

    def visitMov(self, stmt: MovStmt):
        mask = self.maskformat(stmt.pred)

        if isinstance(stmt.src, Label):
            src_str = "$" + stmt.src.ugly
        else:
            src_str = stmt.src.ugly

        if stmt.typ == AsmType.i64:
            assert(stmt.pred == None)
            if stmt.dest.ugly[0] == 'k':
                s = f"kmovq {src_str}, {stmt.dest.ugly}"
            else:
                s = f"movq {src_str}, {stmt.dest.ugly}"
        elif stmt.typ == AsmType.f64x8 and stmt.aligned:
            if isinstance(stmt.src, Constant) and stmt.src.value == 0:
                s = f"vpxord {stmt.dest.ugly} {mask}, {stmt.dest.ugly}, {stmt.dest.ugly}"
            else:
                s = f"vmovupd {src_str} {mask}, {stmt.dest.ugly}"
        else:
            raise NotImplementedError()
        self.addLine(s, stmt.comment)

    def visitLea(self, stmt: LeaStmt):
        mask = self.maskformat(stmt.pred)
        s = f"leaq {stmt.offset}({stmt.src.ugly}) {mask}, {stmt.dest.ugly}"
        self.addLine(s, stmt.comment)

    def visitPrefetch(self, stmt: PrefetchStmt):
        s = f"prefetcht1 {stmt.dest.ugly}"
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
