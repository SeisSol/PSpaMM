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
        assert precision in (Precision.SINGLE, Precision.DOUBLE)
        self.precision = precision
        self.psuffix = {
            Precision.DOUBLE: "d",
            Precision.SINGLE: "s"
        }[precision]
        self.bpsuffix = {
            Precision.DOUBLE: "q",
            Precision.SINGLE: "d"
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

    def visitFma(self, stmt: FmaStmt):
        b = stmt.bcast_src.ugly
        m = stmt.mult_src.ugly
        a = stmt.add_dest.ugly

        op = "nmadd" if stmt.sub else "madd"

        # no broadcasting supported inside the instruction (unlike AVX-512)
        s = f"vf{op}231p{self.psuffix} {b}, {m}, {a}"
        self.addLine(s, stmt.comment)

    def visitMul(self, stmt: MulStmt):
        b = stmt.src.ugly
        m = stmt.mult_src.ugly
        a = stmt.dest.ugly
        s = f"vmulp{self.psuffix} {b}, {m}, {a}"
        self.addLine(s, stmt.comment)

    def visitBcst(self, stmt: BcstStmt):
        b = stmt.bcast_src.ugly
        a = stmt.dest.ugly
        # check if we broadcast a general register
        if isinstance(stmt.bcast_src, Register):
            # reformat bcast_src to be a memory address
            b = f"0({b})"
        regsize = stmt.dest.size()
        instruction = "vmovddup" if self.precision == Precision.DOUBLE and regsize == 16 else f"vbroadcasts{self.psuffix}"
        s = f"{instruction} {b}, {a}"
        self.addLine(s, stmt.comment)

    def visitAdd(self, stmt: AddStmt):
        if isinstance(stmt.src, Constant) and stmt.src.value == 0:
            # avoid 0 instructions
            return
        s = f"addq {stmt.src.ugly}, {stmt.dest.ugly}"
        self.addLine(s, stmt.comment)

    def visitLabel(self, stmt: LabelStmt):
        self.addLine('.align 16', 'Align label')
        s = f"{stmt.label.ugly}:"
        self.addLine(s, stmt.comment)

    def visitCmp(self, stmt: CmpStmt):
        s = f"cmp {stmt.lhs.ugly}, {stmt.rhs.ugly}"
        self.addLine(s, stmt.comment)

    def visitJump(self, stmt: JumpStmt):
        s = f"jl {stmt.destination.ugly}"
        self.addLine(s, stmt.comment)

    def visitMov(self, stmt: MovStmt):
        if isinstance(stmt.src, Label):
            src_str = "$" + stmt.src.ugly
        else:
            src_str = stmt.src.ugly

        if stmt.typ == AsmType.i64:
            s = f"movq {src_str}, {stmt.dest.ugly}"
            self.addLine(s, stmt.comment)
        elif stmt.typ == AsmType.f64x8 and stmt.aligned:
            if isinstance(stmt.src, Constant) and stmt.src.value == 0:
                s = f"vxorps {stmt.dest.ugly_xmm}, {stmt.dest.ugly_xmm}, {stmt.dest.ugly_xmm}"
                self.addLine(s, stmt.comment)
            elif stmt.pred is not None:
                self.addLine(f"vpxor {stmt.dest.ugly}, {stmt.dest.ugly}, {stmt.dest.ugly}", "")
                self.addLine(f"vpblendd {src_str}, {stmt.dest.ugly}, {stmt.pred}, {stmt.dest.ugly}", "")
            elif stmt.expand:
                # TODO: unfinished
                self.addLine(f"vpxor {stmt.temp.ugly}, {stmt.temp.ugly}, {stmt.temp.ugly}")
                regsize = stmt.dest.size()
                if self.precision == Precision.SINGLE and regsize == 32:
                    self.addLine(f"vmovq {stmt.pred.ugly}, {stmt.dest.ugly_xmm}", "")
                    self.addLine(f"vpmovzxb{self.bpsuffix} {stmt.dest.ugly_xmm}, {stmt.dest.ugly}", "")
                    self.addLine(f"vpermd {src_str}, {stmt.dest.ugly}, {stmt.dest.ugly}", "")
                elif regsize == 16:
                    self.addLine(f"vpermilps {src_str}, MISSING_PREDICATE, {stmt.dest.ugly}", "")
                elif self.precision == Precision.DOUBLE:
                    self.addLine(f"vpermpd {src_str}, MISSING_PREDICATE, {stmt.dest.ugly}", "")
                self.addLine(f"vpblendd {stmt.temp.ugly}, {stmt.dest.ugly}, MISSING_PREDICATE, {stmt.dest.ugly}", "")
            else:
                s = f"vmovup{self.psuffix} {src_str}, {stmt.dest.ugly}"
                self.addLine(s, stmt.comment)
        else:
            raise NotImplementedError()

    def visitLea(self, stmt: LeaStmt):
        s = f"leaq {stmt.offset}({stmt.src.ugly}), {stmt.dest.ugly}"
        self.addLine(s, stmt.comment)

    def visitPrefetch(self, stmt: PrefetchStmt):
        if stmt.closeness == "L3":
            suffix = "t2"
        if stmt.closeness == "L2":
            suffix = "t1"
        if stmt.closeness == "L1":
            suffix = "t0"
        s = f"prefetch{suffix} {stmt.dest.ugly}"
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
