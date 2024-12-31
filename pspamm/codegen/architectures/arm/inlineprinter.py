from typing import List
from pspamm.codegen.ast import *
from pspamm.codegen.visitor import Visitor
from pspamm.codegen.operands import *
from pspamm.codegen.precision import *


class InlinePrinter(Visitor):

    show_comments = True
    indent = "  "
    depth = 0
    lmargin = 0
    rmargin = 60
    vpadding = False
    output = None
    stack = None


    def __init__(self, precision: Precision):
        self.output = []
        self.stack = []
        self.precision = precision
        assert precision in (Precision.HALF, Precision.SINGLE, Precision.DOUBLE)

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

        op = "s" if stmt.sub else "a"
        if stmt.bcast is not None:
            s = f"fml{op} {a}, {m}, {b}[{stmt.bcast}]"
        else:
            s = f"fml{op} {a}, {m}, {b}"
        self.addLine(s, stmt.comment)

    def visitMul(self, stmt: MulStmt):
        b = stmt.src.ugly
        m = stmt.mult_src.ugly
        a = stmt.dest.ugly
        s = f"fmul {a}, {m}, {b}"
        self.addLine(s, stmt.comment)

    def visitBcst(self, stmt: BcstStmt):
        b = stmt.bcast_src.ugly if self.precision == Precision.DOUBLE else stmt.bcast_src.ugly_b32
        a = stmt.dest.ugly
        s = f"dup {a}, {b}"
        self.addLine(s, stmt.comment)

    def visitAdd(self, stmt: AddStmt):
        if isinstance(stmt.src, Constant) and stmt.src.value == 0:
            # avoid 0 instructions
            return
        if isinstance(stmt.src, Constant) and (stmt.src.value > 4095 or stmt.src.value < -4095):
            if (stmt.src.value >> 16) & 0xFFFF > 0 and stmt.src.value < 0:
                s = "mov x11, #-1"
                val1 = (stmt.src.value) & 0xFFFF
                s1 = f"movk x11, #{val1}"
                val2 = ((stmt.src.value >> 16) & 0xFFFF)
                s2 = f"movk x11, #{val2}, lsl #16"

                self.addLine(s, "")
                self.addLine(s1, "load lower 16 bit of immediate that requires more than 16 bit")
                self.addLine(s2, "load upper 16 bit of immediate that requires more than 16 bit")

            elif (stmt.src.value >> 16) & 0xFFFF:
                val1 = (stmt.src.value) & 0xFFFF
                s1 = f"mov x11, #{val1}"
                val2 = ((stmt.src.value >> 16) & 0xFFFF)
                s2 = f"movk x11, #{val2}, lsl #16"
                self.addLine(s1, "load lower 16 bit of immediate that requires more than 16 bit")
                self.addLine(s2, "load upper 16 bit of immediate that requires more than 16 bit")
            else:
                s = f"mov x11, {stmt.src.ugly}"
                self.addLine(s, "load lower 16 bit of immediate ")

            if stmt.dest.ugly != "x11":
                s = f"add {stmt.dest.ugly}, {stmt.dest.ugly}, x11"
                self.addLine(s, stmt.comment)
            if stmt.additional is not None:
                s = f"add {stmt.dest.ugly}, {stmt.dest.ugly}, {stmt.additional.ugly}"
                self.addLine(s, stmt.comment)
        else:
            if stmt.additional is not None:
                s = f"add {stmt.dest.ugly}, {stmt.additional.ugly}, {stmt.src.ugly}"
            else:
                s = f"add {stmt.dest.ugly}, {stmt.dest.ugly}, {stmt.src.ugly}"
            self.addLine(s, stmt.comment)

    def visitLabel(self, stmt: LabelStmt):
        s = f"{stmt.label.ugly}:"
        self.addLine(s, stmt.comment)

    def visitCmp(self, stmt: CmpStmt):
        s = f"cmp {stmt.rhs.ugly}, {stmt.lhs.ugly}"
        self.addLine(s, stmt.comment)

    def visitJump(self, stmt: JumpStmt):
        s = f"b.lo {stmt.destination.ugly}"
        self.addLine(s, stmt.comment)

    def visitMov(self, stmt: MovStmt):
        if isinstance(stmt.src, Label):
            src_str = "#" + stmt.src.ugly
        else:
            src_str = stmt.src.ugly
        if stmt.typ == AsmType.f64x8:
            s = f"fmov {stmt.dest.ugly_scalar_1d}, {src_str}"
        else:
            s = f"mov {stmt.dest.ugly}, {src_str}"
        self.addLine(s, stmt.comment)


    def visitLoad(self, stmt: LoadStmt):
        if isinstance(stmt.src, Label):
            src_str = "#" + stmt.src.ugly
        else:
            src_str = stmt.src.ugly

        if stmt.typ == AsmType.i64:
            s = f"ldr {stmt.dest.ugly}, {src_str}"
        elif stmt.typ == AsmType.f64x8 and stmt.aligned:
            if stmt.dest2 is not None:
                s = f"ldp {stmt.dest.ugly_scalar}, {stmt.dest2.ugly_scalar}, {src_str}"
            else:
                s = f"ldr {stmt.dest.ugly_scalar}, {src_str}"
        else:
            raise NotImplementedError()
        self.addLine(s, stmt.comment)


    def visitStore(self, stmt: StoreStmt):
        if isinstance(stmt.src, Label):
            src_str = "#" + stmt.src.ugly
        else:
            src_str = stmt.src.ugly

        if stmt.typ == AsmType.i64:
            s = f"str {src_str}, {stmt.dest.ugly}"
        elif stmt.typ == AsmType.f64x8 and stmt.aligned:
            if stmt.src2 is not None:
                s = f"stp {stmt.src.ugly_scalar}, {stmt.src2.ugly_scalar}, {stmt.dest.ugly}"
            else:
                s = f"str {stmt.src.ugly_scalar}, {stmt.dest.ugly}"
        else:
            raise NotImplementedError()
        self.addLine(s, stmt.comment)


    def visitBlock(self, block: Block):
        self.stack.append(block)
        self.depth += 1
        if self.show_comments:
            self.addLine(None, block.comment)
        for stmt in block.contents:
            stmt.accept(self)
        self.depth -= 1
        self.stack.pop()


def render(s: AsmStmt):
    p = InlinePrinter()
    s.accept(p)
    return "\n".join(p.output)
