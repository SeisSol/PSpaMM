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
        self.ugly_precision = {
            Precision.DOUBLE: "d",
            Precision.SINGLE: "w",
            Precision.HALF: "h",
            Precision.BFLOAT16: "h",
        }[self.precision]

        assert precision in (Precision.BFLOAT16, Precision.HALF, Precision.SINGLE, Precision.DOUBLE)

    def to_addi(self, value):
        ADDILENGTH = 12
        ADDIBLOCK = (1 << ADDILENGTH) - 1
        ADDISBLOCK = (1 << (ADDILENGTH - 1)) - 1

        addipart = value & ADDIBLOCK
        luipart = value >> ADDILENGTH

        if addipart >= ADDISBLOCK:
            addipart = addipart - (1 << ADDILENGTH)
            luipart += 1
        return addipart, luipart

    def show(self):
        print("\n".join(self.output))

    def addLine(self, stmt: str, comment: str):
        line = " " * self.lmargin + self.indent * self.depth

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
        p = self.p_string(stmt.pred)

        op = "fnmsac" if stmt.sub else "fmacc"

        if stmt.bcast or stmt.bcast_src.typeinfo == AsmType.f64:
            s = f"v{op}.vf {a}, {b}, {m}{p}"
        elif stmt.mult_src.typeinfo == AsmType.f64:
            # TODO: remove this hotfix (?)
            s = f"v{op}.vf {a}, {m}, {b}{p}"
        else:
            s = f"v{op}.vv {a}, {b}, {m}{p}"

        self.addLine(s, stmt.comment)

    def visitMul(self, stmt: MulStmt):
        b = stmt.src.ugly
        m = stmt.mult_src.ugly
        a = stmt.dest.ugly

        p = self.p_string(stmt.pred)
        if stmt.mult_src.typeinfo == AsmType.f64:
            s = f"vfmul.vf {a}, {b}, {m}{p}"
        else:
            s = f"vfmul.vv {a}, {b}, {m}{p}"
        self.addLine(s, stmt.comment)

    def visitBcst(self, stmt: BcstStmt):
        if stmt.dest.typeinfo == AsmType.f64:
            s = f"fmv.s {stmt.dest.ugly}, {stmt.bcast_src.ugly}"
        else:
            if isinstance(stmt.bcast_src, Constant):
                s = f"vfmv.v.i {stmt.dest.ugly}, {stmt.bcast_src.ugly}"
            else:
                s = f"vfmv.v.f {stmt.dest.ugly}, {stmt.bcast_src.ugly}"
        self.addLine(s, stmt.comment)

    def visitAdd(self, stmt: AddStmt):
        if isinstance(stmt.src, Constant) and stmt.src.value == 0:
            # avoid 0 instructions
            return
        if isinstance(stmt.src, Constant) and (stmt.src.value > 2047 or stmt.src.value < -2048):
            # we need an intermediate register here

            # TODO: do not hard-code x5 here
            tmp = "x5" if stmt.additional is None else stmt.additional.ugly
            if stmt.src.value < 0:
                addival, luival = self.to_addi(-stmt.src.value)
            else:
                addival, luival = self.to_addi(stmt.src.value)
            self.addLine(f"lui {tmp}, {luival}", f"Intermediate add: place upper 12 bits of {stmt.src.value}")
            if addival != 0:
                self.addLine(f"addi {tmp}, {tmp}, {addival}", f"Intermediate add: place lower 12 bits of {stmt.src.value}")
            if stmt.src.value < 0:
                self.addLine(f"sub {stmt.dest.ugly}, {stmt.dest.ugly}, {tmp}", stmt.comment)
            else:
                self.addLine(f"add {stmt.dest.ugly}, {stmt.dest.ugly}, {tmp}", stmt.comment)
        else:
            # if stmt.src is a Constant but outside of the above range of value < -2048 or value > 2047
            # we can simply add the Constant to a register
            accumulate = stmt.dest.ugly if stmt.additional is None else stmt.additional.ugly
            self.addLine(f"addi {stmt.dest.ugly}, {accumulate}, {stmt.src.ugly}", stmt.comment)

    def visitLabel(self, stmt: LabelStmt):
        s = f"{stmt.label.ugly}:"
        self.addLine(s, stmt.comment)

    def visitCmp(self, stmt: CmpStmt):
        raise NotImplementedError()

    def visitJump(self, stmt: JumpStmt):
        s = f"bne {stmt.cmpreg.ugly}, x0, {stmt.destination.ugly}"
        self.addLine(s, stmt.comment)

    def visitMov(self, stmt: MovStmt):
        if isinstance(stmt.src, Constant):
            if stmt.dest.typeinfo == AsmType.f64x8:
                self.addLine(f"vmv.v.i {stmt.dest.ugly}, {stmt.src.ugly}", stmt.comment)
            else:
                if stmt.src.value < 2**12:
                    self.addLine(f"addi {stmt.dest.ugly}, x0, {stmt.src.value}", stmt.comment)
                elif stmt.src.value < 2**32:
                    addival, luival = self.to_addi(stmt.src.value)
                    self.addLine(f"lui {stmt.dest.ugly}, {luival}", "Intermediate mov: place upper 12 bits")
                    if addival != 0:
                        self.addLine(f"addi {stmt.dest.ugly}, {stmt.dest.ugly}, {addival}", stmt.comment)
                else:
                    raise NotImplementedError()
        elif isinstance(stmt.src, Register):
            if stmt.dest.typeinfo == AsmType.f64x8:
                self.addLine(f"vmv.v.v {stmt.dest.ugly}, {stmt.src.ugly}", stmt.comment)
            else:
                self.addLine(f"addi {stmt.dest.ugly}, {stmt.src.ugly}, 0", stmt.comment)
        else:
            raise NotImplementedError()

    def visitLoad(self, stmt: LoadStmt):
        p = self.p_string(stmt.pred)
        prec = self.precision.size() * 8

        if stmt.dest.typeinfo == AsmType.f64:
            s = f"fl{self.ugly_precision} {stmt.dest.ugly}, {stmt.src.ugly}"
        elif stmt.dest.typeinfo == AsmType.i64:
            s = f"ld {stmt.dest.ugly}, {stmt.src.ugly}"
        elif stmt.dest.typeinfo == AsmType.f64x8 and stmt.aligned:
            s = f"vle{prec}.v {stmt.dest.ugly}, {stmt.src.ugly}{p}"
        else:
            raise NotImplementedError()
        self.addLine(s, stmt.comment)

    def visitStore(self, stmt: StoreStmt):
        p = self.p_string(stmt.pred)
        prec = self.precision.size() * 8

        if stmt.src.typeinfo == AsmType.f64:
            s = f"fs{self.ugly_precision} {stmt.src.ugly}, {stmt.dest.ugly}"
        elif stmt.src.typeinfo == AsmType.i64:
            s = f"sd {stmt.src.ugly}, {stmt.dest.ugly}"
        elif stmt.src.typeinfo == AsmType.f64x8 and stmt.aligned:
            s = f"vse{prec}.v {stmt.src.ugly}, {stmt.dest.ugly}{p}"
        else:
            raise NotImplementedError()
        self.addLine(s, stmt.comment)

    def visitPrefetch(self, stmt: PrefetchStmt):
        pass
    
    def visitRVSetVLStmt(self, stmt: RVSetVLStmt):
        opcode = 'setivli' if isinstance(stmt.requested, Constant) else 'setvli'
        s = f"v{opcode} {stmt.actual.ugly}, {stmt.requested.ugly}, e{self.precision.size() * 8}"
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

    def p_string(self, predicate: Register):
        # returns "pk{/z or /m}, " or an empty string "" with contents in {} being optional
        # at this point the contents are already generated, we simply turn them into a string
        return f', {predicate}' if predicate is not None else ""


def render(s: AsmStmt):
    p = InlinePrinter()
    s.accept(p)
    return "\n".join(p.output)
