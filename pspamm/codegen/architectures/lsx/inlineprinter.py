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
            Precision.DOUBLE: "d",
            Precision.SINGLE: "w"
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

    def prefix(self, register):
        return {
            16: "v",
            32: "xv"
        }[register.size()]
    
    def iname(self, root, refreg, bp):
        prefix = self.prefix(refreg)
        suffix = self.bpsuffix if bp else self.psuffix
        return f"{prefix}{root}.{suffix}"
    
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

    def visitFma(self, stmt: FmaStmt):
        b = stmt.bcast_src.ugly
        m = stmt.mult_src.ugly
        a = stmt.add_dest.ugly

        # nmsub is used for c' = -a*b + c
        op = "fnmsub" if stmt.sub else "fmadd"

        # no broadcasting supported inside the instruction (unlike AVX-512)
        s = f"{self.iname(op, stmt.add_dest, False)} {a}, {m}, {b}, {a}"
        self.addLine(s, stmt.comment)

    def visitMul(self, stmt: MulStmt):
        b = stmt.src.ugly
        m = stmt.mult_src.ugly
        a = stmt.dest.ugly
        s = f"{self.iname('fmul', stmt.dest, False)} {a}, {m}, {b}"
        self.addLine(s, stmt.comment)

    def visitBcst(self, stmt: BcstStmt):
        b = stmt.bcast_src.ugly
        a = stmt.dest.ugly
        # check if we broadcast a general register
        if isinstance(stmt.bcast_src, Register):
            instruction = self.iname('replgr2vr', stmt.dest, True)
        else:
            instruction = self.iname('ldrepl', stmt.dest, True)
        s = f"{instruction} {a}, {b}"
        self.addLine(s, stmt.comment)

    def visitAdd(self, stmt: AddStmt):
        if isinstance(stmt.src, Constant) and stmt.src.value == 0:
            # avoid 0 instructions
            return
        if isinstance(stmt.src, Constant) and (stmt.src.value > 2047 or stmt.src.value < -2048):
            # we need an intermediate register here

            # TODO: do not hard-code x5 here, make well-defined
            itmp = "$r5" if stmt.additional is None else stmt.dest.ugly
            tmp = "$r5" if stmt.additional is None else stmt.additional.ugly
            if stmt.src.value < 0:
                addival, luival = self.to_addi(-stmt.src.value)
            else:
                addival, luival = self.to_addi(stmt.src.value)
            self.addLine(f"lu12i.w {itmp}, {luival}", f"Intermediate add: place upper 12 bits of {stmt.src.value}")
            if addival != 0:
                self.addLine(f"addi.d {itmp}, {itmp}, {addival}", f"Intermediate add: place lower 12 bits of {stmt.src.value}")
            if stmt.src.value < 0:
                self.addLine(f"sub.d {stmt.dest.ugly}, {stmt.dest.ugly}, {tmp}", stmt.comment)
            else:
                self.addLine(f"add.d {stmt.dest.ugly}, {stmt.dest.ugly}, {tmp}", stmt.comment)
        else:
            # if stmt.src is a Constant but outside of the above range of value < -2048 or value > 2047
            # we can simply add the Constant to a register
            accumulate = stmt.dest.ugly if stmt.additional is None else stmt.additional.ugly
            self.addLine(f"addi.d {stmt.dest.ugly}, {accumulate}, {stmt.src.ugly}", stmt.comment)

    def visitLabel(self, stmt: LabelStmt):
        s = f"{stmt.label.ugly}:"
        self.addLine(s, stmt.comment)

    def visitCmp(self, stmt: CmpStmt):
        raise NotImplementedError()

    def visitJump(self, stmt: JumpStmt):
        s = f"bne {stmt.cmpreg.ugly}, $r0, {stmt.destination.ugly}"
        self.addLine(s, stmt.comment)

    def visitMov(self, stmt: MovStmt):
        if isinstance(stmt.src, Constant):
            if stmt.dest.typeinfo in [AsmType.f64x2, AsmType.f64x4]:
                assert stmt.src.ugly == '0'
                self.addLine(f"{self.prefix(stmt.dest)}ldi {stmt.dest.ugly}, {stmt.src.ugly}", stmt.comment)
            else:
                if stmt.src.value < 2**12:
                    self.addLine(f"addi.w {stmt.dest.ugly}, $r0, {stmt.src.value}", stmt.comment)
                elif stmt.src.value < 2**32:
                    addival, luival = self.to_addi(stmt.src.value)
                    self.addLine(f"lu12i.w {stmt.dest.ugly}, {luival}", "Intermediate mov: place upper 12 bits")
                    if addival != 0:
                        self.addLine(f"addi.w {stmt.dest.ugly}, {stmt.dest.ugly}, {addival}", stmt.comment)
                else:
                    raise NotImplementedError()
        elif isinstance(stmt.src, Register):
            if stmt.dest.typeinfo in [AsmType.f64x2, AsmType.f64x4]:
                iname = self.iname('replgr2vr', stmt.dest, True)
                self.addLine(f"{iname} {stmt.dest.ugly}, {stmt.src.ugly}", stmt.comment)
            else:
                self.addLine(f"addi.w {stmt.dest.ugly}, {stmt.src.ugly}, 0", stmt.comment)
        else:
            raise NotImplementedError()

    def visitPrefetch(self, stmt: PrefetchStmt):
        if stmt.closeness == "L3":
            hint = "2"
        if stmt.closeness == "L2":
            hint = "1"
        if stmt.closeness == "L1":
            hint = "0"
        # TODO: maybe preldx here?
        s = f"preld {hint}, {stmt.dest.ugly}"
        self.addLine(s, stmt.comment)
    
    def visitLoad(self, stmt: LoadStmt):
        if stmt.dest.typeinfo == AsmType.f64:
            s = f"fl{self.ugly_precision} {stmt.dest.ugly}, {stmt.src.ugly}"
        elif stmt.dest.typeinfo == AsmType.i64:
            s = f"ld.d {stmt.dest.ugly}, {stmt.src.ugly}"
        elif stmt.dest.typeinfo in [AsmType.f64x2, AsmType.f64x4] and stmt.aligned:
            instr = f'{self.prefix(stmt.dest)}ld'
            s = f"{instr} {stmt.dest.ugly}, {stmt.src.ugly}"
        else:
            raise NotImplementedError()
        self.addLine(s, stmt.comment)

    def visitStore(self, stmt: StoreStmt):
        if stmt.src.typeinfo == AsmType.f64:
            s = f"fs{self.ugly_precision} {stmt.src.ugly}, {stmt.dest.ugly}"
        elif stmt.src.typeinfo == AsmType.i64:
            s = f"st.d {stmt.src.ugly}, {stmt.dest.ugly}"
        elif stmt.src.typeinfo in [AsmType.f64x2, AsmType.f64x4] and stmt.aligned:
            instr = f'{self.prefix(stmt.src)}st'
            s = f"{instr} {stmt.src.ugly}, {stmt.dest.ugly}"
        else:
            raise NotImplementedError()
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
