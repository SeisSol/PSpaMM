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
        s = "fmla {}, {}{}, {}".format(a, p, m, b)

        self.addLine(s, stmt.comment)

    def visitMul(self, stmt: MulStmt):
        b = stmt.src.ugly
        m = stmt.mult_src.ugly
        a = stmt.dest.ugly

        if a != b:
            s1 = "movprfx {}, {}".format(a.split(".")[0], b.split(".")[0])
            self.addLine(s1, "move {} into {}".format(b, a))
            b = a

        p = self.p_string(stmt.pred)
        s = "fmul {}, {}{}, {}".format(a, p, b, m)
        self.addLine(s, stmt.comment)

    def visitBcst(self, stmt: BcstStmt):
        # Used to broadcast a scalar register into a vector register
        b = stmt.bcast_src.ugly
        a = stmt.dest.ugly
        # make sure the src register is a W register when using single precision
        if self.precision == Precision.SINGLE:
            b = "w" + b[1:]
        s = "dup {}, {}".format(a, b)
        self.addLine(s, stmt.comment)

    def visitAdd(self, stmt: AddStmt):
        if isinstance(stmt.src, Constant) and (stmt.src.value > 4095 or stmt.src.value < -4095):
            # This condition is probably related to immediate values being restricted to 12 bits for add instructions
            # https://developer.arm.com/documentation/dui0802/a/A64-General-Instructions/ADD--immediate-
            # https://developer.arm.com/documentation/ddi0596/2020-12/Base-Instructions/ADD--immediate---Add--immediate--
            if (stmt.src.value >> 16) & 0xFFFF > 0 and stmt.src.value < 0:
                s = "mov x11, #-1"
                s1 = "movk x11, #{}".format((stmt.src.value) & 0xFFFF)
                val = ((stmt.src.value >> 16) & 0xFFFF)
                s2 = "movk x11, #{}, lsl #16".format(val)

                self.addLine(s, "")
                self.addLine(s1, "load lower 16 bit of immediate that requires more than 16 bit")
                self.addLine(s2, "load upper 16 bit of immediate that requires more than 16 bit")
            elif (stmt.src.value >> 16) != 0:
                s1 = "mov x11, #{}".format((stmt.src.value) & 0xFFFF)
                val = ((stmt.src.value >> 16) & 0xFFFF)
                s2 = "movk x11, #{}, lsl #16".format(val)
                self.addLine(s1, "load lower 16 bit of immediate that requires more than 16 bit")
                self.addLine(s2, "load upper 16 bit of immediate that requires more than 16 bit")
            else:
                s = "mov x11, {}".format(stmt.src.ugly)
                self.addLine(s, "load lower 16 bit of immediate ")

            if stmt.dest.ugly != "x11":
                s = "add {}, {}, x11".format(stmt.dest.ugly, stmt.dest.ugly)
                self.addLine(s, stmt.comment)
            if stmt.additional is not None:
                s = "add {}, {}, {}".format(stmt.dest.ugly, stmt.dest.ugly, stmt.additional.ugly)
                self.addLine(s, stmt.comment)
        else:
            # if stmt.src is a Constant but outside of the above range of value < -4095 or value > 4095
            # we can simply add the Constant to a register
            if stmt.additional is not None:
                s = "add {}, {}, {}".format(stmt.dest.ugly, stmt.additional.ugly, stmt.src.ugly)
            else:
                s = "add {}, {}, {}".format(stmt.dest.ugly, stmt.dest.ugly, stmt.src.ugly)
            self.addLine(s, stmt.comment)

    def visitLabel(self, stmt: LabelStmt):
        s = "{}:".format(stmt.label.ugly)
        self.addLine(s, stmt.comment)

    def visitCmp(self, stmt: CmpStmt):
        s = "cmp {}, {}".format(stmt.rhs.ugly, stmt.lhs.ugly)
        self.addLine(s, stmt.comment)

    def visitJump(self, stmt: JumpStmt):
        s = "b.lo {}".format(stmt.destination.ugly)
        self.addLine(s, stmt.comment)

    def visitMov(self, stmt: MovStmt):
        if isinstance(stmt.src, Label):
            src_str = ("#" + stmt.src.ugly).split("_")[0]
        else:
            src_str = stmt.src.ugly
        if stmt.typ == AsmType.f64x8:
            s = "fmov {}, {}".format(stmt.dest.ugly, src_str)
        else:
            s = "mov {}, {}".format(stmt.dest.ugly, src_str)
        self.addLine(s, stmt.comment)

    def visitLoad(self, stmt: LoadStmt):
        if isinstance(stmt.src, Label):
            src_str = "#" + stmt.src.ugly
        elif stmt.src.ugly_offset != "0" and stmt.scalar_offs:
            self.addLine("mov {}, #{}".format(stmt.add_reg.ugly, stmt.src.ugly_offset), "move immediate offset into {}".format(stmt.add_reg.ugly))
            # TODO: adapt ugly_lsl_shift to account for possible single precision instead of double precision
            src_str = "[{}, {}, LSL #{}]".format(stmt.src.ugly_base, stmt.add_reg.ugly, stmt.dest.ugly_lsl_shift)
        else:
            src_str = stmt.src.ugly if not stmt.is_B else stmt.src.ugly_no_vl_scaling

        p = self.p_string(stmt.pred)
        prec = "d" if stmt.dest.ugly_precision == "d" else "w"

        if stmt.typ == AsmType.i64:
            s = "add {}, {}, {}".format(stmt.dest.ugly, stmt.dest.ugly, src_str)
        elif stmt.typ == AsmType.f64x8 and stmt.aligned:
            if stmt.is_B:
                s = "ld1r{} {}, {}{}".format(prec, stmt.dest.ugly, p, src_str)
            else:
                s = "ld1{} {}, {}{}".format(prec, stmt.dest.ugly, p, src_str)
        else:
            raise NotImplementedError()
        self.addLine(s, stmt.comment)

    def visitStore(self, stmt: StoreStmt):
        if isinstance(stmt.src, Label):
            src_str = "#" + stmt.src.ugly
        elif stmt.dest.ugly_offset != "0" and stmt.scalar_offs:
            self.addLine("mov {}, #{}".format(stmt.add_reg.ugly, stmt.dest.ugly_offset),
                         "move immediate offset into {}".format(stmt.add_reg.ugly))
            # TODO: adapt ugly_lsl_shift to account for possible single precision instead of double precision
            dest_str = "[{}, {}, LSL #{}]".format(stmt.dest.ugly_base, stmt.add_reg.ugly, stmt.src.ugly_lsl_shift)
        else:
            dest_str = stmt.dest.ugly

        p = self.p_string(stmt.pred)
        prec = "d" if stmt.src.ugly_precision == "d" else "w"

        if stmt.typ == AsmType.i64:
            s = "add {}, {}, {}".format(stmt.dest.ugly, stmt.dest.ugly, src_str)
        elif stmt.typ == AsmType.f64x8 and stmt.aligned:
            s = "st1{} {}, {}{}".format(prec, stmt.src.ugly, p, dest_str)
        else:
            raise NotImplementedError()
        self.addLine(s, stmt.comment)

    def visitPrefetch(self, stmt: PrefetchStmt):
        # https://stackoverflow.com/questions/37070/what-is-the-meaning-of-non-temporal-memory-accesses-in-x86#:~:text=Data%20referenced%20by%20a%20program,%2C%20is%20often%20non%2Dtemporal.
        cache_level = "L1"  # specify cache level to which we prefetch
        temporality = "KEEP"  # could use "STRM" for non-temporal prefetching if needed
        xn = stmt.dest.ugly_base
        offset = stmt.dest.ugly_offset
        src_string = "[{}, {}, MUL VL]".format(xn, offset)
        p = self.p_string(stmt.pred)
        prec = "d" if stmt.precision == Precision.DOUBLE else "w"
        s = "prf{} P{}{}{}, {}{}".format(prec, stmt.access_type, cache_level, temporality, p.split('/')[0], src_string)
        self.addLine(s, "prefetch from memory")

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
        return predicate.value + ", " if predicate is not None else ""


def render(s: AsmStmt):
    p = InlinePrinter()
    s.accept(p)
    return "\n".join(p.output)
