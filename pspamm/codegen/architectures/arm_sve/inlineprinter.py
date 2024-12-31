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
        self.ugly_precision ={
            Precision.DOUBLE: "d",
            Precision.SINGLE: "w",
            Precision.HALF: "h",
            Precision.BFLOAT16: "h",
        }[self.precision]

        assert precision in (Precision.BFLOAT16, Precision.HALF, Precision.SINGLE, Precision.DOUBLE)

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

        op = "s" if stmt.sub else "a"
        if stmt.bcast is not None:
            # NOTE: ignores predicate
            s = f"fml{op} {a}, {m}, {b}[{stmt.bcast}]"
        else:
            s = f"fml{op} {a}, {p}{m}, {b}"

        self.addLine(s, stmt.comment)

    def visitMul(self, stmt: MulStmt):
        b = stmt.src.ugly
        m = stmt.mult_src.ugly
        a = stmt.dest.ugly

        if a != b:
            s1 = f"movprfx {a.split('.')[0]}, {b.split('.')[0]}"
            self.addLine(s1, "move {} into {}".format(b, a))
            b = a

        p = self.p_string(stmt.pred)
        s = f"fmul {a}, {p}{b}, {m}"
        self.addLine(s, stmt.comment)

    def visitBcst(self, stmt: BcstStmt):
        # Used to broadcast a scalar register into a vector register
        b = stmt.bcast_src.ugly
        a = stmt.dest.ugly
        # make sure the src register is a W register when using single/half precision
        if self.precision.size() <= 4:
            b = "w" + b[1:]
        s = f"dup {a}, {b}"
        self.addLine(s, stmt.comment)

    def visitAdd(self, stmt: AddStmt):
        if isinstance(stmt.src, Constant) and stmt.src.value == 0:
            # avoid 0 instructions
            return
        if isinstance(stmt.src, Constant) and (stmt.src.value > 4095 or stmt.src.value < -4095):            
            # This condition is probably related to immediate values being restricted to 12 bits for add instructions
            # https://developer.arm.com/documentation/dui0802/a/A64-General-Instructions/ADD--immediate-
            # https://developer.arm.com/documentation/ddi0596/2020-12/Base-Instructions/ADD--immediate---Add--immediate--
            if (stmt.src.value >> 16) & 0xFFFF > 0 and stmt.src.value < 0:
                s = "mov x11, #-1"
                val1 = (stmt.src.value) & 0xFFFF
                s1 = f"movk x11, #{val1}"
                val2 = ((stmt.src.value >> 16) & 0xFFFF)
                s2 = f"movk x11, #{val2}, lsl #16"

                self.addLine(s, "")
                self.addLine(s1, "load lower 16 bit of immediate that requires more than 16 bit")
                self.addLine(s2, "load upper 16 bit of immediate that requires more than 16 bit")
            elif (stmt.src.value >> 16) != 0:
                val1 = (stmt.src.value) & 0xFFFF
                s1 = "mov x11, #{val1}"
                val2 = ((stmt.src.value >> 16) & 0xFFFF)
                s2 = "movk x11, #{val2}, lsl #16"
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
            # if stmt.src is a Constant but outside of the above range of value < -4095 or value > 4095
            # we can simply add the Constant to a register
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
            src_str = ("#" + stmt.src.ugly).split("_")[0]
        else:
            src_str = stmt.src.ugly
        if stmt.typ == AsmType.f64x8:
            s = f"fmov {stmt.dest.ugly}, {src_str}"
        else:
            s = f"mov {stmt.dest.ugly}, {src_str}"
        self.addLine(s, stmt.comment)

    def visitLoad(self, stmt: LoadStmt):
        if isinstance(stmt.src, Label):
            src_str = "#" + stmt.src.ugly
        elif isinstance(stmt.dest, MemoryAddress) and (stmt.src.ugly_offset != "0" and stmt.scalar_offs):
            self.addLine(f"mov {stmt.add_reg.ugly}, #{stmt.src.ugly_offset}", f"move immediate offset into {stmt.add_reg.ugly}")
            # TODO: adapt ugly_lsl_shift to account for possible single precision instead of double precision
            src_str = f"[{stmt.src.ugly_base}, {stmt.add_reg.ugly}, LSL #{stmt.dest.ugly_lsl_shift}]"
        elif stmt.typ == AsmType.f64x4 or stmt.typ == AsmType.f64x2:
            # (note: the 128-bit and 256-bit broadcasts need the following more rudimentary format here)
            if stmt.src.ugly_offset == '0':
                src_str = f"[{stmt.src.ugly_base}]"
            else:
                src_str = f"[{stmt.src.ugly_base}, #{stmt.src.ugly_offset}]"
        else:
            src_str = stmt.src.ugly if not stmt.is_B else stmt.src.ugly_no_vl_scaling

        p = self.p_string(stmt.pred)
        prec = self.ugly_precision

        if stmt.typ == AsmType.i64:
            s = f"ldr {stmt.dest.ugly}, {src_str}"
        elif stmt.typ == AsmType.f64x8 and stmt.aligned:
            if stmt.is_B:
                s = f"ld1r{prec} {stmt.dest.ugly}, {p}{src_str}"
            else:
                s = f"ld1{prec} {stmt.dest.ugly}, {p}{src_str}"
        elif stmt.typ == AsmType.f64x4 and stmt.aligned:
            s = f"ld1ro{prec} {stmt.dest.ugly}, {p}{src_str}"
        elif stmt.typ == AsmType.f64x2 and stmt.aligned:
            s = f"ld1rq{prec} {stmt.dest.ugly}, {p}{src_str}"
        else:
            raise NotImplementedError()
        self.addLine(s, stmt.comment)

    def visitStore(self, stmt: StoreStmt):
        if isinstance(stmt.src, Label):
            src_str = "#" + stmt.src.ugly
        elif isinstance(stmt.dest, MemoryAddress) and stmt.dest.ugly_offset != "0" and stmt.scalar_offs:
            self.addLine(f"mov {stmt.add_reg.ugly}, #{stmt.dest.ugly_offset}",
                         f"move immediate offset into {stmt.add_reg.ugly}")
            # TODO: adapt ugly_lsl_shift to account for possible single precision instead of double precision
            regsize = stmt.add_dest.size() // 16
            dest_str = f"[{stmt.dest.ugly_base}, {stmt.add_reg.ugly}, LSL #{stmt.src.ugly_lsl_shift}]"
        else:
            dest_str = stmt.dest.ugly

        p = self.p_string(stmt.pred)
        prec = self.ugly_precision

        if stmt.typ == AsmType.i64:
            s = f"str {src_str}, {stmt.dest.ugly}"
        elif stmt.typ == AsmType.f64x8 and stmt.aligned:
            s = f"st1{prec} {stmt.src.ugly}, {p}{dest_str}"
        else:
            raise NotImplementedError()
        self.addLine(s, stmt.comment)

    def visitPrefetch(self, stmt: PrefetchStmt):
        # https://stackoverflow.com/questions/37070/what-is-the-meaning-of-non-temporal-memory-accesses-in-x86#:~:text=Data%20referenced%20by%20a%20program,%2C%20is%20often%20non%2Dtemporal.
        cache_level = "L1"  # specify cache level to which we prefetch
        temporality = "KEEP"  # could use "STRM" for non-temporal prefetching if needed
        xn = stmt.dest.ugly_base
        offset = stmt.dest.ugly_offset
        src_string = f"[{xn}, {offset}, MUL VL]"
        p = self.p_string(stmt.pred)
        prec = self.ugly_precision
        s = f"prf{prec} P{stmt.access_type}{cache_level}{temporality}, {p.split('/')[0]}{src_string}"
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
