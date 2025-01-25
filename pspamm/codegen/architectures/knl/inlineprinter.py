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
        self.bpsuffix = {
            Precision.DOUBLE: "q",
            Precision.SINGLE: "d",
            Precision.HALF: "w",
            Precision.BFLOAT16: "w",
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

    def maskformat(self, pred, ignoreZero = False):
        if pred is None:
            return ''
        elif pred.zero and not ignoreZero:
            return f'%{{{pred.register.ugly}%}}%{{z%}}'
        else:
            return f'%{{{pred.register.ugly}%}}'

    def visitFma(self, stmt: FmaStmt):
        mask = self.maskformat(stmt.pred)
        b = stmt.bcast_src.ugly
        m = stmt.mult_src.ugly
        a = stmt.add_dest.ugly
        regsize = stmt.add_dest.size() // 16
        extent = regsize * self.broadcast_multiplier
        op = "nmadd" if stmt.sub else "madd"
        if stmt.bcast is not None:
            s = f"vf{op}231{self.alupsuffix} {b}%{{1to{extent}%}}, {m}, {a} {mask}"
        else:
            if stmt.mult_src.typeinfo == AsmType.i64:
                # in this case, m is a Register that points to alpha; manually format to be a memory address
                s = f"vf{op}231{self.alupsuffix} 0({m})%{{1to{extent}%}}, {b}, {a} {mask}"
            else:
                s = f"vf{op}231{self.alupsuffix} {b}, {m}, {a} {mask}"
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
            s = f"vmul{self.alupsuffix} 0({m})%{{1to{extent}%}}, {b}, {a} {mask}"
        else:
            s = f"vmul{self.alupsuffix} {b}, {m}, {a} {mask}"
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
        s = f"{instruction} {b}, {a} {mask}"
        self.addLine(s, stmt.comment)

    def visitAdd(self, stmt: AddStmt):
        if isinstance(stmt.src, Constant) and stmt.src.value == 0:
            # avoid 0 instructions
            return
        
        # only used for scalar addition right now
        s = f"addq {stmt.src.ugly}, {stmt.dest.ugly}"
        self.addLine(s, stmt.comment)

    def visitLabel(self, stmt: LabelStmt):
        self.addLine('.align 16', 'Align label')
        s = f"{stmt.label.ugly}:"
        self.addLine(s, stmt.comment)

    def visitCmp(self, stmt: CmpStmt):
        mask = self.maskformat(stmt.pred)
        s = f"cmp {stmt.lhs.ugly}, {stmt.rhs.ugly} {mask}"
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
            # FIXME: no hack
            if stmt.dest.ugly[2] == 'k':
                s = f"kmovq {src_str}, {stmt.dest.ugly}"
            else:
                s = f"movq {src_str}, {stmt.dest.ugly}"
        elif stmt.typ == AsmType.f64x8 and stmt.aligned:
            if isinstance(stmt.src, Constant) and stmt.src.value == 0:
                suffix = 'd' if self.bpsuffix == 'w' else self.bpsuffix
                s = f"vpxor{suffix} {stmt.dest.ugly}, {stmt.dest.ugly}, {stmt.dest.ugly} {mask}"
            elif stmt.expand:
                if isinstance(stmt.src, MemoryAddress):
                    s = f"vpexpand{self.bpsuffix} {src_str}, {stmt.dest.ugly} {mask}"
                else:
                    s = f"vpcompress{self.bpsuffix} {src_str}, {stmt.dest.ugly} {mask}"
            else:
                if self.bpsuffix == 'w' and stmt.pred is not None:
                    instr = "vmovsh"
                else:
                    instr = f"vmovup{self.psuffix}"
                mask = self.maskformat(stmt.pred, True)
                s = f"{instr} {src_str}, {stmt.dest.ugly} {mask}"
        else:
            raise NotImplementedError()
        self.addLine(s, stmt.comment)

    def visitLea(self, stmt: LeaStmt):
        mask = self.maskformat(stmt.pred)
        s = f"leaq {stmt.offset}({stmt.src.ugly}), {stmt.dest.ugly} {mask}"
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
