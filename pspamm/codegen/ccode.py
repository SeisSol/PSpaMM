from pspamm.codegen.ast import *
from pspamm.codegen.analysis import *
from pspamm.codegen.precision import *

import pspamm.architecture


def make_cfunc(funcName:str, template:str, body:Block, flop:int, starting_regs:List[Register], precision: Precision) -> str:
    Printer_class = pspamm.architecture.get_class("pspamm.codegen.architectures." + pspamm.architecture.arch + ".inlineprinter").InlinePrinter

    printer = Printer_class(precision)
    printer.lmargin = 4
    body.accept(printer)
    body_text = "\n".join(printer.output)

    analyzer = Analyzer(starting_regs)
    body.accept(analyzer)
    regs = ['"{}"'.format(reg.clobbered) for reg in analyzer.clobbered_registers]
    regs.sort()
    clobbered = ",".join(regs)
    return template.format(funcName = funcName,
                           body_text = body_text,
                           clobbered = clobbered,
                           flop = flop,
                           real_type = Precision.getCType(precision))

