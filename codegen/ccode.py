from codegen.ast import *
from codegen.analysis import *

import architecture


def make_cfunc(funcName:str, template:str, body:Block, flop:int) -> str:
    Printer_class = architecture.get_class("codegen.architectures." + architecture.arch + ".inlineprinter.InlinePrinter")

    printer = Printer_class()
    printer.lmargin = 4
    body.accept(printer)
    body_text = "\n".join(printer.output)

    analyzer = Analyzer()
    body.accept(analyzer)
    regs = ['"{}"'.format(reg.clobbered) for reg in analyzer.clobbered_registers]
    regs.sort()
    clobbered = ",".join(regs)
    return template.format(funcName = funcName,
                           body_text = body_text,
                           clobbered = clobbered,
                           flop = flop)

