from codegen.ast import *
from codegen.analysis import *

import architecture


def make_cfunc(funcName:str, body:Block) -> str:
    Gen = architecture.get_class(architecture.arch + ".inlineprinter.InlinePrinter")

    printer = Gen()
    printer.lmargin = 4
    body.accept(printer)
    body_text = "\n".join(printer.output)

    analyzer = Analyzer()
    body.accept(analyzer)
    regs = [f'"{reg.clobbered}"' for reg in analyzer.clobbered_registers]
    regs.sort()
    clobbered = ",".join(regs)

    template = architecture.generator.template

    return template.format(funcName = funcName,
                           body_text = body_text,
                           clobbered = clobbered)

