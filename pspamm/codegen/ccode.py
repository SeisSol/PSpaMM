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
    analyzer.collect(body)
    regs = set('"{}"'.format(reg.clobbered) for reg in analyzer.clobbered_registers)
    regs.add('"memory"')
    regs.add('"cc"')
    # TODO: maybe regs.add('"redzone"') ?
    clobbered = ", ".join(sorted(regs))
    arglist = ", ".join(sorted(arg.arg for arg in analyzer.input_operands))
    return template.format(funcName = funcName,
                           body_text = body_text,
                           args = arglist,
                           clobbered = clobbered,
                           flop = flop,
                           real_type = Precision.getCType(precision))

