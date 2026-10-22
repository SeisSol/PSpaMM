import pypspamm.architecture
from pypspamm.codegen.analysis import *
from pypspamm.codegen.ast import *
from pypspamm.codegen.precision import *


def make_cfunc(
    funcName: str,
    template: str,
    body: Block,
    flop: int,
    starting_regs: List[Register],
    precision: Precision,
    prelude: str = "",
) -> str:
    """The C function of a kernel.

    prelude is C code placed ahead of the inline assembly, inside the function:
    data the kernel carries with it, such as the values of an operand that were
    known when it was generated (see MatMul.prelude).
    """
    Printer_class = pypspamm.architecture.get_class(
        "pypspamm.codegen.architectures."
        + pypspamm.architecture.arch
        + ".inlineprinter"
    ).InlinePrinter

    printer = Printer_class(precision)
    printer.lmargin = 4
    body.accept(printer)
    body_text = "\n".join(printer.output)

    analyzer = Analyzer(starting_regs)
    analyzer.collect(body)
    regs = set(
        f'"{reg.clobbered}"'
        for reg in analyzer.clobbered_registers
        if reg.clobbered is not None
    )
    regs.add('"memory"')
    regs.add('"cc"')
    # TODO: maybe regs.add('"redzone"') ?
    clobbered = ", ".join(sorted(regs))
    arglist = ", ".join(sorted(arg.arg for arg in analyzer.input_operands))
    return template.format(
        funcName=funcName,
        body_text=body_text,
        args=arglist,
        clobbered=clobbered,
        flop=flop,
        real_type=Precision.getCType(precision),
        prelude=prelude,
    )
