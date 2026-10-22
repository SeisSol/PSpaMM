from typing import Sequence

import pypspamm.architecture
from pypspamm.codegen.analysis import *
from pypspamm.codegen.arguments import (
    Argument,
    declarations,
    mentions,
    operands,
    unused,
)
from pypspamm.codegen.ast import *
from pypspamm.codegen.precision import *


def make_cfunc(
    funcName: str,
    template: str,
    body: Block,
    flop: int,
    reserved: List[Register],
    precision: Precision,
    prelude: str = "",
    arguments: Sequence[Argument] = (),
) -> str:
    """The C function of a kernel.

    reserved are registers the kernel keeps for itself beyond those its
    statements name. arguments are those the C function binds to their
    registers (see codegen/arguments.py); the assembly gets them as operands it
    may change, so they are not clobbered, and the parameters the kernel reads
    in neither way are marked as used. prelude is C code placed ahead of the
    inline assembly, inside the function: data the kernel carries with it, such
    as the values of an operand that were known when it was generated (see
    MatMul.prelude).
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

    analyzer = Analyzer(reserved)
    analyzer.collect(body)
    bound = set(argument.register for argument in arguments)
    regs = set(
        f'"{reg.clobbered}"'
        for reg in analyzer.clobbered_registers
        if reg.clobbered is not None and reg not in bound
    )
    regs.add('"memory"')
    regs.add('"cc"')
    # TODO: maybe regs.add('"redzone"') ?
    clobbered = ", ".join(sorted(regs))
    arglist = ", ".join(sorted(arg.arg for arg in analyzer.input_operands))
    real_type = Precision.getCType(precision)
    read = [
        name
        for source in [argument.expression for argument in arguments]
        + [operand.source for operand in analyzer.input_operands]
        for name in mentions(source)
    ]
    return template.format(
        funcName=funcName,
        body_text=body_text,
        args=arglist,
        clobbered=clobbered,
        flop=flop,
        real_type=real_type,
        prelude=prelude,
        arguments=unused(read) + declarations(arguments, real_type),
        outputs=operands(arguments),
    )
