"""How a kernel takes its arguments.

A kernel is a C function of A, B, C, alpha, beta and prefetch around one piece
of inline assembly, which expects every argument it reads in a register of its
own. A pointer goes in as it is: the C function binds it to that register as an
operand the assembly may change, so the assembly starts with it in place. A
value whose register cannot hold its C type, such as alpha in a general purpose
register, goes in through memory instead, as an input operand the assembly
loads. An argument the assembly does not read goes in neither way.
"""

import re
from typing import List, Sequence

from pypspamm.codegen.operands import Register

#: the parameters of every kernel, by name
PARAMETERS = ("A", "B", "C", "alpha", "beta", "prefetch")


class Argument:
    """A parameter of the kernel and the register the assembly reads it in.

    expression is what the C function binds to the register, of the C type
    ctype ({real_type} stands for the scalar type), for an argument that goes
    in as it is; without one, the assembly loads the parameter from memory.
    """

    def __init__(
        self, name: str, register: Register, expression: str = None, ctype: str = None
    ):
        assert name in PARAMETERS
        self.name = name
        self.register = register
        self.expression = expression
        self.ctype = ctype

    @property
    def bound(self) -> bool:
        return self.expression is not None

    @property
    def variable(self) -> str:
        """The C variable that holds the argument in its register."""

        return f"pspamm_{self.name}"


def declarations(arguments: Sequence[Argument], real_type: str) -> str:
    """The C code that binds each argument to its register, as the GNU
    extension of explicit register variables does for the operands of inline
    assembly."""

    lines = []
    for argument in arguments:
        ctype = argument.ctype.format(real_type=real_type)
        register = argument.register.clobbered
        lines.append(
            f'  register {ctype} {argument.variable} __asm__("{register}") = '
            f"{argument.expression};\n"
        )
    return "".join(lines)


def operands(arguments: Sequence[Argument]) -> str:
    """The output operands of the assembly that the arguments come in: the
    assembly may change them."""

    return ", ".join(f'"+r"({argument.variable})' for argument in arguments)


def unused(referenced: List[str]) -> str:
    """Statements that mark the parameters the kernel does not read as used,
    against warnings about unused parameters."""

    return "".join(
        f"  (void){name};\n" for name in PARAMETERS if name not in referenced
    )


def mentions(expression: str) -> List[str]:
    """The parameters a C expression reads."""

    return [name for name in PARAMETERS if re.search(rf"\b{name}\b", expression)]
