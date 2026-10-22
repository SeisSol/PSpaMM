"""C = alpha A B + beta C for a sparse A whose values are built into the kernel,
with A in sliced ELLPACK (see codegen/ellpack.py).

The counterpart of MatMul for that one case. It reads A and lays it out, and
leaves the kernel to the generator of the architecture (make_ellpack), which
says whether it can build it at all (ellpack_registers).
"""

import numpy

import pypspamm.architecture
from pypspamm.codegen.architectures import registry
from pypspamm.codegen.ellpack import Ellpack
from pypspamm.codegen.precision import Precision, Types
from pypspamm.codegen.sugar import block
from pypspamm.codegen.table import bits, table_prelude
from pypspamm.cursors import Matrix
from pypspamm.matmul import select_architecture


def ellpack_supported(arch: str, precision: str, k: int) -> bool:
    """Whether the generator of an architecture builds the ELLPACK kernel for
    this precision and a B with k rows."""

    module = registry.parse(arch)[0].module
    generator = pypspamm.architecture.get_class(
        "pypspamm.codegen.architectures." + module + ".generator"
    ).Generator
    if not hasattr(generator, "ellpack_registers"):
        return False
    instance = generator(Precision.fromCode(precision))
    instance.v_len = registry.parse(arch)[1]
    return instance.ellpack_registers(k) is not None


class EllpackMatMul:
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        lda: int,
        ldb: int,
        ldc: int,
        alpha: str,
        beta: str,
        amtx_filename: str,
        arch: str = "knl",
        precision: str = "d",
        **kwargs,  # Accept and ignore args which don't belong
    ) -> None:
        assert lda == 0, "the values of a sparse A are built in"
        assert ldb >= k and ldc >= m

        self.m, self.n, self.k = m, n, k
        self.ldb, self.ldc = ldb, ldc
        try:
            self.alpha = float(alpha)
        except ValueError:
            self.alpha = "generic"
        try:
            self.beta = float(beta)
        except ValueError:
            self.beta = "generic"

        self.precision = Precision.fromCode(precision)
        self.types = Types.uniform(self.precision)
        self.arch, self.generator, v_len_regs = select_architecture(
            arch, self.precision
        )
        self.generator.v_len = v_len_regs
        self.v_size = self.generator.get_v_size()

        values = numpy.zeros((m, k))
        loaded = numpy.asarray(Matrix.load(amtx_filename))
        rows, cols = min(m, loaded.shape[0]), min(k, loaded.shape[1])
        values[:rows, :cols] = loaded[:rows, :cols]
        self.layout = Ellpack(values, self.v_size)

        self.flop = 2 * n * int(numpy.count_nonzero(values))

        self.body, words, self.starting_regs = self.generator.make_ellpack(
            self.layout,
            n,
            ldb,
            ldc,
            self.alpha,
            self.beta,
            self.types,
            lambda value: bits(value, self.types.a),
        )
        self.prelude = table_prelude("A", words, self.types.a)

    def make(self):
        loads, self.arguments, self.reserved_regs = self.generator.take_arguments(
            self.starting_regs, False, self.body
        )
        asm = block("kernel")
        if loads:
            asm.add(block("Load arguments", *loads))
        asm.add(self.body)
        return asm
