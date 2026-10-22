#!/usr/bin/env python3

import argparse
import json
import os

import pypspamm.architecture
from pypspamm.codegen.architectures import *
from pypspamm.codegen.architectures import registry
from pypspamm.codegen.ccode import *
from pypspamm.matmul import *
from pypspamm.metagen.metagen import *

mtx_formats = ["any", "csc", "csr", "bsc", "bsr", "bcsc", "bcsr"]


def capabilities() -> dict:
    """What this generator supports beyond the matrix multiplication itself,
    for a caller that has to know before it asks for a kernel: whether it
    takes the values of a sparse A to build in (abake), the forms such a
    kernel can take (aformat) and the architectures that have the ELLPACK
    form."""

    try:
        with open(os.path.join(os.path.dirname(__file__), "VERSION")) as f:
            version = f.read().strip()
    except OSError:
        version = None
    ellpack = []
    for archspec in registry.ARCHITECTURES:
        generator = pypspamm.architecture.get_class(
            "pypspamm.codegen.architectures." + archspec.module + ".generator"
        ).Generator
        if hasattr(generator, "make_ellpack"):
            ellpack.append(archspec.prefix)
    return {
        "version": version,
        "abake": True,
        "aformat": ["auto", "blocked", "ellpack"],
        "ellpack": sorted(ellpack),
    }


class PrintCapabilities(argparse.Action):
    """Prints the capabilities as JSON and exits, without the arguments of a
    kernel, as --version would."""

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(
            option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        print(json.dumps(capabilities(), sort_keys=True))
        parser.exit()


def generate(archs, params) -> None:
    text = MetaGenerator(archs).generate(dict(params))

    if params["output_filename"] is None:
        print(text)
    else:
        mode = "w" if params["output_overwrite"] else "a"
        with open(params["output_filename"], mode) as f:
            f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a sparse matrix multiplication algorithm for C = alpha * A * B + beta * C."
    )

    parser.add_argument(
        "--capabilities",
        action=PrintCapabilities,
        help="Print what this generator supports, as JSON, and exit",
    )
    parser.add_argument("m", type=int, help="Number of rows of A and C")
    parser.add_argument("n", type=int, help="Number of cols of B and C")
    parser.add_argument("k", type=int, help="Number of cols of A, rows of B")

    parser.add_argument(
        "lda", type=int, help="Leading dimension of A (zero if A is sparse)"
    )
    parser.add_argument(
        "ldb", type=int, help="Leading dimension of B (zero if B is sparse)"
    )
    parser.add_argument("ldc", type=int, help="Leading dimension of C")

    parser.add_argument("alpha", type=str, help="alpha, 1.0 or generic")

    parser.add_argument("beta", type=str, help="beta, 1.0, 0.0, or generic")

    parser.add_argument("--bm", type=int, help="Size of m-blocks")
    parser.add_argument("--bn", type=int, help="Size of n-blocks")
    parser.add_argument("--bk", type=int, help="Size of k-blocks")

    parser.add_argument(
        "--arch",
        help=(
            "Architecture, or a comma separated list of them, in which case a "
            "dispatcher is generated that selects between them at run time. "
            "List them from the most to the least capable; the last one is the "
            "fallback. The presets "
            + ", ".join(sorted(registry.PRESETS))
            + " each stand for such a list."
        ),
        default="knl",
    )
    parser.add_argument(
        "--precision",
        help="Precision of the matrix multiplication, either half (h), single (s), or double (d)",
        default="d",
    )

    parser.add_argument("--prefetching", help="Prefetching")

    parser.add_argument(
        "--mtx_filename", help="Path to MTX file describing the sparse matrix"
    )
    parser.add_argument(
        "--mtx_format",
        help="Constraint on sparsity pattern (accepted, but has no effect)",
        choices=mtx_formats,
        default="any",
    )

    parser.add_argument(
        "--scheduling",
        help="Instruction scheduling level",
        choices=["none", "peephole", "pipeline"],
        default="none",
    )

    parser.add_argument(
        "--amtx_filename", help="Path to MTX file describing the sparse matrix"
    )
    parser.add_argument(
        "--bmtx_filename", help="Path to MTX file describing the sparse matrix"
    )
    parser.add_argument(
        "--aformat",
        choices=["auto", "blocked", "ellpack"],
        default="auto",
        help=(
            "How a kernel with the values of A built in (--abake) works through A: "
            "in vector tiles like any sparse A (blocked), or in sliced ELLPACK, "
            "gathering from B (ellpack), where the architecture has the kernel; "
            "auto takes the one estimated to be faster on the architecture"
        ),
    )
    parser.add_argument(
        "--abake",
        action="store_true",
        help=(
            "The values in the file of a sparse A are the values of A: the kernel "
            "keeps them in a table of its own and does not read A"
        ),
    )

    parser.add_argument("--output_funcname", help="Name for generated C++ function")
    parser.add_argument("--output_filename", help="Path to destination C++ file")
    parser.add_argument(
        "--output_overwrite", action="store_true", help="Overwrite output file"
    )

    args = parser.parse_args()
    params = dict(args.__dict__)
    archs = [name.strip() for name in params.pop("arch").split(",")]
    generate(archs, params)


if __name__ == "__main__":
    main()
