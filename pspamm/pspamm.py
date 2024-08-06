#!/usr/bin/env python3

import argparse

import pspamm.architecture

from pspamm.matmul import *

from pspamm.codegen.ccode import *
from pspamm.codegen.architectures import *


mtx_formats = ['any','csc','csr','bsc','bsr','bcsc','bcsr']


def generate(alg: MatMul) -> None:

    block = alg.make()

    text = make_cfunc(alg.output_funcname, alg.generator.get_template(), block, alg.flop, alg.starting_regs + alg.mask_regs, alg.generator.get_precision())

    if alg.output_filename is None:
        print(text)
    else:
        mode = "a"
        if alg.output_overwrite:
            mode = "w"
        with open(alg.output_filename, mode) as f:
            f.write(text)



def main() -> None:
    parser = argparse.ArgumentParser(description='Generate a sparse matrix multiplication algorithm for C = alpha * A * B + beta * C.')

    parser.add_argument("m", type=int, help="Number of rows of A and C")
    parser.add_argument("n", type=int, help="Number of cols of B and C")
    parser.add_argument("k", type=int, help="Number of cols of A, rows of B")

    parser.add_argument("lda", type=int, help="Leading dimension of A (zero if A is sparse)")
    parser.add_argument("ldb", type=int, help="Leading dimension of B (zero if B is sparse)")
    parser.add_argument("ldc", type=int, help="Leading dimension of C")

    parser.add_argument("alpha", type=str, help="alpha, 1.0 or generic")

    parser.add_argument("beta", type=str, help="beta, 1.0, 0.0, or generic")

    parser.add_argument("--bm", type=int, help="Size of m-blocks")
    parser.add_argument("--bn", type=int, help="Size of n-blocks")
    parser.add_argument("--bk", type=int, help="Size of k-blocks")

    parser.add_argument("--arch", help="Architecture", default="knl")
    parser.add_argument("--precision", help="Precision of the matrix multiplication, either half (h), single (s), or double (d)", default="d")

    parser.add_argument("--prefetching", help="Prefetching")

    parser.add_argument("--mtx_filename", help="Path to MTX file describing the sparse matrix")
    parser.add_argument("--mtx_format", help="Constraint on sparsity pattern", choices=mtx_formats, default="Any")

    parser.add_argument("--output_funcname", help="Name for generated C++ function")
    parser.add_argument("--output_filename", help="Path to destination C++ file")
    parser.add_argument("--output_overwrite", action="store_true", help="Overwrite output file")

    args = parser.parse_args()
    alg = MatMul(**args.__dict__)
    generate(alg)

if __name__ == "__main__":
    main()
