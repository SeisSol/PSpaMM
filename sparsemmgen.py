#!/usr/bin/python3.6

import argparse

import architecture

from matmul import *

from codegen.ccode import *
from codegen.architectures import *


mtx_formats = ['any','csc','csr','bsc','bsr','bcsc','bcsr']


def main(alg: MatMul) -> None:

	block = MatMul.make(alg)

	text = make_cfunc(alg.output_funcname, block)

	if alg.output_filename is None:
		print(text)
	else:
		with open(alg.output_filename, "a") as f:
			f.write(text)



if __name__=="__main__":

	parser = argparse.ArgumentParser(description='Generate a sparse matrix multiplication algorithm.')

	parser.add_argument("m", type=int, help="Number of rows of A and C")
	parser.add_argument("n", type=int, help="Number of cols of B and C")
	parser.add_argument("k", type=int, help="Number of cols of A, rows of B")

	parser.add_argument("lda", type=int, help="Leading dimension of A (zero if A is sparse)")
	parser.add_argument("ldb", type=int, help="Leading dimension of B (zero if B is sparse)")
	parser.add_argument("ldc", type=int, help="Leading dimension of C")

	parser.add_argument("beta", type=int, help="beta, if zero then C is set to 0 otherwise new result is added to old C")

	parser.add_argument("--bm", type=int, help="Size of m-blocks")
	parser.add_argument("--bn", type=int, help="Size of n-blocks")
	parser.add_argument("--bk", type=int, help="Size of k-blocks")

	parser.add_argument("--arch", help="Architecture", default="knl")

	parser.add_argument("--mtx_filename", help="Path to MTX file describing the sparse matrix")
	parser.add_argument("--mtx_format", help="Constraint on sparsity pattern", choices=mtx_formats, default="Any")

	parser.add_argument("--output_funcname", help="Name for generated C++ function")
	parser.add_argument("--output_filename", help="Path to destination C++ file")

	args = parser.parse_args()
	alg = MatMul(**args.__dict__)
	main(alg)