#!/usr/bin/python3.6

import argparse

#from codegen import *
from components.parameters import Parameters
from codegen.ccode import *
import dxsp_unrolled	
import architecture
import arm.generator
import arm.operands
import knl.generator
import knl.operands




mtx_formats = ['any','csc','csr','bsc','bsr','bcsc','bcsr']


def main(params: Parameters) -> None:
	architecture.init()
	architecture.arch = params.arch
	architecture.generator = architecture.get_class(params.arch + ".generator")
	architecture.operands = architecture.get_class(params.arch + ".operands")

	generator = dxsp_unrolled
	params = generator.choose_params(params)       # type: ignore
	block = generator.make_alg(params)             # type: ignore

	# MyPy does not support 'module interfaces'
	# https://github.com/python/mypy/issues/1741

	text = make_cfunc(params.output_funcname, block)


	if params.output_filename is None:
		print(text)

	else:
		with open(params.output_filename, "a") as f:
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

	parser.add_argument("--v_size", type=int, help="Size of vectors")
	parser.add_argument("--arch", help="Architecture", default="knl")

	parser.add_argument("mtx_filename", help="Path to MTX file describing the sparse matrix")
	parser.add_argument("--mtx_format", help="Constraint on sparsity pattern", choices=mtx_formats, default="Any")

	parser.add_argument("--output_funcname", help="Name for generated C++ function")
	parser.add_argument("--output_filename", help="Path to destination C++ file")

	args = parser.parse_args()
	params = Parameters(**args.__dict__)
	main(params)





