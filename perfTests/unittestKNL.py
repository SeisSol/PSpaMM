#!/usr/bin/python3.6
from collections import namedtuple
import subprocess
import numpy as np
from random import sample

def generateMTX(k, n, nnz):

	assert(nnz <= k * n)

	filename = str(k) + 'x' + str(n) + '_' + str(nnz) + '.mtx'

	f = open('mtx/' + filename, 'w')

	f.write('%%MatrixMarket matrix coordinate integer general\n%\n' + str(k) + ' ' + str(n) + ' ' + str(nnz))

	zeros = set()

	for i in range(1,k+1):
		for j in range(1,n+1):
			zeros.add((i,j))

	nonzeros = sample(zeros, nnz)

	for entry in nonzeros:
		f.write('\n' + str(entry[0]) + ' ' + str(entry[1]) + ' 1')

	f.close()

	return filename


DenseKernel = namedtuple("DenseKernel", "name m n k lda ldb ldc beta block_sizes")
SparseKernel = namedtuple("SparseKernel", "name m n k lda ldb ldc beta block_sizes mtx")

kernels = []

kernels.append(SparseKernel("test1", 8, 56, 56, 8, 0, 8, 0, [(8, 28), (8,1)],generateMTX(56, 56, 294)))
kernels.append(DenseKernel("test2", 8, 56, 56, 8, 56, 8, 0, [(8, 14), (8, 5)]))



for kern in kernels:

	arguments = ['./../sparsemmgen.py', str(kern.m), str(kern.n), str(kern.k), str(kern.lda), str(kern.ldb), str(kern.ldc), str(kern.beta)]

	if isinstance(kern, SparseKernel):
		arguments += ['--mtx_filename', 'mtx/' + kern.mtx]
	for bs in kern.block_sizes:
		bm = bs[0]
		bn = bs[1]

		assert(bm % 8 == 0 and (bn+1) * (bm / 8) <= 32)

		additional_args = ['--output_funcname', kern.name + "_" + str(bm) + "_" + str(bn), '--output_filename', 'knl/' + kern.name + '_' + str(bm) + '_' + str(bn) + '.h']
		additional_args += ['--bm', str(bm), '--bn', str(bn), '--bk', '1']

		try:
			subprocess.check_output(arguments + additional_args,stderr=subprocess.STDOUT)
		except subprocess.CalledProcessError as e:
			raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

