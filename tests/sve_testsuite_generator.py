from collections import namedtuple
import subprocess
import numpy as np
import random
import sys
import os.path
import testsuite_generator as test_generator

SparseKernel = namedtuple('SparseKernel', 'name m n k lda ldb ldc alpha beta block_sizes mtx delta')
DenseKernel = namedtuple('DenseKernel', 'name m n k lda ldb ldc alpha beta block_sizes delta')

SparseKernelS = namedtuple('SparseKernelS', 'name m n k lda ldb ldc alpha beta block_sizes mtx delta')
DenseKernelS = namedtuple('DenseKernelS', 'name m n k lda ldb ldc alpha beta block_sizes delta')

setup_prefetching = """
template <typename T>
void setup_prefetch(T* prefetch, T* matrix, unsigned n, unsigned ldc) {
 posix_memalign(reinterpret_cast<void **>(&prefetch), 64, {ldc}*{n}*sizeof({T}));
 std::memcpy(prefetch, matrix, {ldc}*{n}*sizeof({T}));
}
"""

def generateMTX(k, n, nnz):
    return test_generator.generateMTX(k, n, nnz)

def make(kernels, arch):
    f = open('sve_testsuite.cpp', 'w')

    f.write(test_generator.head_of_testsuite)

    include_single_prec = False

    for kern in kernels:
        arguments = [sys.executable, './../pspamm.py', str(kern.m), str(kern.n), str(kern.k), str(kern.lda),
                     str(kern.ldb), str(kern.ldc), str(kern.alpha), str(kern.beta)]

        if isinstance(kern, SparseKernel) or isinstance(kern, SparseKernelS):
            arguments += ['--mtx_filename', kern.mtx]

        prec = 's' if isinstance(kern, SparseKernelS) or isinstance(kern, DenseKernelS) else 'd'
        arguments += ['--precision', prec]
        if prec == 's':
            include_single_prec = True

        block_sizes = list(set(kern.block_sizes))

        for bs in block_sizes:
            bm = bs[0]
            bn = bs[1]

            if arch == "knl":
                assert (bm % 8 == 0 and (bn + 1) * (bm / 8) <= 32)
            elif arch == "arm":
                assert (bm % 2 == 0 and (bn + 1) * (bm / 2) + bn <= 32)
            elif arch == "arm_sve":
                # this is for A64fx only, with SVE_vector_bits = 512
                v_len = 8 if prec == 'd' else 16
                # this should be the same assertion as in ../scripts/max_arm_sve.py
                assert ((bn + 1) * (bm / v_len) + bn <= 32)

            name = kern.name + '_' + str(bm) + '_' + str(bn)

            additional_args = ['--output_funcname', name, '--output_filename', arch + '/' + name + '.h',
                               '--output_overwrite']
            additional_args += ['--bm', str(bm), '--bn', str(bn), '--arch', arch, '--prefetching', 'BL2viaC']

            try:
                subprocess.check_output(arguments + additional_args, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

            f.write('#include "' + arch + '/' + kern.name + '_' + str(bm) + '_' + str(bn) + '.h"\n')

    f.write('\n')
    # necessary functions are defined in testsuite_generator.py
    f.write(test_generator.function_definitions)
    f.write(setup_prefetching)
    f.write(test_generator.setup_main)
    # add variable declarations for single precision test cases
    if include_single_prec:
        f.write("""  std::tuple<float*, float*, float*, float*, float*> fpointers;
  float falpha; float fbeta;
  double* prefetch;
  float* fprefetch;
  """)

    for kern in kernels:

        block_sizes = list(set(kern.block_sizes))

        for bs in block_sizes:
            bm = bs[0]
            bn = bs[1]
            name = kern.name + '_' + str(bm) + '_' + str(bn)

            if isinstance(kern, SparseKernel) or isinstance(kern, SparseKernelS):
                mtx = kern.mtx
            else:
                mtx = ""
            # for double precision: set prec to '' to conform to test_generator.function_definitions
            prec = 'f' if isinstance(kern, SparseKernelS) or isinstance(kern, DenseKernelS) else ''

            f.write("""
  {p}alpha = {alpha}; {p}beta = {beta}; ldb = {ldb};
  {p}pointers = pre<{T}>({m}, {n}, {k}, {lda}, ldb, {ldc}, "{mtx}");
  setup_prefetch({p}prefetch, std::get<3>({p}pointers), {n}, {ldc});
  {name}(std::get<0>({p}pointers), std::get<{sparse}>({p}pointers), std::get<3>({p}pointers), {p}alpha, {p}beta, {p}prefetch);
  result = post<{T}>({m}, {n}, {k}, {lda}, &ldb, {ldc}, &{p}alpha, &{p}beta, std::get<0>({p}pointers), std::get<1>({p}pointers), std::get<3>({p}pointers), std::get<4>({p}pointers), {delta:.7f});
  results.push_back(std::make_tuple("{name}", result));
  free(std::get<0>({p}pointers)); free(std::get<1>({p}pointers)); free(std::get<2>({p}pointers)); free(std::get<3>({p}pointers)); free(std::get<4>({p}pointers));  free({p}prefetch);
""".format(m=kern.m, n=kern.n, k=kern.k, lda=kern.lda, ldb=kern.ldb, ldc=kern.ldc, alpha=kern.alpha, beta=kern.beta,
           mtx=mtx, delta=kern.delta, name=name, sparse=2 if kern.ldb == 0 else 1, p=prec, T="float" if prec == 'f' else "double"))

    f.write(test_generator.end_of_testsuite)
