#!/usr/bin/python3
from collections import namedtuple
import subprocess
import numpy as np
import random
import sys
import os
import testsuite_generator as test_generator
from pspamm.codegen.precision import *

BASEDIR = 'build'

SparseKernel = test_generator.SparseKernel
DenseKernel = test_generator.DenseKernel

setup_prefetching = """
template <typename T>
void setup_prefetch(T*& prefetch, T* matrix, unsigned n, unsigned ldc) {
 posix_memalign(reinterpret_cast<void **>(&prefetch), 64, ldc*n*sizeof(T));
 std::memcpy(prefetch, matrix, ldc*n*sizeof(T));
}
"""

def generateMTX(k, n, nnz):
    return test_generator.generateMTX(k, n, nnz)

def make(kernels, arch):
    os.makedirs(os.path.join(BASEDIR, arch), exist_ok=True)

    f = open(os.path.join(BASEDIR, f'{arch}_testsuite.cpp'), 'w')

    f.write(test_generator.head_of_testsuite)

    testcases = []

    for kern in kernels:
        arguments = ['pspamm-generator', str(kern.m), str(kern.n), str(kern.k), str(kern.lda),
                     str(kern.ldb), str(kern.ldc), str(kern.alpha), str(kern.beta)]

        if isinstance(kern, SparseKernel):
            arguments += ['--mtx_filename', kern.mtx]

        prec = 's' if kern.precision == Precision.SINGLE else 'd'
        arguments += ['--precision', prec]

        block_sizes = list(set(bs if len(bs) > 2 else (bs[0], bs[1], 1) for bs in kern.block_sizes))

        for bs in block_sizes:
            bm = bs[0]
            bn = bs[1]
            bk = bs[2]

            if arch == "knl":
                assert (bm % 8 == 0 and (bn + 1) * (bm / 8) <= 32)
            elif arch == "arm":
                assert (bm % 2 == 0 and (bn + 1) * (bm / 2) + bn <= 32)
            elif arch.startswith("arm_sve"):
                veclen = int(arch[7:])
                assert veclen % 128 == 0 and veclen <= 2048
                reglen = veclen // 128
                v_len = (16 // kern.precision.size()) * reglen
                # this should be the same assertion as in ../scripts/max_arm_sve.py
                # ceiling division
                vm = -(bm // -v_len)
                if not ((bn + bk) * vm + bn * bk <= 32):
                    print(f'Skipping block size {bm}x{bn}x{bk} for {arch} / {prec}')
                    continue

            name = kern.name + '_' + str(bm) + '_' + str(bn) + '_' + str(bk)

            additional_args = ['--output_funcname', name, '--output_filename', os.path.join(BASEDIR, arch, name + '.h'),
                               '--output_overwrite']
            additional_args += ['--bm', str(bm), '--bn', str(bn), '--bk', str(bk), '--arch', arch, '--prefetching', 'BL2viaC']

            try:
                subprocess.check_output(arguments + additional_args, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

            f.write('#include "' + arch + '/' + name + '.h"\n')

            if isinstance(kern, SparseKernel):
                mtx = kern.mtx
            else:
                mtx = ""
            
            prec2 = 'f' if kern.precision == Precision.SINGLE else ''

            testcases += [
                """
{{
  unsigned ldb = {ldb};
  {T} alpha = {alpha};
  {T} beta = {beta};
  {T}* prefetch = nullptr;
  auto pointers = pre<{T}>({m}, {n}, {k}, {lda}, ldb, {ldc}, "{mtx}");
  setup_prefetch(prefetch, std::get<3>(pointers), {n}, {ldc});
  {name}(std::get<0>(pointers), std::get<{sparse}>(pointers), std::get<3>(pointers), alpha, beta, prefetch);
  const auto result = post<{T}>({m}, {n}, {k}, {lda}, &ldb, {ldc}, &alpha, &beta, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), {delta:.7f});
  results.push_back(std::make_tuple("{name}", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers)); free(prefetch);
}}
""".format(m=kern.m, n=kern.n, k=kern.k, lda=kern.lda, ldb=kern.ldb, ldc=kern.ldc, alpha=kern.alpha, beta=kern.beta,
           mtx=mtx, delta=kern.delta, name=name, sparse=2 if kern.ldb == 0 else 1, p=prec2, T=kern.precision.ctype())
            ]

    f.write('\n')
    # necessary functions are defined in testsuite_generator.py
    f.write(test_generator.function_definitions)
    f.write(setup_prefetching)
    f.write(test_generator.setup_main)
  
    for testcase in testcases:
        f.write(testcase)

    f.write(test_generator.end_of_testsuite)
