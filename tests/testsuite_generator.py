from collections import namedtuple
import subprocess
import numpy as np
import random
import sys
import os.path
from pspamm.codegen.precision import *

BASEDIR = 'build'

SparseKernel = namedtuple('SparseKernel', 'name precision m n k lda ldb ldc alpha beta block_sizes mtx delta')
DenseKernel = namedtuple('DenseKernel', 'name precision m n k lda ldb ldc alpha beta block_sizes delta')

head_of_testsuite = """#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <tuple>


long long pspamm_num_total_flops = 0;
"""

function_definitions = """
template <typename T>
void pretty_print(unsigned M, unsigned N, unsigned LDC, T* C) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      printf("%.4f ", C[m * LDC + n]);
    }
    printf("\\n");
  }
}

template <typename T>
void gemm_ref(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC, T ALPHA, T BETA, T* A, T* B, T* C) {
  for (unsigned col = 0; col < N; ++col) {
    for (unsigned row = 0; row < M; ++row) {
      C[row + col * LDC] = BETA * C[row + col * LDC];
    }
  }
  for (unsigned col = 0; col < N; ++col) {
    for (unsigned row = 0; row < M; ++row) {
      for (unsigned s = 0; s < K; ++s) {
        C[row + col * LDC] += ALPHA * A[row + s * LDA] * B[s + col * LDB];
      }
    }
  }
}

template <typename T>
void setup_prefetch(T*& prefetch, T* matrix, unsigned n, unsigned ldc) {
 posix_memalign(reinterpret_cast<void **>(&prefetch), 64, ldc*n*sizeof(T));
 std::memcpy(prefetch, matrix, ldc*n*sizeof(T));
}

template <typename T>
std::tuple<T*, T*, T*, T*, T*> pre(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC, std::string MTX) {

  if(LDB == 0)
    LDB = K;

  T *A;
  T *B;
  T *Bsparse;
  T *Cref;
  T *C;

  int resA = posix_memalign(reinterpret_cast<void **>(&A), 64, LDA*LDB*sizeof(T));
  int resB = posix_memalign(reinterpret_cast<void **>(&B), 64, LDB*N*sizeof(T));
  int resBsparse = posix_memalign(reinterpret_cast<void **>(&Bsparse), 64, LDB*N*sizeof(T));  
  int resCref = posix_memalign(reinterpret_cast<void **>(&Cref), 64, LDC*N*sizeof(T));
  int resC = posix_memalign(reinterpret_cast<void **>(&C), 64, LDC*N*sizeof(T));

  std::string line;
  std::ifstream f(MTX);
  getline(f, line);
  getline(f, line);
  getline(f, line);

  for(int i = 0; i < LDA*LDB; i++)
    A[i] = (T)rand() / RAND_MAX;

  for(int i = 0; i < LDB*N; i++)
    if(MTX.compare(""))
      B[i] = 0;
    else 
      B[i] = (T)rand() / RAND_MAX;

  for(int i = 0; i < LDC*N; i++)
  {
    Cref[i] = (T)rand() / RAND_MAX;
    C[i] = Cref[i];
  }

  if(MTX.compare(""))
  {
    while(getline(f, line)) {
      std::vector<std::string> result;
      std::istringstream iss(line);
      for(std::string s; iss >> s; )
        result.push_back(s);
      if(std::atoi(result[0].c_str()) <= K && std::atoi(result[1].c_str()) <= N)
        B[std::atoi(result[0].c_str()) - 1 + LDB * (std::atoi(result[1].c_str()) - 1)] = std::stod(result[2]);
    }
  }

  int counter = 0;

  for(int i = 0; i < N; i++)
  {
    for(int j = 0; j < K; j++)
    {
      if(B[j + i * LDB] != 0 || !MTX.compare(""))
      {
        Bsparse[counter] = B[j + i * LDB];
        counter++;
      }
    }
  }

  f.close();

  return std::make_tuple(A, B, Bsparse, C, Cref);
}

template <typename T>
int post(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned* LDB, unsigned LDC, T* ALPHA, T* BETA, T* A, T* B, T* C, T* Cref, T DELTA) {

  if(*LDB == 0)
    *LDB = K;

  gemm_ref(M, N, K, LDA, *LDB, LDC, *ALPHA, *BETA, A, B, Cref);
    
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      // we use the relative error instead of the absolute error because of an issue we found for sparse single precision 
      // kernels presumably due to limited precision of floats
      if(std::abs((C[i + j * LDC] - Cref[i + j * LDC])) / Cref[i + j * LDC] > DELTA) {
        return 0;
      }
    }
  }

  return 1;
}
"""

setup_main = """
int main()
{
  std::vector<std::tuple<std::string, int>> results;

"""

setup_single_testcase = """
{{
  unsigned ldb = {ldb};
  {precision} alpha = {alpha};
  {precision} beta = {beta};
  {precision}* prefetch = nullptr;
  auto pointers = pre<{precision}>({m}, {n}, {k}, {lda}, ldb, {ldc}, "{mtx}");
  setup_prefetch(prefetch, std::get<3>(pointers), {n}, {ldc});
  {name}(std::get<0>(pointers), std::get<{sparse}>(pointers), std::get<3>(pointers), {alpha}, {beta}, nullptr);
  const auto result = post<{precision}>({m}, {n}, {k}, {lda}, &ldb, {ldc}, &alpha, &beta, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), {delta:.7f});
  results.push_back(std::make_tuple("{name}", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));
}}
"""

end_of_testsuite = """

  int correct = 0;
  for(int i = 0; i < results.size(); i++)
  {
    if(std::get<1>(results[i]))
    {
      ++correct;
      printf("%s succeeded.\\n", (std::get<0>(results[i])).c_str());
    }
    else
    {
      printf("%s failed!\\n", (std::get<0>(results[i])).c_str());
    }
  }

  printf("\\n%i out of %lu test successful!\\n", correct, results.size());

  return correct == results.size() ? 0 : 1;
}
"""


def generateMTX(k, n, nnz):
    assert (nnz <= k * n)
    os.makedirs(os.path.join(BASEDIR, 'mtx'), exist_ok=True)

    filename = os.path.join(BASEDIR, 'mtx', str(k) + 'x' + str(n) + '_' + str(nnz) + '.mtx')

    if os.path.isfile(filename):
        return filename

    with open(filename, 'w') as f:

      f.write('%%MatrixMarket matrix coordinate real general\n%\n' + str(k) + ' ' + str(n) + ' ' + str(nnz))

      zeros = set()

      for i in range(1, k + 1):
          for j in range(1, n + 1):
              zeros.add((i, j))

      nonzeros = random.sample(sorted(zeros), nnz)

      for entry in nonzeros:
          f.write('\n' + str(entry[0]) + ' ' + str(entry[1]) + ' ' + str(random.uniform(0.00001, 1000)))

    return filename


def make(kernels, arch):
    os.makedirs(os.path.join(BASEDIR, arch), exist_ok=True)
    f = open(os.path.join(BASEDIR, f'{arch}_testsuite.cpp'), 'w')

    f.write(head_of_testsuite)

    testcases = []

    for kern in kernels:

        arguments = ['pspamm-generator', str(kern.m), str(kern.n), str(kern.k), str(kern.lda), str(kern.ldb),
                     str(kern.ldc), str(kern.alpha), str(kern.beta)]

        if isinstance(kern, SparseKernel):
            arguments += ['--mtx_filename', kern.mtx]

        prec = 's' if kern.precision == Precision.SINGLE else 'd'
        arguments += ['--precision', prec]

        block_sizes = list(set(bs if len(bs) > 2 else (bs[0], bs[1], 1) for bs in kern.block_sizes))

        for bs in block_sizes:
            bm = bs[0]
            bn = bs[1]
            bk = bs[2]

            if arch.startswith("arm_sve"):
              veclen = int(arch[7:]) if arch[7:] != '' else 128
            else:
              veclen = int(arch[3:]) if arch[3:] != '' else 128
            assert veclen % 128 == 0
            reglen = veclen // 128
            v_len = (16 // kern.precision.size()) * reglen
            # this should be the same assertion as in ../scripts/max_arm_sve.py
            # ceiling division
            vm = -(bm // -v_len)
            v_size = v_len

            if arch.startswith("knl"):
              print(f'{bn} {bk} {vm} {bm} {v_size}')
              if not ((bn+bk) * vm <= 32) or not (kern.m % v_size) == 0 or not (bm % v_size) == 0:
                print(f'Skipping block size {bm}x{bn}x{bk} for {arch} / {prec}')
                continue
            elif arch.startswith("hsw"):
              if not ((bn+bk) * vm + bn * bk <= 16) or not (kern.m % v_size) == 0 or not (bm % v_size) == 0:
                print(f'Skipping block size {bm}x{bn}x{bk} for {arch} / {prec}')
                continue
            elif arch.startswith("arm_sve"):
              if not ((bn+bk) * vm + bn * bk <= 32):
                print(f'Skipping block size {bm}x{bn}x{bk} for {arch} / {prec}')
                continue
            elif arch.startswith("arm"):
              if not ((bn+bk) * vm + bn * bk <= 32) or not (kern.m % v_size) == 0 or not (bm % v_size) == 0:
                print(f'Skipping block size {bm}x{bn}x{bk} for {arch} / {prec}')
                continue

            name = kern.name + '_' + str(bm) + '_' + str(bn) + '_' + str(bk)

            additional_args = ['--output_funcname', name, '--output_filename', os.path.join(BASEDIR, arch, name + '.h'),
                               '--output_overwrite']
            additional_args += ['--bm', str(bm), '--bn', str(bn), '--bk', str(bk), '--arch', arch]

            try:
                print(' '.join(arguments + additional_args))
                subprocess.check_output(arguments + additional_args, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

            f.write('#include "' + arch + '/' + name + '.h"\n')

            if isinstance(kern, SparseKernel):
                mtx = kern.mtx
            else:
                mtx = ""

            testcases += [
              setup_single_testcase.format(
                m=kern.m, n=kern.n, k=kern.k, lda=kern.lda, ldb=kern.ldb, ldc=kern.ldc, alpha=kern.alpha,
                beta=kern.beta, mtx=mtx, delta=kern.delta, name=name, sparse=2 if kern.ldb == 0 else 1,
                precision=kern.precision.ctype())
            ]

    f.write('\n')

    f.write(function_definitions)
    f.write(setup_main)

    for testcase in testcases:
      f.write(testcase)

    f.write(end_of_testsuite)
