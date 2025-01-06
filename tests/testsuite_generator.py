from collections import namedtuple
import subprocess
import numpy as np
import random
import sys
import os.path
from pspamm.codegen.precision import *

BASEDIR = 'build'

TestKernel = namedtuple('TestKernel', 'name precision m n k lda ldb ldc alpha beta block_sizes amtx bmtx delta')

head_of_testsuite = """#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <tuple>
#include <iomanip>


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
std::tuple<T*, T*, T*, T*, T*, T*> pre(const std::string& name, unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC, const std::string& AMTX, const std::string& BMTX) {

  std::cout << name << ": " << std::flush;

  if(LDB == 0) {
    LDB = K;
  }
  if(LDA == 0) {
    LDA = M;
  }

  T* A;
  T* Asparse;
  T* B;
  T* Bsparse;
  T* Cref;
  T* C;

  int resA = posix_memalign(reinterpret_cast<void **>(&A), 64, LDA*LDB*sizeof(T));
  int resAsparse = posix_memalign(reinterpret_cast<void **>(&Asparse), 64, LDA*LDB*sizeof(T));  
  int resB = posix_memalign(reinterpret_cast<void **>(&B), 64, LDB*N*sizeof(T));
  int resBsparse = posix_memalign(reinterpret_cast<void **>(&Bsparse), 64, LDB*N*sizeof(T));  
  int resCref = posix_memalign(reinterpret_cast<void **>(&Cref), 64, LDC*N*sizeof(T));
  int resC = posix_memalign(reinterpret_cast<void **>(&C), 64, LDC*N*sizeof(T));

  std::string line;

  std::ifstream bf(BMTX);
  getline(bf, line);
  getline(bf, line);
  getline(bf, line);

  std::ifstream af(AMTX);
  getline(af, line);
  getline(af, line);
  getline(af, line);

  for(int i = 0; i < LDB*N; i++) {
    if(BMTX.compare("")) {
      B[i] = 0;
    }
    else {
      B[i] = (T)rand() / RAND_MAX;
    }
  }

  for(int i = 0; i < LDA*LDB; i++) {
    if(AMTX.compare("")) {
      A[i] = 0;
    }
    else {
      A[i] = (T)rand() / RAND_MAX;
    }
  }

  for(int i = 0; i < LDC*N; i++)
  {
    Cref[i] = (T)rand() / RAND_MAX;
    C[i] = Cref[i];
  }

  if(BMTX.compare(""))
  {
    while(getline(bf, line)) {
      std::vector<std::string> result;
      std::istringstream iss(line);
      for(std::string s; iss >> s; ) {
        result.push_back(s);
      }
      if(std::atoi(result[0].c_str()) <= K && std::atoi(result[1].c_str()) <= N) {
        B[std::atoi(result[0].c_str()) - 1 + LDB * (std::atoi(result[1].c_str()) - 1)] = std::stod(result[2]);
      }
    }
  }
  if(AMTX.compare(""))
  {
    while(getline(af, line)) {
      std::vector<std::string> result;
      std::istringstream iss(line);
      for(std::string s; iss >> s; ) {
        result.push_back(s);
      }
      if(std::atoi(result[0].c_str()) <= M && std::atoi(result[1].c_str()) <= K) {
        A[std::atoi(result[0].c_str()) - 1 + LDA * (std::atoi(result[1].c_str()) - 1)] = std::stod(result[2]);
      }
    }
  }

  {
    int counter = 0;
    for(int i = 0; i < N; i++)
    {
      for(int j = 0; j < K; j++)
      {
        if(B[j + i * LDB] != 0 || !BMTX.compare(""))
        {
          Bsparse[counter] = B[j + i * LDB];
          ++counter;
        }
      }
    }
  }
  {
    int counter = 0;
    for(int i = 0; i < K; i++)
    {
      for(int j = 0; j < M; j++)
      {
        if(A[j + i * LDA] != 0 || !AMTX.compare(""))
        {
          Asparse[counter] = A[j + i * LDA];
          ++counter;
        }
      }
    }
  }

  af.close();
  bf.close();

  return std::make_tuple(A, Asparse, B, Bsparse, C, Cref);
}

template <typename T>
bool post(unsigned M, unsigned N, unsigned K, unsigned* LDA, unsigned* LDB, unsigned LDC, T* ALPHA, T* BETA, T* A, T* B, T* C, T* Cref, T DELTA) {

  if(*LDB == 0) {
    *LDB = K;
  }
  if(*LDA == 0) {
    *LDA = M;
  }

  gemm_ref(M, N, K, *LDA, *LDB, LDC, *ALPHA, *BETA, A, B, Cref);
  
  double diffAbsMax = 0;
  double diffRelMax = 0;
  int failedCount = 0;
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      // we use the relative error instead of the absolute error because of an issue we found for sparse single precision 
      // kernels presumably due to limited precision of floats
      const double diffAbs = std::abs((static_cast<double>(C[i + j * LDC]) - static_cast<double>(Cref[i + j * LDC])));
      const double diffRel = diffAbs / std::abs(static_cast<double>(Cref[i + j * LDC]));

      diffAbsMax = std::max(diffAbs, diffAbsMax);
      diffRelMax = std::max(diffRel, diffRelMax);

      failedCount += diffRel > DELTA ? 1 : 0;

      if(diffRel > DELTA) {
        // use for more detailed test outputs
        // std::cout << i << " " << j << " " << diffRel << " " << C[i+j*LDC] << " " << Cref[i+j*LDC] << std::endl;
      }
    }
  }

  const bool failed = failedCount > 0;

  const std::string resultString = failed ? "fail" : "success";

  std::cout << std::scientific << resultString << " " << failedCount << " / " << M*N << " (abs: " << diffAbsMax << ", rel: " << diffRelMax << ")" << std::endl;

  return !failed;
}
"""

setup_main = """
int main()
{{
  int results = 0;
  int correct = 0;

  std::cout << "Running tests for {arch}" << std::endl;

"""

setup_single_testcase = """
{{
  unsigned lda = {lda};
  unsigned ldb = {ldb};
  {precision} alpha = {alpha};
  {precision} beta = {beta};
  {precision}* prefetch = nullptr;
  auto pointers = pre<{precision}>(\"{name}\", {m}, {n}, {k}, lda, ldb, {ldc}, "{amtx}", "{bmtx}");
  setup_prefetch(prefetch, std::get<4>(pointers), {n}, {ldc});
  {name}(std::get<{asparse}>(pointers), std::get<{bsparse}>(pointers), std::get<4>(pointers), {alpha}, {beta}, nullptr);
  const auto result = post<{precision}>({m}, {n}, {k}, &lda, &ldb, {ldc}, &alpha, &beta, std::get<0>(pointers), std::get<2>(pointers), std::get<4>(pointers), std::get<5>(pointers), {delta:.15e});
  
  if (result) {{
    ++correct;
  }}
  ++results;

  free(std::get<0>(pointers));
  free(std::get<1>(pointers));
  free(std::get<2>(pointers));
  free(std::get<3>(pointers));
  free(std::get<4>(pointers));
  free(std::get<5>(pointers));
}}
"""

end_of_testsuite = """

  std::cout << correct << " out of " << results << " succeeded." << std::endl;

  return correct == results ? 0 : 1;
}
"""


def generateMTX(k, n, nnz, bk=1, bn=1):
    random.seed(k*n + nnz)

    if k < bk:
      k = bk
    if n < bn:
      n = bn

    assert k % bk == 0
    assert n % bn == 0

    assert nnz <= k * n

    true_nzz = nnz * bk * bn

    os.makedirs(os.path.join(BASEDIR, 'mtx'), exist_ok=True)

    filename = os.path.join(BASEDIR, 'mtx', f'{k}-{bk}-{n}-{bn}-{nnz}.mtx')

    if os.path.isfile(filename):
        return filename

    with open(filename, 'w') as f:

      f.write(f'%%MatrixMarket matrix coordinate real general\n%\n{k} {n} {true_nzz}')

      zeros = set()

      for i in range(1, k + 1, bk):
        for j in range(1, n + 1, bn):
          zeros.add((i, j))

      nonzeros = random.sample(sorted(zeros), nnz)

      for entry in nonzeros:
        for ii in range(bk):
          for jj in range(bn):
            f.write('\n' + str(entry[0] + ii) + ' ' + str(entry[1] + jj) + ' ' + str(random.uniform(0.00001, 1000)))

    return filename


def make(kernels, arch):
    os.makedirs(os.path.join(BASEDIR, arch), exist_ok=True)
    f = open(os.path.join(BASEDIR, f'{arch}_testsuite.cpp'), 'w')

    f.write(head_of_testsuite)

    testcases = []

    for kern in kernels:

        arguments = ['pspamm-generator', str(kern.m), str(kern.n), str(kern.k), str(kern.lda), str(kern.ldb),
                     str(kern.ldc), str(kern.alpha), str(kern.beta)]

        if kern.amtx is not None:
          arguments += ['--amtx_filename', kern.amtx]
        if kern.bmtx is not None:
          arguments += ['--bmtx_filename', kern.bmtx]

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
            elem128 = (16 // kern.precision.size())

            if arch.startswith("knl"):
              if not ((bn+bk) * vm <= 32):
                print(f'Skipping block size {bm}x{bn}x{bk} for {arch} / {prec}')
                continue
            elif arch.startswith("hsw"):
              if not ((bn+bk) * vm + bn * bk <= 16) or not (kern.m % v_size) == 0 or not (bm % v_size) == 0:
                print(f'Skipping block size {bm}x{bn}x{bk} for {arch} / {prec}')
                continue
            elif arch.startswith("arm_sve"):
              vkext = -(bk // -elem128)
              isvkext = bn*vkext < 16 if elem128 == 2 else bn*vkext < 8
              vk = vkext if isvkext else bk
              if not ((bn+bk) * vm + bn * vk <= 32):
                print(f'Skipping block size {bm}x{bn}x{bk} for {arch} / {prec}')
                continue
            elif arch.startswith("arm"):
              vk = -(bk // -elem128)
              if not ((bn+bk) * vm + bn * vk <= 32) or not (kern.m % v_size) == 0 or not (bm % v_size) == 0:
                print(f'Skipping block size {bm}x{bn}x{bk} for {arch} / {prec}')
                continue
            elif arch.startswith("rvv"):
              if not ((bn+bk) * vm <= 32) or not (bn*bk <= 30) or not (kern.m % v_size) == 0 or not (bm % v_size) == 0:
                print(f'Skipping block size {bm}x{bn}x{bk} for {arch} / {prec}')
                continue

            name = f'{kern.name}_{kern.precision}_{bm}_{bn}_{bk}'

            additional_args = ['--output_funcname', name, '--output_filename', os.path.join(BASEDIR, arch, name + '.h'),
                               '--output_overwrite']
            additional_args += ['--bm', str(bm), '--bn', str(bn), '--bk', str(bk), '--arch', arch]

            try:
                print(' '.join(arguments + additional_args))
                subprocess.check_output(arguments + additional_args, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"The command\n{' '.join(e.cmd)}\n returned with an error (code {e.returncode}):\n{e.output.decode('utf-8')}")

            f.write('#include "' + arch + '/' + name + '.h"\n')

            testcases += [
              setup_single_testcase.format(
                m=kern.m, n=kern.n, k=kern.k, lda=kern.lda, ldb=kern.ldb, ldc=kern.ldc, alpha=kern.alpha,
                beta=kern.beta, delta=kern.delta, name=name,
                amtx=kern.amtx or '', bmtx = kern.bmtx or '',
                asparse=1 if kern.lda == 0 else 0, bsparse=3 if kern.ldb == 0 else 2,
                precision=kern.precision.ctype())
            ]

    f.write('\n')

    f.write(function_definitions)
    f.write(setup_main.format(arch=arch))

    for testcase in testcases:
      f.write(testcase)

    f.write(end_of_testsuite)
