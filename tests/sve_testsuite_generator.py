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

# include function definitions for single precision case
single_precision_function_definitions = """
void fgemm_ref(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC, float ALPHA, float BETA, float* A, float* B, float* C) {
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

std::tuple<float*, float*, float*, float*, float*> fpre(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC, std::string MTX) {

  if(LDB == 0)
    LDB = K;

  float *A;
  float *B;
  float *Bsparse;
  float *Cref;
  float *C;

  int resA = posix_memalign(reinterpret_cast<void **>(&A), 64, LDA*LDB*sizeof(float));
  int resB = posix_memalign(reinterpret_cast<void **>(&B), 64, LDB*N*sizeof(float));
  int resBsparse = posix_memalign(reinterpret_cast<void **>(&Bsparse), 64, LDB*N*sizeof(float));  
  int resCref = posix_memalign(reinterpret_cast<void **>(&Cref), 64, LDC*N*sizeof(float));
  int resC = posix_memalign(reinterpret_cast<void **>(&C), 64, LDC*N*sizeof(float));

  std::string line;
  std::ifstream f(MTX);
  getline(f, line);
  getline(f, line);
  getline(f, line);

  for(int i = 0; i < LDA*LDB; i++)
    A[i] = (float)rand() / RAND_MAX;

  for(int i = 0; i < LDB*N; i++)
    if(MTX.compare(""))
      B[i] = 0;
    else 
      B[i] = (float)rand() / RAND_MAX;

  for(int i = 0; i < LDC*N; i++)
  {
    Cref[i] = (float)rand() / RAND_MAX;
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

int fpost(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC, float ALPHA, float BETA, float* A, float* B, float* C, float* Cref, float DELTA) {

  if(LDB == 0)
    LDB = K;

  fgemm_ref(M, N, K, LDA, LDB, LDC, ALPHA, BETA, A, B, Cref);
        
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++)
      if(std::abs(C[i + j * LDC] - Cref[i + j * LDC]) > DELTA)
        return 0;

  return 1;
}

"""
def generateMTX(k, n, nnz):
    return test_generator.generateMTX(k, n, nnz)

def make(kernels, arch):
    f = open('sve_testsuite.cpp', 'w')

    f.write(test_generator.head_of_testsuite)

    for kern in kernels:
        arguments = [sys.executable, './../pspamm.py', str(kern.m), str(kern.n), str(kern.k), str(kern.lda),
                     str(kern.ldb), str(kern.ldc), str(kern.alpha), str(kern.beta)]

        if isinstance(kern, SparseKernel):
            arguments += ['--mtx_filename', kern.mtx]

        prec = 's' if isinstance(kern, SparseKernelS) or isinstance(kern, DenseKernelS) else 'd'
        arguments += ['--precision', prec]

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
                assert ((bn + 1) * (bm / 8) + bn <= 32)

            name = kern.name + '_' + str(bm) + '_' + str(bn)

            additional_args = ['--output_funcname', name, '--output_filename', arch + '/' + name + '.h',
                               '--output_overwrite']
            additional_args += ['--bm', str(bm), '--bn', str(bn), '--arch', arch]

            try:
                subprocess.check_output(arguments + additional_args, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

            f.write('#include "' + arch + '/' + kern.name + '_' + str(bm) + '_' + str(bn) + '.h"\n')

    f.write('\n')

    f.write(single_precision_function_definitions)

    f.write(test_generator.function_definitions)

    for kern in kernels:

        block_sizes = list(set(kern.block_sizes))

        for bs in block_sizes:
            bm = bs[0]
            bn = bs[1]
            name = kern.name + '_' + str(bm) + '_' + str(bn)

            if isinstance(kern, SparseKernel):
                mtx = kern.mtx
            else:
                mtx = ""
            # for double precision: set prec to '' to conform to test_generator.function_definitions
            prec = 'f' if isinstance(kern, SparseKernelS) or isinstance(kern, DenseKernelS) else ''

            f.write("""
  ldb = {ldb}; alpha = {alpha}; beta = {beta};
  {p}pointers = {p}pre({m}, {n}, {k}, {lda}, ldb, {ldc}, "{mtx}");
  {name}(std::get<0>({p}pointers), std::get<{sparse}>({p}pointers), std::get<3>({p}pointers), alpha, beta, nullptr);
  result = {p}post({m}, {n}, {k}, {lda}, &ldb, {ldc}, &alpha, &beta, std::get<0>({p}pointers), std::get<1>({p}pointers), std::get<3>({p}pointers), std::get<4>({p}pointers), {delta:.7f});
  results.push_back(std::make_tuple("{name}", result));
  free(std::get<0>({p}pointers)); free(std::get<1>({p}pointers)); free(std::get<2>({p}pointers)); free(std::get<3>({p}pointers)); free(std::get<4>({p}pointers));
""".format(m=kern.m, n=kern.n, k=kern.k, lda=kern.lda, ldb=kern.ldb, ldc=kern.ldc, alpha=kern.alpha, beta=kern.beta,
           mtx=mtx, delta=kern.delta, name=name, sparse=2 if kern.ldb == 0 else 1, p=prec))

    f.write(test_generator.end_of_testsuite)
