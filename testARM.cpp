#include <stdlib.h>
#include <cblas.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>


#include "gemms_arm_sparse.h"
#include "gemms_arm_dense.h"

#define M 8
#define N 56
#define K 56
#define S 294
#define ITER 10000000

void gemm_ref(unsigned m, unsigned n, unsigned k, double* A, double* B, double beta, double* C) {
  if (beta == 0.0) {
    memset(C, 0, m*n * sizeof(double));
  }
  for (unsigned col = 0; col < n; ++col) {
    for (unsigned row = 0; row < m; ++row) {
      for (unsigned s = 0; s < k; ++s) {
        C[row + col * m] += A[row + s * m] * B[s + col * k];
      }
    }
  }
}


int main(void) {

  double *A;
  double *B;
  double *C;
  double *Bsparse;

  int resA = posix_memalign(reinterpret_cast<void **>(&A), 64, M*K*sizeof(double));
  int resB = posix_memalign(reinterpret_cast<void **>(&B), 64, K*N*sizeof(double));
  int resBsparse = posix_memalign(reinterpret_cast<void **>(&Bsparse), 64, K*N*sizeof(double));
  int resC = posix_memalign(reinterpret_cast<void **>(&C), 64, M*N*sizeof(double));

  std::string line;
  std::ifstream f("56x56.mtx");
  getline(f, line);
  getline(f, line);

  int counter = 0;

  while(getline(f, line)) {
    B[counter] = std::stod(line);
    counter++;
  }

  for(int i = 0; i < M*K; i++)
  {
    A[i] = 1;
  }

  counter = 0;
  
  for(int i = 0; i < N*K; i++)
  {
    if(B[i] != 0)
    {
      Bsparse[counter] = B[i];
      counter++;
    }
  }

//  printf("A\n");

//  for(int i = 0; i < M*K; i++)
//  {
//    if(i % K == 0)
//      printf("\n");
//    printf("%f  ", A[((i * M) % (M * K)) + i / K]);
//  }

//  printf("\n");

//  printf("B\n");

//  for(int i = 0; i < N * K; i++)
//  {
//    if(i % N == 0)
//      printf("\n");
//    printf("%f  ", B[((i * K) % (K * N)) + i / N]);
//  }

//  printf("\n");


  clock_t start, end;
  double cpu_time_used;

  double min_time_sparse = 999999999999999999; 

  for(int i = 0; i < ITER/20; i++)
  {
    gemm_sparse(A,Bsparse,C);
  }

  for(int j = 0; j < 1; j++)
  {
    start = clock();
    for(int i = 0; i < ITER; i++)
    {
      gemm_sparse(A,Bsparse,C);
    }
    end = clock();
    if(((double) (end - start)) < min_time_sparse)
      min_time_sparse = ((double) (end - start));
  }



  double min_time_dense = 999999999999999999; 

  for(int i = 0; i < ITER/20; i++)
  {
    gemm_dense(A,B,C);
  }

  for(int j = 0; j < 1; j++)
  {
    start = clock();
    for(int i = 0; i < ITER; i++)
    {
      gemm_dense(A,B,C);
    }
    end = clock();
    if(((double) (end - start)) < min_time_dense)
      min_time_dense = ((double) (end - start));
  }



  double min_time_openblas = 999999999999999999; 

  for(int i = 0; i < ITER/20; i++)
  {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, M, B, K, 0, C, M);
  }

  for(int j = 0; j < 1; j++)
  {
    start = clock();
    for(int i = 0; i < ITER; i++)
    {
       cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, M, B, K, 0, C, M);
    }
    end = clock();
    if(((double) (end - start)) < min_time_openblas)
      min_time_openblas = ((double) (end - start));
  }
  

//  printf("C\n");

//  for(int i = 0; i < M * N; i++)
//  {
//    if(i % N == 0)
//      printf("\n");
//    printf("%f  ", C[((i * M) % (M * N)) + i / N]);
//  }

  min_time_sparse = min_time_sparse / CLOCKS_PER_SEC;
  min_time_dense = min_time_dense / CLOCKS_PER_SEC;
  min_time_openblas = min_time_openblas / CLOCKS_PER_SEC;

  printf("\n");

  long sparseFLOP = M * S * ((long) ITER);
  long denseFLOP = M * N * K * ((long) ITER);

  double sparseFLOPS =sparseFLOP/ min_time_sparse;
  double denseFLOPS = denseFLOP / min_time_dense;
  double FLOPSopenblas = denseFLOP / min_time_openblas;


  printf("Matrix Multiplication M = %i, N = %i, K = %i, non-zero elements: %i\n\n", M, N, K, S);

  printf("dense MM FLOP:  %ld\n", denseFLOP);
  printf("sparse MM FLOP: %ld\n", sparseFLOP);

  printf("time used by sparse MM:      %f\n", min_time_sparse);
  printf("time used by dense MM:       %f\n", min_time_dense);
  printf("time used by dense openblas: %f\n", min_time_openblas);

  printf("GFLOPS by sparse MM: %f\n", sparseFLOPS / 1000000000);
  printf("GFLOPS by dense MM:  %f\n", denseFLOPS / 1000000000);
  printf("GFLOPS by openbblas: %f\n", FLOPSopenblas / 1000000000);

  return 0;
}
