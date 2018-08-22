#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>

#include "gemms_knl_dense.h"
#include "gemms_knl_sparse.h"
#include "gemms_libxsmm_dense.h"
#include "gemms_libxsmm_sparse.h"

#define M 8
#define N 56
#define K 56
#define S 294
#define ITER 10000000

void gemm_ref(unsigned m, unsigned n, unsigned k, double* A, double* B, double beta, double* C){
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
  double *C1;
  double *C2;
  double *C3;
  double *C4;
  double *Bsparse;

  int resA = posix_memalign(reinterpret_cast<void **>(&A), 64, M*K*sizeof(double));
  int resB = posix_memalign(reinterpret_cast<void **>(&B), 64, K*N*sizeof(double));
  int resBsparse = posix_memalign(reinterpret_cast<void **>(&Bsparse), 64, K*N*sizeof(double));
  int resC1 = posix_memalign(reinterpret_cast<void **>(&C1), 64, M*N*sizeof(double));
  int resC2 = posix_memalign(reinterpret_cast<void **>(&C2), 64, M*N*sizeof(double));
  int resC3 = posix_memalign(reinterpret_cast<void **>(&C3), 64, M*N*sizeof(double));
  int resC4 = posix_memalign(reinterpret_cast<void **>(&C4), 64, M*N*sizeof(double));

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

  printf("B\n");

  for(int i = 0; i < N * K; i++)
  {
    if(i % N == 0)
      printf("\n");
    printf("%f  ", B[((i * K) % (K * N)) + i / N]);
  }

  printf("\n");

  clock_t start, end;
  double cpu_time_used;

  double min_time_sparse = 999999999999999999; 

  for(int i = 0; i < ITER/20; i++)
  {
  //  gemm_sparse(A,Bsparse,C1);
  }

  for(int j = 0; j < 1; j++)
  {
    start = clock();
    for(int i = 0; i < ITER; i++)
    {
    //  gemm_sparse(A,Bsparse,C1);
    }
    end = clock();
    if(((double) (end - start)) < min_time_sparse)
      min_time_sparse = ((double) (end - start));
  }



  double min_time_dense = 999999999999999999; 

  for(int i = 0; i < ITER/20; i++)
  {
  //  gemm_dense(A,B,C2);
  }

  for(int j = 0; j < 1; j++)
  {
    start = clock();
    for(int i = 0; i < ITER; i++)
    {
   //   gemm_dense(A,B,C2);
    }
    end = clock();
    if(((double) (end - start)) < min_time_dense)
      min_time_dense = ((double) (end - start));
  }



  double min_time_libxsmm_dense = 999999999999999999; 

  for(int i = 0; i < ITER/20; i++)
  {
   // gemm_libxsmm_dense(A,B,C3);
  }

  for(int j = 0; j < 1; j++)
  {
    start = clock();
    for(int i = 0; i < ITER; i++)
    {
   //   gemm_libxsmm_dense(A,B,C3);
    }
    end = clock();
    if(((double) (end - start)) < min_time_libxsmm_dense)
      min_time_libxsmm_dense = ((double) (end - start));
  }
 
  double min_time_libxsmm_sparse = 999999999999999999; 

  for(int i = 0; i < ITER/20; i++)
  {
    gemm_libxsmm_sparse(A,Bsparse,C4);
  }

  for(int j = 0; j < 1; j++)
  {
    start = clock();
    for(int i = 0; i < ITER; i++)
    {
      gemm_libxsmm_sparse(A,Bsparse,C4);
    }
    end = clock();
    if(((double) (end - start)) < min_time_libxsmm_sparse)
      min_time_libxsmm_sparse = ((double) (end - start));
  }
  
 

  printf("\nC1\n");

  for(int i = 0; i < M * N; i++)
  {
    if(i % N == 0)
      printf("\n");
    printf("%f ", C1[((i * M) % (M * N)) + i / N]);
  }



  printf("\nC2\n");

  for(int i = 0; i < M * N; i++)
  {
    if(i % N == 0)
      printf("\n");
    printf("%f ", C2[((i * M) % (M * N)) + i / N]);
  }



  printf("\nC3\n");

  for(int i = 0; i < M * N; i++)
  {
    if(i % N == 0)
      printf("\n");
    printf("%f ", C3[((i * M) % (M * N)) + i / N]);
  }



  printf("\nC4\n");

  for(int i = 0; i < M * N; i++)
  {
    if(i % N == 0)
      printf("\n");
    printf("%f ", C4[((i * M) % (M * N)) + i / N]);
  }

/*
  for(int i = 0; i < N * M; i++)
  {
    printf("C1[%i] = %f\n", i, C1[i]);
    printf("C2[%i] = %f\n", i, C2[i]);
    printf("C3[%i] = %f\n", i, C3[i]);
    printf("C4[%i] = %f\n", i, C4[i]);
    printf("------------------------\n");
 }
*/


  min_time_sparse = min_time_sparse / CLOCKS_PER_SEC;
  min_time_dense = min_time_dense / CLOCKS_PER_SEC;
  min_time_libxsmm_sparse = min_time_libxsmm_sparse / CLOCKS_PER_SEC;
  min_time_libxsmm_dense = min_time_libxsmm_dense / CLOCKS_PER_SEC;

  printf("\n");

  long sparseFLOP = M * S * ((long) ITER);
  long denseFLOP = M * N * K * ((long) ITER);

  double sparseFLOPS =sparseFLOP/ min_time_sparse;
  double denseFLOPS = denseFLOP / min_time_dense;
  double sparseFLOPSlibxsmm = sparseFLOP / min_time_libxsmm_sparse;
  double denseFLOPSlibxsmm = denseFLOP / min_time_libxsmm_dense;


  printf("Matrix Multiplication M = %i, N = %i, K = %i, non-zero elements: %i\n\n", M, N, K, S);

  printf("dense MM FLOP:  %ld\n", denseFLOP);
  printf("sparse MM FLOP: %ld\n", sparseFLOP);

  printf("time used by sparse MM:      %f\n", min_time_sparse);
  printf("time used by dense MM:       %f\n", min_time_dense);
  printf("time used by libxsmm dense:  %f\n", min_time_libxsmm_dense);
  printf("time used by libxsmm sparse: %f\n", min_time_libxsmm_sparse);

  printf("GFLOPS by sparse MM:      %f\n", sparseFLOPS / 1000000000);
  printf("GFLOPS by dense MM:       %f\n", denseFLOPS / 1000000000);
  printf("GFLOPS by libxsmm sparse: %f\n", sparseFLOPSlibxsmm / 1000000000);
  printf("GFLOPS by libxsmm dense:  %f\n", denseFLOPSlibxsmm / 1000000000);

  return 0;
}
