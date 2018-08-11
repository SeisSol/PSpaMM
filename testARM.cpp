#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>

#include "gemms_sparse.h"
#include "gemms_libxsmm.h"




#define M 8
#define N 56
#define K 56

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

  printf("A\n");

  for(int i = 0; i < M*K; i++)
  {
    if(i % K == 0)
      printf("\n");
    printf("%f  ", A[((i * M) % (M * K)) + i / K]);
  }

  printf("\n");

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

  for(int i = 0; i < 1000000; i++)
  {
    gemm(A,Bsparse,C);
  }

  for(int j = 0; j < 1; j++)
  {
    start = clock();
    for(int i = 0; i < 10000000; i++)
    {
      gemm(A,Bsparse,C);
    }
    end = clock();
    if(((double) (end - start)) < min_time_sparse)
      min_time_sparse = ((double) (end - start));
  }

  printf("time used by sparse MM : %f\n", min_time_sparse / CLOCKS_PER_SEC);


  double min_time_libxsmm = 999999999999999999; 

  for(int i = 0; i < 1000000; i++)
  {
    gemm_libxsmm(A,B,C);
  }

  for(int j = 0; j < 1; j++)
  {
    start = clock();
    for(int i = 0; i < 10000000; i++)
    {
      gemm_libxsmm(A,B,C);
    }
    end = clock();
    if(((double) (end - start)) < min_time_libxsmm)
      min_time_libxsmm = ((double) (end - start));
  }
  
  printf("time used by dense libxsmm : %f\n", min_time_libxsmm / CLOCKS_PER_SEC);

  printf("C\n");

  for(int i = 0; i < M * N; i++)
  {
    if(i % N == 0)
      printf("\n");
    printf("%f  ", C[((i * M) % (M * N)) + i / N]);
  }

  printf("\n");

  return 0;
}
