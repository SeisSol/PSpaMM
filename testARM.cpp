#include <stdlib.h> 
#include <cblas.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

#include "gemms.h"



#define M 8
#define N 56
#define K 56

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
  start = clock();
  for(int i = 0; i < 1; i++)
  {
    gemm(A,Bsparse,C);
  }
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("time used by sparse MM : %f\n", cpu_time_used);
/*
  start = clock();
  for(int i = 0; i < 1; i++)
  {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 4, 3, 56, 1, A, 4, B, 56, 1, C, 4);
  }
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  
  printf("time used by blas : %f\n", cpu_time_used);
*/
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
