#include <fstream>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <stdio.h>

#include "knl/gemm_sparse.h"
#include "knl/gemm_dense.h"
#include "knl/gemm_libxsmm_sparse.h"
#include "knl/gemm_libxsmm_dense.h"


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


int main(int argc, char** argv) {

  int num_threads = omp_get_max_threads();

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  int BETA = atoi(argv[4]);
  int S = atoi(argv[5]);
  int ITER = atoi(argv[6]);
  std::string mtx = argv[7];

  double *A;
  double *B;
  double *C;
  double *C1;
  double *C2;
  double *C3;
  double *C4;
  double *Bsparse;

  int resA = posix_memalign(reinterpret_cast<void **>(&A), 64, num_threads*M*K*sizeof(double));
  int resB = posix_memalign(reinterpret_cast<void **>(&B), 64, num_threads*K*N*sizeof(double));
  int resBsparse = posix_memalign(reinterpret_cast<void **>(&Bsparse), 64, num_threads*K*N*sizeof(double));  
  int resC = posix_memalign(reinterpret_cast<void **>(&C), 64, num_threads*M*N*sizeof(double));
  int resC1 = posix_memalign(reinterpret_cast<void **>(&C1), 64, num_threads*M*N*sizeof(double));
  int resC2 = posix_memalign(reinterpret_cast<void **>(&C2), 64, num_threads*M*N*sizeof(double));
  int resC3 = posix_memalign(reinterpret_cast<void **>(&C3), 64, num_threads*M*N*sizeof(double));
  int resC4 = posix_memalign(reinterpret_cast<void **>(&C4), 64, num_threads*M*N*sizeof(double));

  std::string line;
  std::ifstream f(mtx);
  getline(f, line);
  getline(f, line);

  int counter = 0;

  while(getline(f, line)) {
    for(int i = 0; i < num_threads; i++)
      B[i * K * N + counter] = std::stod(line);
    counter++;
  }

  for(int i = 0; i < num_threads*M*K; i++)
    A[i] = 1;

  counter = 0;
  
  for(int i = 0; i < N*K; i++)
  {
    if(B[i] != 0)
    {
      for(int j = 0; j < num_threads; j++)
        Bsparse[j * K * N + counter] = B[i];
      counter++;
    }
  }

  for(int i = 0; i < num_threads; i++)
    gemm_ref(M, N, K, &A[i * M * K], &B[i * K * N], BETA, &C[i * M * N]);

  double start, end;

  start = omp_get_wtime();
  #pragma omp parallel for
  for(int i = 0; i < ITER; i++)
    gemm_sparse(&A[omp_get_thread_num() * M * K],&Bsparse[omp_get_thread_num() * K * N],&C1[omp_get_thread_num() * M * N]);
  end = omp_get_wtime();
  double min_time_sparse = end - start;


  start = omp_get_wtime();
  #pragma omp parallel for
  for(int i = 0; i < ITER; i++)
    gemm_dense(&A[omp_get_thread_num() * M * K],&B[omp_get_thread_num() * K * N],&C2[omp_get_thread_num() * M * N]);
  end = omp_get_wtime();
  double min_time_dense = end - start;


  start = omp_get_wtime();
  #pragma omp parallel for
  for(int i = 0; i < ITER; i++)
    gemm_libxsmm_sparse(&A[omp_get_thread_num() * M * K],&B[omp_get_thread_num() * K * N],&C2[omp_get_thread_num() * M * N]);
  end = omp_get_wtime();
  double min_time_libxsmm_sparse = end - start;


  start = omp_get_wtime();
  #pragma omp parallel for
  for(int i = 0; i < ITER; i++)
    gemm_libxsmm_dense(&A[omp_get_thread_num() * M * K],&B[omp_get_thread_num() * K * N],&C2[omp_get_thread_num() * M * N]);
  end = omp_get_wtime();
  double min_time_libxsmm_dense = end - start;


 

  double max_deviation1 = 0;
  double max_deviation2 = 0;
  double max_deviation3 = 0;
  double max_deviation4 = 0;


  for(int i = 0; i < num_threads * N * M; i++)
  {
    if(std::abs(C[i] - C1[i]) > max_deviation1)
      max_deviation1 = std::abs(C[i] - C1[i]);

    if(std::abs(C[i] - C2[i]) > max_deviation2)
      max_deviation2 = std::abs(C[i] - C2[i]);

    if(std::abs(C[i] - C3[i]) > max_deviation3)
      max_deviation3 = std::abs(C[i] - C3[i]);

    if(std::abs(C[i] - C4[i]) > max_deviation4)
      max_deviation4 = std::abs(C[i] - C4[i]);
  }

  long sparseFLOP = M * S * ((long) ITER);
  long denseFLOP = M * N * K * ((long) ITER);

  double sparseGFLOPS = (sparseFLOP/ min_time_sparse) / 1000000000;
  double denseGFLOPS = (denseFLOP / min_time_dense) / 1000000000;
  double sparseGFLOPSlibxsmm = (sparseFLOP / min_time_libxsmm_sparse) / 1000000000;
  double denseGFLOPSlibxsmm = (denseFLOP / min_time_libxsmm_dense) / 1000000000;

  std::string state1;
  std::string state2;
  std::string state3;
  std::string state4;
  
  state1 = state2 = state3 = state4 = "FAIL";

  if(max_deviation1 < 0.00001)
    state1 = "SUCCESS";
  if(max_deviation2 < 0.00001)
    state2 = "SUCCESS";
  if(max_deviation3 < 0.00001)
    state3 = "SUCCESS";
  if(max_deviation4 < 0.00001)
    state4 = "SUCCESS";

  printf("\n%s %s %s %s %f %f %f %f", state1.c_str(), state2.c_str(), state3.c_str(), state4.c_str(), sparseGFLOPS, denseGFLOPS, sparseGFLOPSlibxsmm, denseGFLOPSlibxsmm);

  return 0;
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


//  printf("C\n");

//  for(int i = 0; i < M * N; i++)
//  {
//    if(i % N == 0)
//      printf("\n");
//    printf("%f  ", C[((i * M) % (M * N)) + i / N]);
//  }
