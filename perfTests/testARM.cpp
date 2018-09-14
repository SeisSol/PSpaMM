#include <fstream>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <stdio.h>

#include "arm/gemm_sparse.h"
#include "arm/gemm_dense.h"


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
  double *Bsparse;

  int resA = posix_memalign(reinterpret_cast<void **>(&A), 64, num_threads*M*K*sizeof(double));
  int resB = posix_memalign(reinterpret_cast<void **>(&B), 64, num_threads*K*N*sizeof(double));
  int resBsparse = posix_memalign(reinterpret_cast<void **>(&Bsparse), 64, num_threads*K*N*sizeof(double));  
  int resC = posix_memalign(reinterpret_cast<void **>(&C), 64, num_threads*M*N*sizeof(double));
  int resC1 = posix_memalign(reinterpret_cast<void **>(&C1), 64, num_threads*M*N*sizeof(double));
  int resC2 = posix_memalign(reinterpret_cast<void **>(&C2), 64, num_threads*M*N*sizeof(double));
  int resC3 = posix_memalign(reinterpret_cast<void **>(&C3), 64, num_threads*M*N*sizeof(double));

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


//  start = omp_get_wtime();
//  #pragma omp parallel for
//  for(int i = 0; i < ITER; i++)
//    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, &A[omp_get_thread_num() * M * K], M, &B[omp_get_thread_num() * K * N], K, 0, &C3[omp_get_thread_num() * M * N], M);
//  end = omp_get_wtime();
  double min_time_openblas = end - start;
 

  double max_deviation1 = 0;
  double max_deviation2 = 0;
  double max_deviation3 = 0;


  for(int i = 0; i < num_threads * N * M; i++)
  {
    if(std::abs(C[i] - C1[i]) > max_deviation1)
      max_deviation1 = std::abs(C[i] - C1[i]);

    if(std::abs(C[i] - C2[i]) > max_deviation2)
      max_deviation2 = std::abs(C[i] - C2[i]);

    if(std::abs(C[i] - C3[i]) > max_deviation3)
      max_deviation3 = std::abs(C[i] - C3[i]);
  }

//  printf("Matrix Multiplication M = %i, N = %i, K = %i, non-zero elements: %i\n\n", M, N, K, S);

//  printf("Max deviation:\n");
//  printf("sparse: %f\n", max_deviation1);
//  printf("dense: %f\n", max_deviation2);
//  printf("openblas: %f\n", max_deviation3);

//  printf("\n");

  long sparseFLOP = M * S * ((long) ITER);
  long denseFLOP = M * N * K * ((long) ITER);

  double sparseFLOPS = sparseFLOP/ min_time_sparse;
  double denseFLOPS = denseFLOP / min_time_dense;
  double FLOPSopenblas = denseFLOP / min_time_openblas;



//  printf("dense MM FLOP:  %ld\n", denseFLOP);
//  printf("sparse MM FLOP: %ld\n", sparseFLOP);

//  printf("time used by sparse MM:      %f\n", min_time_sparse);
//  printf("time used by dense MM:       %f\n", min_time_dense);
//  printf("time used by dense openblas: %f\n", min_time_openblas);

//  printf("GFLOPS by sparse MM: %f\n", sparseFLOPS / 1000000000);
//  printf("GFLOPS by dense MM:  %f\n", denseFLOPS / 1000000000);
//  printf("GFLOPS by openbblas: %f\n", FLOPSopenblas / 1000000000);

//  printf("\n");

  std::string state1;
  std::string state2;
  
  if(max_deviation1 < 0.00001)
    state1 = "SUCCESS";
  else
    state1 = "FAIL";

  if(max_deviation2 < 0.00001)
    state2 = "SUCCESS";
  else
    state2 = "FAIL";

  printf("\n%s %s %f %f", state1.c_str(), state2.c_str(), sparseFLOPS/1000000000, denseFLOPS/1000000000);

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
