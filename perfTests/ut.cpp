#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <tuple>

#include "knl/test1_8_28.h"

void gemm_ref(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC, double BETA, double* A, double* B, double* C) {
  if (BETA == 0.0) {
    memset(C, 0, M * N * sizeof(double));
  }
  for (unsigned col = 0; col < N; ++col) {
    for (unsigned row = 0; row < M; ++row) {
      for (unsigned s = 0; s < K; ++s) {
        C[row + col * LDC] += A[row + s * LDA] * B[s + col * LDB];
      }
    }
  }
}


int test1(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC, double BETA, std::string MTX, double DELTA) {
  double *A;
  double *B;
  double *Bsparse;
  double *Cref;
  double *C;

  int resA = posix_memalign(reinterpret_cast<void **>(&A), 64, LDA*K*sizeof(double));
  int resB = posix_memalign(reinterpret_cast<void **>(&B), 64, K*N*sizeof(double));
  int resBsparse = posix_memalign(reinterpret_cast<void **>(&Bsparse), 64, K*N*sizeof(double));  
  int resCref = posix_memalign(reinterpret_cast<void **>(&Cref), 64, LDC*N*sizeof(double));
  int resC = posix_memalign(reinterpret_cast<void **>(&C), 64, LDC*N*sizeof(double));

  std::string line;
  std::ifstream f(MTX);
  getline(f, line);
  getline(f, line);

  for(int i = 0; i < LDA*K; i++)
    A[i] = (double)rand() / RAND_MAX;

  for(int i = 0; i < K*N; i++)
    B[i] = 0;

  for(int i = 0; i < LDC*N; i++)
  {
    Cref[i] = (double)rand() / RAND_MAX;
    C[i] = Cref[i];
  }


  int counter = 0;

  while(getline(f, line)) {
    std::vector<std::string> result;
    std::istringstream iss(s);
    for(std::string s; iss >> s; )
      result.push_back(s);

    B[result[0] + K * result[1]] = std::stod(result[2]);
    Bsparse[counter] = std::stod(result[2]);

    counter++;
  }

  gemm_ref(M, N, K, LDA, B, LDC, BETA, A, B, Cref);

  test1_8_28(A,Bsparse, C);

  double max_deviation = 0;

  for(int i = 0; i < N * M; i++)
    if(std::abs(C[i] - C1[i]) > max_deviation)
      max_deviation = std::abs(Cref[i] - C[i]);

  if(max_deviation < DELTA)
    return 1;
  else
    return 0;
}

int main()
{
  std::vector<std::tuple<std::string, int>> results;
  int result;

  // test1
  result = test1(8, 56, 56, 8, 0, 8, 0, "../mtx/56x56_294.mtx", 0.000001);
  results.append(std::make_tuple("test1", result));


  int correct = 0;
  for(int i = 0; i < results.size())
  {
    if(std::get<1>(results[i]))
      correct++;
    else
      printf("%s failed!\n", std::get<0>(results[i]));
  }

  printf("\n%i out of %i test successful!\n", correct, results.size());

  return 0;
}