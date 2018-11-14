#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <tuple>


long long sparsemmgen_num_total_flops = 0;
#include "knl/knl_only_test9_8_31.h"
#include "knl/knl_only_test9_8_2.h"
#include "knl/knl_only_test9_16_7.h"
#include "knl/knl_only_test9_8_20.h"


void gemm_ref(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC, double BETA, double* A, double* B, double* C) {
  if (BETA == 0.0) {
    memset(C, 0, LDC * N * sizeof(double));
  }
  for (unsigned col = 0; col < N; ++col) {
    for (unsigned row = 0; row < M; ++row) {
      for (unsigned s = 0; s < K; ++s) {
        C[row + col * LDC] += A[row + s * LDA] * B[s + col * LDB];
      }
    }
  }
}

std::tuple<double*, double*, double*, double*, double*> pre(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC, std::string MTX) {

  if(LDB == 0)
    LDB = K;

  double *A;
  double *B;
  double *Bsparse;
  double *Cref;
  double *C;

  int resA = posix_memalign(reinterpret_cast<void **>(&A), 64, LDA*LDB*sizeof(double));
  int resB = posix_memalign(reinterpret_cast<void **>(&B), 64, LDB*N*sizeof(double));
  int resBsparse = posix_memalign(reinterpret_cast<void **>(&Bsparse), 64, LDB*N*sizeof(double));  
  int resCref = posix_memalign(reinterpret_cast<void **>(&Cref), 64, LDC*N*sizeof(double));
  int resC = posix_memalign(reinterpret_cast<void **>(&C), 64, LDC*N*sizeof(double));

  std::string line;
  std::ifstream f(MTX);
  getline(f, line);
  getline(f, line);
  getline(f, line);

  for(int i = 0; i < LDA*LDB; i++)
    A[i] = (double)rand() / RAND_MAX;

  for(int i = 0; i < LDB*N; i++)
    if(MTX.compare(""))
      B[i] = 0;
    else 
      B[i] = (double)rand() / RAND_MAX;

  for(int i = 0; i < LDC*N; i++)
  {
    Cref[i] = (double)rand() / RAND_MAX;
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

int post(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC, double BETA, double* A, double* B, double* C, double* Cref, double DELTA) {

  if(LDB == 0)
    LDB = K;

  gemm_ref(M, N, K, LDA, LDB, LDC, BETA, A, B, Cref);
    
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++)
      if(std::abs(C[i + j * LDC] - Cref[i + j * LDC]) > DELTA)
        return 0;

  return 1;
}

int main()
{
  std::vector<std::tuple<std::string, int>> results;
  std::tuple<double*, double*, double*, double*, double*> pointers;
  int result;


  pointers = pre(24, 40, 40, 32, 60, 32, "");
  knl_only_test9_8_31(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), nullptr, nullptr, nullptr);
  result = post(24, 40, 40, 32, 60, 32, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test9_8_31", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(24, 40, 40, 32, 60, 32, "");
  knl_only_test9_8_2(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), nullptr, nullptr, nullptr);
  result = post(24, 40, 40, 32, 60, 32, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test9_8_2", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(24, 40, 40, 32, 60, 32, "");
  knl_only_test9_16_7(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), nullptr, nullptr, nullptr);
  result = post(24, 40, 40, 32, 60, 32, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test9_16_7", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(24, 40, 40, 32, 60, 32, "");
  knl_only_test9_8_20(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), nullptr, nullptr, nullptr);
  result = post(24, 40, 40, 32, 60, 32, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test9_8_20", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));


  int correct = 0;
  for(int i = 0; i < results.size(); i++)
  {
    if(std::get<1>(results[i]))
      correct++;
    else
      printf("%s failed!\n", (std::get<0>(results[i])).c_str());
  }

  printf("\n%i out of %lu test successful!\n", correct, results.size());

  return 0;
}