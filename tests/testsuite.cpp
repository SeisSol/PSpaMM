#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <tuple>


long long sparsemmgen_num_total_flops = 0;
#include "knl/test1_8_1.h"
#include "knl/test1_8_28.h"
#include "knl/test1_8_31.h"
#include "knl/test1_8_4.h"
#include "knl/test2_8_5.h"
#include "knl/test2_8_2.h"
#include "knl/test2_8_31.h"
#include "knl/test2_8_20.h"
#include "knl/test3_8_3.h"
#include "knl/test3_8_5.h"
#include "knl/test3_8_31.h"
#include "knl/test3_8_28.h"
#include "knl/knl_only_test1_8_1.h"
#include "knl/knl_only_test1_8_2.h"
#include "knl/knl_only_test2_8_31.h"
#include "knl/knl_only_test2_8_2.h"
#include "knl/knl_only_test2_16_7.h"
#include "knl/knl_only_test2_8_20.h"
#include "knl/knl_only_test3_8_1.h"
#include "knl/knl_only_test3_8_2.h"
#include "knl/knl_only_test4_8_3.h"
#include "knl/knl_only_test4_8_20.h"
#include "knl/knl_only_test5_8_14.h"
#include "knl/knl_only_test5_8_5.h"
#include "knl/knl_only_test5_32_5.h"
#include "knl/knl_only_test5_40_5.h"
#include "knl/knl_only_test5_32_2.h"
#include "knl/knl_only_test6_8_1.h"
#include "knl/knl_only_test7_8_1.h"
#include "knl/knl_only_test7_8_24.h"
#include "knl/knl_only_test8_8_1.h"
#include "knl/knl_only_test8_8_2.h"
#include "knl/knl_only_test9_8_31.h"
#include "knl/knl_only_test9_8_2.h"
#include "knl/knl_only_test9_16_7.h"
#include "knl/knl_only_test9_8_20.h"
#include "knl/knl_only_test10_8_28.h"
#include "knl/knl_only_test11_8_3.h"
#include "knl/knl_only_test11_8_20.h"
#include "knl/knl_only_test12_8_14.h"
#include "knl/knl_only_test12_8_5.h"
#include "knl/knl_only_test12_32_5.h"
#include "knl/knl_only_test12_40_5.h"
#include "knl/knl_only_test12_32_2.h"
#include "knl/knl_only_test13_8_1.h"
#include "knl/knl_only_test14_8_1.h"
#include "knl/knl_only_test14_8_24.h"


void gemm_ref(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC, double ALPHA, double BETA, double* A, double* B, double* C) {
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

int post(unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC, double ALPHA, double BETA, double* A, double* B, double* C, double* Cref, double DELTA) {

  if(LDB == 0)
    LDB = K;

  gemm_ref(M, N, K, LDA, LDB, LDC, ALPHA, BETA, A, B, Cref);
    
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


  pointers = pre(8, 56, 56, 8, 0, 8, "mtx/56x56_30.mtx");
  test1_8_1(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), -1.1, 0.0, nullptr);
  result = post(8, 56, 56, 8, 0, 8, -1.1, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test1_8_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 56, 56, 8, 0, 8, "mtx/56x56_30.mtx");
  test1_8_28(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), -1.1, 0.0, nullptr);
  result = post(8, 56, 56, 8, 0, 8, -1.1, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test1_8_28", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 56, 56, 8, 0, 8, "mtx/56x56_30.mtx");
  test1_8_31(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), -1.1, 0.0, nullptr);
  result = post(8, 56, 56, 8, 0, 8, -1.1, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test1_8_31", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 56, 56, 8, 0, 8, "mtx/56x56_30.mtx");
  test1_8_4(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), -1.1, 0.0, nullptr);
  result = post(8, 56, 56, 8, 0, 8, -1.1, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test1_8_4", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 40, 40, 8, 40, 8, "");
  test2_8_5(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 2.5, 1.0, nullptr);
  result = post(8, 40, 40, 8, 40, 8, 2.5, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test2_8_5", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 40, 40, 8, 40, 8, "");
  test2_8_2(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 2.5, 1.0, nullptr);
  result = post(8, 40, 40, 8, 40, 8, 2.5, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test2_8_2", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 40, 40, 8, 40, 8, "");
  test2_8_31(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 2.5, 1.0, nullptr);
  result = post(8, 40, 40, 8, 40, 8, 2.5, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test2_8_31", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 40, 40, 8, 40, 8, "");
  test2_8_20(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 2.5, 1.0, nullptr);
  result = post(8, 40, 40, 8, 40, 8, 2.5, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test2_8_20", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 56, 56, 8, 56, 8, "");
  test3_8_3(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 1.0, 0.0, nullptr);
  result = post(8, 56, 56, 8, 56, 8, 1.0, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test3_8_3", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 56, 56, 8, 56, 8, "");
  test3_8_5(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 1.0, 0.0, nullptr);
  result = post(8, 56, 56, 8, 56, 8, 1.0, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test3_8_5", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 56, 56, 8, 56, 8, "");
  test3_8_31(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 1.0, 0.0, nullptr);
  result = post(8, 56, 56, 8, 56, 8, 1.0, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test3_8_31", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 56, 56, 8, 56, 8, "");
  test3_8_28(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 1.0, 0.0, nullptr);
  result = post(8, 56, 56, 8, 56, 8, 1.0, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test3_8_28", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 2, 1, 8, 0, 8, "mtx/1x2_1.mtx");
  knl_only_test1_8_1(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), 1.0, 0.0, nullptr);
  result = post(8, 2, 1, 8, 0, 8, 1.0, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test1_8_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 2, 1, 8, 0, 8, "mtx/1x2_1.mtx");
  knl_only_test1_8_2(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), 1.0, 0.0, nullptr);
  result = post(8, 2, 1, 8, 0, 8, 1.0, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test1_8_2", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(24, 40, 40, 32, 0, 24, "mtx/40x40_20.mtx");
  knl_only_test2_8_31(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), 1000, 1.0, nullptr);
  result = post(24, 40, 40, 32, 0, 24, 1000, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test2_8_31", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(24, 40, 40, 32, 0, 24, "mtx/40x40_20.mtx");
  knl_only_test2_8_2(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), 1000, 1.0, nullptr);
  result = post(24, 40, 40, 32, 0, 24, 1000, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test2_8_2", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(24, 40, 40, 32, 0, 24, "mtx/40x40_20.mtx");
  knl_only_test2_16_7(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), 1000, 1.0, nullptr);
  result = post(24, 40, 40, 32, 0, 24, 1000, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test2_16_7", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(24, 40, 40, 32, 0, 24, "mtx/40x40_20.mtx");
  knl_only_test2_8_20(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), 1000, 1.0, nullptr);
  result = post(24, 40, 40, 32, 0, 24, 1000, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test2_8_20", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 2, 1, 8, 0, 16, "mtx/1x2_2.mtx");
  knl_only_test3_8_1(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), -2.0, 0.0, nullptr);
  result = post(8, 2, 1, 8, 0, 16, -2.0, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test3_8_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 2, 1, 8, 0, 16, "mtx/1x2_2.mtx");
  knl_only_test3_8_2(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), -2.0, 0.0, nullptr);
  result = post(8, 2, 1, 8, 0, 16, -2.0, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test3_8_2", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 20, 10, 40, 0, 8, "mtx/10x20_1.mtx");
  knl_only_test4_8_3(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), 35.222222222222, 0.0, nullptr);
  result = post(8, 20, 10, 40, 0, 8, 35.222222222222, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test4_8_3", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 20, 10, 40, 0, 8, "mtx/10x20_1.mtx");
  knl_only_test4_8_20(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), 35.222222222222, 0.0, nullptr);
  result = post(8, 20, 10, 40, 0, 8, 35.222222222222, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test4_8_20", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(64, 5, 10, 64, 0, 64, "mtx/10x5_1.mtx");
  knl_only_test5_8_14(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), 2.3, 0.0, nullptr);
  result = post(64, 5, 10, 64, 0, 64, 2.3, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test5_8_14", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(64, 5, 10, 64, 0, 64, "mtx/10x5_1.mtx");
  knl_only_test5_8_5(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), 2.3, 0.0, nullptr);
  result = post(64, 5, 10, 64, 0, 64, 2.3, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test5_8_5", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(64, 5, 10, 64, 0, 64, "mtx/10x5_1.mtx");
  knl_only_test5_32_5(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), 2.3, 0.0, nullptr);
  result = post(64, 5, 10, 64, 0, 64, 2.3, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test5_32_5", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(64, 5, 10, 64, 0, 64, "mtx/10x5_1.mtx");
  knl_only_test5_40_5(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), 2.3, 0.0, nullptr);
  result = post(64, 5, 10, 64, 0, 64, 2.3, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test5_40_5", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(64, 5, 10, 64, 0, 64, "mtx/10x5_1.mtx");
  knl_only_test5_32_2(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), 2.3, 0.0, nullptr);
  result = post(64, 5, 10, 64, 0, 64, 2.3, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test5_32_2", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 1, 1, 16, 0, 56, "mtx/1x1_1.mtx");
  knl_only_test6_8_1(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), 1.0, 0.0, nullptr);
  result = post(8, 1, 1, 16, 0, 56, 1.0, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test6_8_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 24, 40, 8, 0, 8, "mtx/40x24_1.mtx");
  knl_only_test7_8_1(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), 1.0, 0.0, nullptr);
  result = post(8, 24, 40, 8, 0, 8, 1.0, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test7_8_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 24, 40, 8, 0, 8, "mtx/40x24_1.mtx");
  knl_only_test7_8_24(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers), 1.0, 0.0, nullptr);
  result = post(8, 24, 40, 8, 0, 8, 1.0, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test7_8_24", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 2, 1, 8, 1, 8, "");
  knl_only_test8_8_1(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 2.5, 0.0, nullptr);
  result = post(8, 2, 1, 8, 1, 8, 2.5, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test8_8_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 2, 1, 8, 1, 8, "");
  knl_only_test8_8_2(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 2.5, 0.0, nullptr);
  result = post(8, 2, 1, 8, 1, 8, 2.5, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test8_8_2", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(24, 40, 40, 32, 60, 32, "");
  knl_only_test9_8_31(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 0.1, 1.0, nullptr);
  result = post(24, 40, 40, 32, 60, 32, 0.1, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test9_8_31", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(24, 40, 40, 32, 60, 32, "");
  knl_only_test9_8_2(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 0.1, 1.0, nullptr);
  result = post(24, 40, 40, 32, 60, 32, 0.1, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test9_8_2", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(24, 40, 40, 32, 60, 32, "");
  knl_only_test9_16_7(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 0.1, 1.0, nullptr);
  result = post(24, 40, 40, 32, 60, 32, 0.1, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test9_16_7", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(24, 40, 40, 32, 60, 32, "");
  knl_only_test9_8_20(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 0.1, 1.0, nullptr);
  result = post(24, 40, 40, 32, 60, 32, 0.1, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test9_8_20", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(56, 28, 56, 56, 56, 56, "");
  knl_only_test10_8_28(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 1.0, 0.0, nullptr);
  result = post(56, 28, 56, 56, 56, 56, 1.0, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test10_8_28", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 20, 10, 40, 10, 8, "");
  knl_only_test11_8_3(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 234234.123123, 0.0, nullptr);
  result = post(8, 20, 10, 40, 10, 8, 234234.123123, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test11_8_3", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 20, 10, 40, 10, 8, "");
  knl_only_test11_8_20(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 234234.123123, 0.0, nullptr);
  result = post(8, 20, 10, 40, 10, 8, 234234.123123, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test11_8_20", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(64, 5, 10, 64, 12, 64, "");
  knl_only_test12_8_14(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 1.0, 1.0, nullptr);
  result = post(64, 5, 10, 64, 12, 64, 1.0, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test12_8_14", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(64, 5, 10, 64, 12, 64, "");
  knl_only_test12_8_5(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 1.0, 1.0, nullptr);
  result = post(64, 5, 10, 64, 12, 64, 1.0, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test12_8_5", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(64, 5, 10, 64, 12, 64, "");
  knl_only_test12_32_5(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 1.0, 1.0, nullptr);
  result = post(64, 5, 10, 64, 12, 64, 1.0, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test12_32_5", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(64, 5, 10, 64, 12, 64, "");
  knl_only_test12_40_5(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 1.0, 1.0, nullptr);
  result = post(64, 5, 10, 64, 12, 64, 1.0, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test12_40_5", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(64, 5, 10, 64, 12, 64, "");
  knl_only_test12_32_2(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 1.0, 1.0, nullptr);
  result = post(64, 5, 10, 64, 12, 64, 1.0, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test12_32_2", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 1, 1, 16, 1, 56, "");
  knl_only_test13_8_1(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 0.0, 0.0, nullptr);
  result = post(8, 1, 1, 16, 1, 56, 0.0, 0.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test13_8_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 24, 40, 8, 41, 8, "");
  knl_only_test14_8_1(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 2.0, 1.0, nullptr);
  result = post(8, 24, 40, 8, 41, 8, 2.0, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test14_8_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 24, 40, 8, 41, 8, "");
  knl_only_test14_8_24(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), 2.0, 1.0, nullptr);
  result = post(8, 24, 40, 8, 41, 8, 2.0, 1.0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("knl_only_test14_8_24", result));
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