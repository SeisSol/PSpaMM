#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <stdio.h>
#include <tuple>

#include "arm/test1_8_1.h"
#include "arm/test1_4_7.h"
#include "arm/test1_6_7.h"
#include "arm/test1_8_4.h"
#include "arm/test2_8_5.h"
#include "arm/test2_8_2.h"
#include "arm/test2_6_7.h"
#include "arm/test3_8_3.h"
#include "arm/test3_8_5.h"
#include "arm/test3_6_7.h"
#include "arm/test3_4_7.h"
#include "arm/arm_only_test1_2_3.h"
#include "arm/arm_only_test1_2_1.h"
#include "arm/arm_only_test2_2_3.h"
#include "arm/arm_only_test2_2_2.h"
#include "arm/arm_only_test3_8_5.h"
#include "arm/arm_only_test3_6_7.h"
#include "arm/arm_only_test4_4_4.h"
#include "arm/arm_only_test4_6_7.h"
#include "arm/arm_only_test4_4_3.h"
#include "arm/arm_only_test4_8_4.h"
#include "arm/arm_only_test5_2_1.h"
#include "arm/arm_only_test6_2_1.h"
#include "arm/arm_only_test6_2_2.h"
#include "arm/arm_only_test7_8_1.h"
#include "arm/arm_only_test7_14_3.h"
#include "arm/arm_only_test7_8_5.h"
#include "arm/arm_only_test7_8_2.h"
#include "arm/arm_only_test8_2_3.h"
#include "arm/arm_only_test8_2_1.h"
#include "arm/arm_only_test9_2_3.h"
#include "arm/arm_only_test9_2_2.h"
#include "arm/arm_only_test10_8_5.h"
#include "arm/arm_only_test10_6_7.h"
#include "arm/arm_only_test11_4_4.h"
#include "arm/arm_only_test11_6_7.h"
#include "arm/arm_only_test11_4_3.h"
#include "arm/arm_only_test11_8_4.h"
#include "arm/arm_only_test12_2_1.h"
#include "arm/arm_only_test13_2_3.h"
#include "arm/arm_only_test13_2_1.h"
#include "arm/arm_only_test14_8_1.h"
#include "arm/arm_only_test14_14_3.h"
#include "arm/arm_only_test14_8_5.h"
#include "arm/arm_only_test14_8_2.h"


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


  pointers = pre(8, 56, 56, 8, 0, 8, "mtx/56x56_30.mtx");
  test1_8_1(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(8, 56, 56, 8, 0, 8, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test1_8_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 56, 56, 8, 0, 8, "mtx/56x56_30.mtx");
  test1_4_7(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(8, 56, 56, 8, 0, 8, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test1_4_7", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 56, 56, 8, 0, 8, "mtx/56x56_30.mtx");
  test1_6_7(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(8, 56, 56, 8, 0, 8, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test1_6_7", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 56, 56, 8, 0, 8, "mtx/56x56_30.mtx");
  test1_8_4(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(8, 56, 56, 8, 0, 8, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test1_8_4", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 40, 40, 8, 40, 8, "");
  test2_8_5(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(8, 40, 40, 8, 40, 8, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test2_8_5", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 40, 40, 8, 40, 8, "");
  test2_8_2(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(8, 40, 40, 8, 40, 8, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test2_8_2", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 40, 40, 8, 40, 8, "");
  test2_6_7(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(8, 40, 40, 8, 40, 8, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test2_6_7", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 56, 56, 8, 56, 8, "");
  test3_8_3(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(8, 56, 56, 8, 56, 8, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test3_8_3", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 56, 56, 8, 56, 8, "");
  test3_8_5(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(8, 56, 56, 8, 56, 8, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test3_8_5", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 56, 56, 8, 56, 8, "");
  test3_6_7(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(8, 56, 56, 8, 56, 8, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test3_6_7", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(8, 56, 56, 8, 56, 8, "");
  test3_4_7(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(8, 56, 56, 8, 56, 8, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("test3_4_7", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(2, 3, 4, 2, 0, 2, "mtx/4x3_5.mtx");
  arm_only_test1_2_3(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(2, 3, 4, 2, 0, 2, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test1_2_3", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(2, 3, 4, 2, 0, 2, "mtx/4x3_5.mtx");
  arm_only_test1_2_1(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(2, 3, 4, 2, 0, 2, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test1_2_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(2, 3, 4, 20, 0, 14, "mtx/4x3_5.mtx");
  arm_only_test2_2_3(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(2, 3, 4, 20, 0, 14, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test2_2_3", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(2, 3, 4, 20, 0, 14, "mtx/4x3_5.mtx");
  arm_only_test2_2_2(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(2, 3, 4, 20, 0, 14, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test2_2_2", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(32, 80, 50, 32, 0, 32, "mtx/50x80_294.mtx");
  arm_only_test3_8_5(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(32, 80, 50, 32, 0, 32, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test3_8_5", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(32, 80, 50, 32, 0, 32, "mtx/50x80_294.mtx");
  arm_only_test3_6_7(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(32, 80, 50, 32, 0, 32, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test3_6_7", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(32, 32, 32, 34, 0, 32, "mtx/32x32_24.mtx");
  arm_only_test4_4_4(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(32, 32, 32, 34, 0, 32, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test4_4_4", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(32, 32, 32, 34, 0, 32, "mtx/32x32_24.mtx");
  arm_only_test4_6_7(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(32, 32, 32, 34, 0, 32, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test4_6_7", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(32, 32, 32, 34, 0, 32, "mtx/32x32_24.mtx");
  arm_only_test4_4_3(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(32, 32, 32, 34, 0, 32, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test4_4_3", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(32, 32, 32, 34, 0, 32, "mtx/32x32_24.mtx");
  arm_only_test4_8_4(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(32, 32, 32, 34, 0, 32, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test4_8_4", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(2, 1, 1, 2, 0, 8, "mtx/1x1_1.mtx");
  arm_only_test5_2_1(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(2, 1, 1, 2, 0, 8, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test5_2_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(2, 2, 2, 2, 0, 2, "mtx/2x2_1.mtx");
  arm_only_test6_2_1(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(2, 2, 2, 2, 0, 2, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test6_2_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(2, 2, 2, 2, 0, 2, "mtx/2x2_1.mtx");
  arm_only_test6_2_2(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(2, 2, 2, 2, 0, 2, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test6_2_2", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(16, 5, 7, 16, 0, 16, "mtx/7x5_35.mtx");
  arm_only_test7_8_1(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(16, 5, 7, 16, 0, 16, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test7_8_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(16, 5, 7, 16, 0, 16, "mtx/7x5_35.mtx");
  arm_only_test7_14_3(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(16, 5, 7, 16, 0, 16, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test7_14_3", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(16, 5, 7, 16, 0, 16, "mtx/7x5_35.mtx");
  arm_only_test7_8_5(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(16, 5, 7, 16, 0, 16, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test7_8_5", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(16, 5, 7, 16, 0, 16, "mtx/7x5_35.mtx");
  arm_only_test7_8_2(std::get<0>(pointers), std::get<2>(pointers), std::get<3>(pointers));
  result = post(16, 5, 7, 16, 0, 16, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test7_8_2", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(2, 3, 4, 2, 4, 2, "");
  arm_only_test8_2_3(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(2, 3, 4, 2, 4, 2, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test8_2_3", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(2, 3, 4, 2, 4, 2, "");
  arm_only_test8_2_1(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(2, 3, 4, 2, 4, 2, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test8_2_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(2, 3, 4, 20, 12, 14, "");
  arm_only_test9_2_3(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(2, 3, 4, 20, 12, 14, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test9_2_3", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(2, 3, 4, 20, 12, 14, "");
  arm_only_test9_2_2(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(2, 3, 4, 20, 12, 14, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test9_2_2", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(32, 80, 50, 32, 50, 32, "");
  arm_only_test10_8_5(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(32, 80, 50, 32, 50, 32, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test10_8_5", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(32, 80, 50, 32, 50, 32, "");
  arm_only_test10_6_7(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(32, 80, 50, 32, 50, 32, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test10_6_7", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(32, 32, 32, 33, 68, 32, "");
  arm_only_test11_4_4(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(32, 32, 32, 33, 68, 32, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test11_4_4", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(32, 32, 32, 33, 68, 32, "");
  arm_only_test11_6_7(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(32, 32, 32, 33, 68, 32, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test11_6_7", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(32, 32, 32, 33, 68, 32, "");
  arm_only_test11_4_3(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(32, 32, 32, 33, 68, 32, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test11_4_3", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(32, 32, 32, 33, 68, 32, "");
  arm_only_test11_8_4(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(32, 32, 32, 33, 68, 32, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test11_8_4", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(2, 1, 1, 2, 1, 8, "");
  arm_only_test12_2_1(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(2, 1, 1, 2, 1, 8, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test12_2_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(2, 3, 3, 2, 3, 2, "");
  arm_only_test13_2_3(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(2, 3, 3, 2, 3, 2, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test13_2_3", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(2, 3, 3, 2, 3, 2, "");
  arm_only_test13_2_1(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(2, 3, 3, 2, 3, 2, 0, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test13_2_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(16, 5, 7, 16, 7, 16, "");
  arm_only_test14_8_1(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(16, 5, 7, 16, 7, 16, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test14_8_1", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(16, 5, 7, 16, 7, 16, "");
  arm_only_test14_14_3(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(16, 5, 7, 16, 7, 16, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test14_14_3", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(16, 5, 7, 16, 7, 16, "");
  arm_only_test14_8_5(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(16, 5, 7, 16, 7, 16, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test14_8_5", result));
  free(std::get<0>(pointers)); free(std::get<1>(pointers)); free(std::get<2>(pointers)); free(std::get<3>(pointers)); free(std::get<4>(pointers));

  pointers = pre(16, 5, 7, 16, 7, 16, "");
  arm_only_test14_8_2(std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers));
  result = post(16, 5, 7, 16, 7, 16, 1, std::get<0>(pointers), std::get<1>(pointers), std::get<3>(pointers), std::get<4>(pointers), 0.0000001);
  results.push_back(std::make_tuple("arm_only_test14_8_2", result));
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