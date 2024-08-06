#!/usr/bin/env python3

import sve_testsuite_generator as generator

import pspamm.scripts.max_arm_sve as max_sve

import sys

v_len = 4

if len(sys.argv) == 2:
    v_len = int(sys.argv[1]) // 128

blocksize_algs = [max_sve]
v_size = lambda prec: (16 // prec.size()) * v_len
v_size_d = v_size(Precision.DOUBLE)
v_size_s = v_size(Precision.SINGLE)
bitlen = v_len * 128
kernels = []

# define the maximum allowed difference between elements of our solution and the reference solution for
# double and single precision
delta_sp = 1e-4 # epsilon is around e-7 => /2 ... For most cases, 1e-6 is enough
delta_dp = 1e-7 # epsilon is around e-15 => /2

# test cases for double precision multiplication
kernels.append(generator.DenseKernel("sve_mixed_test1", Precision.DOUBLE, 9, 9, 9, 9, 9, 9, 1.0, 0.0, [(3, 3)] + [x.getBlocksize(9, 9, 1, v_size_d) for x in blocksize_algs], delta_dp))
kernels.append(generator.SparseKernel("sve_mixed_test2", Precision.DOUBLE, 9, 9, 9, 9, 0, 9, 4.0, 2.5, [(3, 3)] + [x.getBlocksize(9, 9, 1, v_size_d) for x in blocksize_algs], generator.generateMTX(9, 9, 20), delta_dp))
kernels.append(generator.SparseKernel("sve_mixed_test3", Precision.DOUBLE, 18, 18, 18, 18, 0, 18, 3.4, -2.5, [(1, 1), (3, 3), (6, 6)] + [x.getBlocksize(18, 18, 1, v_size_d) for x in blocksize_algs], generator.generateMTX(18, 18, 59), delta_dp))
kernels.append(generator.SparseKernel("sve_mixed_test4", Precision.DOUBLE, 80, 80, 80, 80, 0, 80, 0.0, -2.5, [(4, 4), (8, 8)] + [x.getBlocksize(80, 80, 1, v_size_d) for x in blocksize_algs], generator.generateMTX(80, 80, 312), delta_dp))
kernels.append(generator.SparseKernel("sve_mixed_test5", Precision.DOUBLE, 8, 8, 8, 10, 0, 8, 3.0, -0.9, [(2, 2), (4, 4)] + [x.getBlocksize(8, 8, 1, v_size_d) for x in blocksize_algs], generator.generateMTX(8, 8, 6), delta_dp))
kernels.append(generator.DenseKernel("sve_mixed_test6", Precision.DOUBLE, 8, 8, 8, 10, 8, 8, 3.0, -0.9, [(2, 2), (4, 4)] + [x.getBlocksize(8, 8, 1, v_size_d) for x in blocksize_algs], delta_dp))

kernels.append(generator.DenseKernel("sve_test3", Precision.DOUBLE, 4, 4, 4, 4, 4, 4, 2.0, 2.0, [(4, 4)], delta_dp))

kernels.append(generator.SparseKernel("sve_test1", Precision.DOUBLE, 8, 56, 56, 8, 0, 8, 1.0, 0.0, [(8, 4), (8,1)] + [x.getBlocksize(8, 56, 1, v_size_d) for x in blocksize_algs], generator.generateMTX(56, 56, 30), delta_dp))
kernels.append(generator.DenseKernel("sve_test2", Precision.DOUBLE, 8, 40, 40, 8, 40, 8, 3.0, 2.0, [(8, 5), (8,2)] + [x.getBlocksize(8, 40, 1, v_size_d) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sve_test3", Precision.DOUBLE, 8, 56, 56, 8, 56, 8, 0.0, 0.0, [(8, 3), (8, 5)] + [x.getBlocksize(8, 56, 1, v_size_d) for x in blocksize_algs], delta_dp))

kernels.append(generator.SparseKernel("sve_arm_only_test1", Precision.DOUBLE, 2, 3, 4, 2, 0, 2, 1.1233, 0.0, [(2, 1), (2,3)] + [x.getBlocksize(2, 3, 1, v_size_d) for x in blocksize_algs], generator.generateMTX(4, 3, 5), delta_dp))
kernels.append(generator.SparseKernel("sve_arm_only_test2", Precision.DOUBLE, 2, 3, 4, 20, 0, 14, 1.0, 1.0, [(2, 2), (2,3)] + [x.getBlocksize(2, 3, 1, v_size_d) for x in blocksize_algs], generator.generateMTX(4, 3, 5), delta_dp))
kernels.append(generator.SparseKernel("sve_arm_only_test3", Precision.DOUBLE, 32, 80, 50, 32, 0, 32, 1.0, 3.0, [(8, 5)] + [x.getBlocksize(32, 80, 1, v_size_d) for x in blocksize_algs], generator.generateMTX(50, 80, 294), delta_dp))
kernels.append(generator.SparseKernel("sve_arm_only_test4", Precision.DOUBLE, 32, 32, 32, 34, 0, 32, 1.0, 0.0, [(4, 4), (4,3)] + [x.getBlocksize(32, 32, 1, v_size_d) for x in blocksize_algs], generator.generateMTX(32, 32, 24), delta_dp))
kernels.append(generator.SparseKernel("sve_arm_only_test5", Precision.DOUBLE, 2, 1, 1, 2, 0, 8, 1.0, -1.0, [(2, 1)] + [x.getBlocksize(2, 1, 1, v_size_d) for x in blocksize_algs], generator.generateMTX(1, 1, 1), delta_dp))
kernels.append(generator.SparseKernel("sve_arm_only_test6", Precision.DOUBLE, 2, 2, 2, 2, 0, 2, 2.0, 234234.123, [(2, 1)] + [x.getBlocksize(2, 2, 1, v_size_d) for x in blocksize_algs], generator.generateMTX(2, 2, 1), delta_dp))
kernels.append(generator.SparseKernel("sve_arm_only_test7", Precision.DOUBLE, 16, 5, 7, 16, 0, 16, 0.0, -1.123, [(8, 1), (8,2)] + [x.getBlocksize(16, 5, 1, v_size_d) for x in blocksize_algs], generator.generateMTX(7, 5, 35), delta_dp))

kernels.append(generator.DenseKernel("sve_arm_only_test8", Precision.DOUBLE, 2, 3, 4, 2, 4, 2, 1.0, 0.0, [(2, 1), (2,3)] + [x.getBlocksize(2, 3, 1, v_size_d) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sve_arm_only_test9", Precision.DOUBLE, 2, 3, 4, 20, 12, 14, 2.0, 1.123, [(2, 2), (2,3)] + [x.getBlocksize(2, 3, 1, v_size_d) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sve_arm_only_test10", Precision.DOUBLE, 32, 80, 50, 32, 50, 32, 0.0, 0.2, [(8, 5)] + [x.getBlocksize(32, 80, 1, v_size_d) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sve_arm_only_test11", Precision.DOUBLE, 32, 32, 32, 33, 68, 32, 1231.0, 14443.0, [(4, 4), (4,3)] + [x.getBlocksize(32, 32, 1, v_size_d) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sve_arm_only_test12", Precision.DOUBLE, 2, 1, 1, 2, 1, 8, 1.0, 3.0, [(2, 1)] + [x.getBlocksize(2, 1, 1, v_size_d) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sve_arm_only_test13", Precision.DOUBLE, 2, 3, 3, 2, 3, 2, 1.0, 0.0, [(2, 1)] + [x.getBlocksize(2, 3, 1, v_size_d) for x in blocksize_algs], delta_dp))
kernels.append(generator.DenseKernel("sve_arm_only_test14", Precision.DOUBLE, 16, 5, 7, 16, 7, 16, 1.0, 1.0, [(8, 1), (8,2)] + [x.getBlocksize(16, 5, 1, v_size_d) for x in blocksize_algs], delta_dp))

kernels.append(generator.DenseKernel("sve_arm_only_test15", Precision.DOUBLE, 23, 29, 31, 23, 31, 23, 1.32, 0.96, [x.getBlocksize(23, 29, 1, v_size_d) for x in blocksize_algs], delta_dp))
kernels.append(generator.SparseKernel("sve_arm_only_test16", Precision.DOUBLE, 23, 29, 31, 23, 0, 23, 1.32, 0.96, [x.getBlocksize(23, 29, 1, v_size_d) for x in blocksize_algs], generator.generateMTX(31, 29, 61), delta_dp))

# test cases for single precision multiplication
kernels.append(generator.DenseKernel("sve_single_prec_test_S1", Precision.SINGLE, 9, 9, 9, 9, 9, 9, 1.24, 0.87, [x.getBlocksize(9, 9, 1, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernel("sve_single_prec_test_S2", Precision.SINGLE, 15, 15, 15, 15, 15, 15, -3.14, 6.28, [x.getBlocksize(15, 15, 1, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernel("sve_single_prec_test_S3", Precision.SINGLE, 23, 23, 23, 23, 23, 23, 1.5, -0.66, [x.getBlocksize(23, 23, 1, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernel("sve_single_prec_test_S4", Precision.SINGLE, 23, 31, 13, 23, 13, 23, 2.0, 0.0, [x.getBlocksize(23, 31, 1, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.SparseKernel("sve_single_prec_test_S5", Precision.SINGLE, 9, 9, 9, 9, 0, 9, 1.24, 0.87, [x.getBlocksize(9, 9, 1, v_size_s) for x in blocksize_algs], generator.generateMTX(9, 9, 8), delta_sp))
kernels.append(generator.SparseKernel("sve_single_prec_test_S6", Precision.SINGLE, 15, 15, 15, 15, 0, 15, -3.14, 6.28, [x.getBlocksize(15, 15, 1, v_size_s) for x in blocksize_algs], generator.generateMTX(15, 15, 22), delta_sp))
kernels.append(generator.SparseKernel("sve_single_prec_test_S7", Precision.SINGLE, 23, 23, 23, 23, 0, 23, 1.5, -0.66, [x.getBlocksize(23, 23, 1, v_size_s) for x in blocksize_algs], generator.generateMTX(23, 23, 52), delta_sp))
kernels.append(generator.SparseKernel("sve_single_prec_test_S8", Precision.SINGLE, 23, 31, 13, 23, 0, 23, 2.0, 0.0, [x.getBlocksize(23, 31, 1, v_size_s) for x in blocksize_algs], generator.generateMTX(13, 31, 40), delta_sp))

kernels.append(generator.DenseKernel("sve_half_prec_test_S1", Precision.HALF, 9, 9, 9, 9, 9, 9, 1.24, 0.87, [x.getBlocksize(9, 9, 1, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernel("sve_half_prec_test_S2", Precision.HALF, 15, 15, 15, 15, 15, 15, -3.14, 6.28, [x.getBlocksize(15, 15, 1, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernel("sve_half_prec_test_S3", Precision.HALF, 23, 23, 23, 23, 23, 23, 1.5, -0.66, [x.getBlocksize(23, 23, 1, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.DenseKernel("sve_half_prec_test_S4", Precision.HALF, 23, 31, 13, 23, 13, 23, 2.0, 0.0, [x.getBlocksize(23, 31, 1, v_size_s) for x in blocksize_algs], delta_sp))
kernels.append(generator.SparseKernel("sve_half_prec_test_S5", Precision.HALF, 9, 9, 9, 9, 0, 9, 1.24, 0.87, [x.getBlocksize(9, 9, 1, v_size_s) for x in blocksize_algs], generator.generateMTX(9, 9, 8), delta_sp))
kernels.append(generator.SparseKernel("sve_half_prec_test_S6", Precision.HALF, 15, 15, 15, 15, 0, 15, -3.14, 6.28, [x.getBlocksize(15, 15, 1, v_size_s) for x in blocksize_algs], generator.generateMTX(15, 15, 22), delta_sp))
kernels.append(generator.SparseKernel("sve_half_prec_test_S7", Precision.HALF, 23, 23, 23, 23, 0, 23, 1.5, -0.66, [x.getBlocksize(23, 23, 1, v_size_s) for x in blocksize_algs], generator.generateMTX(23, 23, 52), delta_sp))
kernels.append(generator.SparseKernel("sve_half_prec_test_S8", Precision.HALF, 23, 31, 13, 23, 0, 23, 2.0, 0.0, [x.getBlocksize(23, 31, 1, v_size_s) for x in blocksize_algs], generator.generateMTX(13, 31, 40), delta_sp))

generator.make(kernels, f"arm_sve{bitlen}")
