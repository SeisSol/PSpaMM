#!/usr/bin/env python3

import sve_testsuite_generator as generator

import scripts.max_arm_sve as max_sve

blocksize_algs = [max_sve]
v_size = 8
v_size_s = 16
kernels = []

kernels.append(generator.DenseKernel("sve_mixed_test1", 9, 9, 9, 9, 9, 9, 1.0, 0.0, [(3, 3)] + [x.getBlocksize(9, 9, 1, v_size) for x in blocksize_algs], 0.0000001))
kernels.append(generator.SparseKernel("sve_mixed_test2", 9, 9, 9, 9, 0, 9, 4.0, 2.5, [(3, 3)] + [x.getBlocksize(9, 9, 1, v_size) for x in blocksize_algs], generator.generateMTX(9, 9, 20), 0.0000001))
kernels.append(generator.SparseKernel("sve_mixed_test3", 18, 18, 18, 18, 0, 18, 3.4, -2.5, [(1, 1), (3, 3), (6, 6), (9, 9)] + [x.getBlocksize(18, 18, 1, v_size) for x in blocksize_algs], generator.generateMTX(18, 18, 59), 0.0000001))
kernels.append(generator.SparseKernel("sve_mixed_test4", 80, 80, 80, 80, 0, 80, 0.0, -2.5, [(4, 4), (8, 8)] + [x.getBlocksize(80, 80, 1, v_size) for x in blocksize_algs], generator.generateMTX(80, 80, 312), 0.0000001))
kernels.append(generator.SparseKernel("sve_mixed_test5", 8, 8, 8, 10, 0, 8, 3.0, -0.9, [(2, 2), (4, 4)] + [x.getBlocksize(8, 8, 1, v_size) for x in blocksize_algs], generator.generateMTX(8, 8, 6), 0.0000001))
kernels.append(generator.DenseKernel("sve_mixed_test6", 8, 8, 8, 10, 8, 8, 3.0, -0.9, [(2, 2), (4, 4)] + [x.getBlocksize(8, 8, 1, v_size) for x in blocksize_algs], 0.0000001))

kernels.append(generator.DenseKernel("sve_test3", 4, 4, 4, 4, 4, 4, 2.0, 2.0, [(4, 4)], 0.0000001))

kernels.append(generator.SparseKernel("sve_test1", 8, 56, 56, 8, 0, 8, 1.0, 0.0, [(8, 4), (8,1)] + [x.getBlocksize(8, 56, 1, v_size) for x in blocksize_algs], generator.generateMTX(56, 56, 30), 0.0000001))
kernels.append(generator.DenseKernel("sve_test2", 8, 40, 40, 8, 40, 8, 3.0, 2.0, [(8, 5), (8,2)] + [x.getBlocksize(8, 40, 1, v_size) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("sve_test3", 8, 56, 56, 8, 56, 8, 0.0, 0.0, [(8, 3), (8, 5)] + [x.getBlocksize(8, 56, 1, v_size) for x in blocksize_algs], 0.0000001))

kernels.append(generator.SparseKernel("sve_arm_only_test1", 2, 3, 4, 2, 0, 2, 1.1233, 0.0, [(2, 1), (2,3)] + [x.getBlocksize(2, 3, 1, v_size) for x in blocksize_algs], generator.generateMTX(4, 3, 5), 0.0000001))
kernels.append(generator.SparseKernel("sve_arm_only_test2", 2, 3, 4, 20, 0, 14, 1.0, 1.0, [(2, 2), (2,3)] + [x.getBlocksize(2, 3, 1, v_size) for x in blocksize_algs], generator.generateMTX(4, 3, 5), 0.0000001))
kernels.append(generator.SparseKernel("sve_arm_only_test3", 32, 80, 50, 32, 0, 32, 1.0, 3.0, [(8, 5)] + [x.getBlocksize(32, 80, 1, v_size) for x in blocksize_algs], generator.generateMTX(50, 80, 294), 0.0000001))
kernels.append(generator.SparseKernel("sve_arm_only_test4", 32, 32, 32, 34, 0, 32, 1.0, 0.0, [(4, 4), (4,3)] + [x.getBlocksize(32, 32, 1, v_size) for x in blocksize_algs], generator.generateMTX(32, 32, 24), 0.0000001))
kernels.append(generator.SparseKernel("sve_arm_only_test5", 2, 1, 1, 2, 0, 8, 1.0, -1.0, [(2, 1)] + [x.getBlocksize(2, 1, 1, v_size) for x in blocksize_algs], generator.generateMTX(1, 1, 1), 0.0000001))
kernels.append(generator.SparseKernel("sve_arm_only_test6", 2, 2, 2, 2, 0, 2, 2.0, 234234.123, [(2, 1)] + [x.getBlocksize(2, 2, 1, v_size) for x in blocksize_algs], generator.generateMTX(2, 2, 1), 0.0000001))
kernels.append(generator.SparseKernel("sve_arm_only_test7", 16, 5, 7, 16, 0, 16, 0.0, -1.123, [(8, 1), (8,2)] + [x.getBlocksize(16, 5, 1, v_size) for x in blocksize_algs], generator.generateMTX(7, 5, 35), 0.0000001))

kernels.append(generator.DenseKernel("sve_arm_only_test8", 2, 3, 4, 2, 4, 2, 1.0, 0.0, [(2, 1), (2,3)] + [x.getBlocksize(2, 3, 1, v_size) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("sve_arm_only_test9", 2, 3, 4, 20, 12, 14, 2.0, 1.123, [(2, 2), (2,3)] + [x.getBlocksize(2, 3, 1, v_size) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("sve_arm_only_test10", 32, 80, 50, 32, 50, 32, 0.0, 0.2, [(8, 5)] + [x.getBlocksize(32, 80, 1, v_size) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("sve_arm_only_test11", 32, 32, 32, 33, 68, 32, 1231.0, 14443.0, [(4, 4), (4,3)] + [x.getBlocksize(32, 32, 1, v_size) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("sve_arm_only_test12", 2, 1, 1, 2, 1, 8, 1.0, 3.0, [(2, 1)] + [x.getBlocksize(2, 1, 1, v_size) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("sve_arm_only_test13", 2, 3, 3, 2, 3, 2, 1.0, 0.0, [(2, 1)] + [x.getBlocksize(2, 3, 1, v_size) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("sve_arm_only_test14", 16, 5, 7, 16, 7, 16, 1.0, 1.0, [(8, 1), (8,2)] + [x.getBlocksize(16, 5, 1, v_size) for x in blocksize_algs], 0.0000001))

kernels.append(generator.DenseKernel("sve_arm_only_test15", 23, 29, 31, 23, 31, 23, 1.32, 0.96, [x.getBlocksize(23, 29, 1, v_size) for x in blocksize_algs], 0.0000001))
kernels.append(generator.SparseKernel("sve_arm_only_test16", 23, 29, 31, 23, 0, 23, 1.32, 0.96, [x.getBlocksize(23, 29, 1, v_size) for x in blocksize_algs], generator.generateMTX(31, 29, 61), 0.0000001))

# TODO: test cases for single precision multiplication.
#kernels.append(generator.DenseKernelS("sve_mixed_test_S1", 9, 9, 9, 9, 9, 9, 1.0, 0.0, [x.getBlocksize(9, 9, 1, v_size_s) for x in blocksize_algs], 0.0000001))


generator.make(kernels, "arm_sve")
