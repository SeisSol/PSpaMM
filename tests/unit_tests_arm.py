#!/usr/bin/env python3

import testsuite_generator as generator

import scripts.max_arm as max_square
import scripts.old_arm as old

blocksize_algs = [max_square, old]

kernels = []

kernels.append(generator.DenseKernel("test3", 4, 4, 4, 4, 4, 4, 2.0, 2.0, [(4, 4)], 0.0000001))

kernels.append(generator.SparseKernel("test1", 8, 56, 56, 8, 0, 8, 1.0, 0.0, [(8, 4), (8,1)] + [x.getBlocksize(8, 56, 1) for x in blocksize_algs], generator.generateMTX(56, 56, 30), 0.0000001))
kernels.append(generator.DenseKernel("test2", 8, 40, 40, 8, 40, 8, 3.0, 2.0, [(8, 5), (8,2)] + [x.getBlocksize(8, 40, 1) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("test3", 8, 56, 56, 8, 56, 8, 0.0, 0.0, [(8, 3), (8, 5)] + [x.getBlocksize(8, 56, 1) for x in blocksize_algs], 0.0000001))

kernels.append(generator.SparseKernel("arm_only_test1", 2, 3, 4, 2, 0, 2, 1.1233, 0.0, [(2, 1), (2,3)] + [x.getBlocksize(2, 3, 1) for x in blocksize_algs], generator.generateMTX(4, 3, 5), 0.0000001))
kernels.append(generator.SparseKernel("arm_only_test2", 2, 3, 4, 20, 0, 14, 1.0, 1.0, [(2, 2), (2,3)] + [x.getBlocksize(2, 3, 1) for x in blocksize_algs], generator.generateMTX(4, 3, 5), 0.0000001))
kernels.append(generator.SparseKernel("arm_only_test3", 32, 80, 50, 32, 0, 32, 1.0, 3.0, [(8, 5)] + [x.getBlocksize(32, 80, 1) for x in blocksize_algs], generator.generateMTX(50, 80, 294), 0.0000001))
kernels.append(generator.SparseKernel("arm_only_test4", 32, 32, 32, 34, 0, 32, 1.0, 0.0, [(4, 4), (4,3)] + [x.getBlocksize(32, 32, 1) for x in blocksize_algs], generator.generateMTX(32, 32, 24), 0.0000001))
kernels.append(generator.SparseKernel("arm_only_test5", 2, 1, 1, 2, 0, 8, 1.0, -1.0, [(2, 1)] + [x.getBlocksize(2, 1, 1) for x in blocksize_algs], generator.generateMTX(1, 1, 1), 0.0000001))
kernels.append(generator.SparseKernel("arm_only_test6", 2, 2, 2, 2, 0, 2, 2.0, 234234.123, [(2, 1)] + [x.getBlocksize(2, 2, 1) for x in blocksize_algs], generator.generateMTX(2, 2, 1), 0.0000001))
kernels.append(generator.SparseKernel("arm_only_test7", 16, 5, 7, 16, 0, 16, 0.0, -1.123, [(8, 1), (8,2)] + [x.getBlocksize(16, 5, 1) for x in blocksize_algs], generator.generateMTX(7, 5, 35), 0.0000001))

kernels.append(generator.DenseKernel("arm_only_test8", 2, 3, 4, 2, 4, 2, 1.0, 0.0, [(2, 1), (2,3)] + [x.getBlocksize(2, 3, 1) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("arm_only_test9", 2, 3, 4, 20, 12, 14, 2.0, 1.123, [(2, 2), (2,3)] + [x.getBlocksize(2, 3, 1) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("arm_only_test10", 32, 80, 50, 32, 50, 32, 0.0, 0.2, [(8, 5)] + [x.getBlocksize(32, 80, 1) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("arm_only_test11", 32, 32, 32, 33, 68, 32, 1231.0, 14443.0, [(4, 4), (4,3)] + [x.getBlocksize(32, 32, 1) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("arm_only_test12", 2, 1, 1, 2, 1, 8, 1.0, 3.0, [(2, 1)] + [x.getBlocksize(2, 1, 1) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("arm_only_test13", 2, 3, 3, 2, 3, 2, 1.0, 0.0, [(2, 1)] + [x.getBlocksize(2, 3, 1) for x in blocksize_algs], 0.0000001))
kernels.append(generator.DenseKernel("arm_only_test14", 16, 5, 7, 16, 7, 16, 1.0, 1.0, [(8, 1), (8,2)] + [x.getBlocksize(16, 5, 1) for x in blocksize_algs], 0.0000001))


generator.make(kernels, "arm")
