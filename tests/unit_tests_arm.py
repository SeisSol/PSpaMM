#!/usr/bin/python3.6

import testsuite_generator as generator

kernels = []

kernels.append(generator.SparseKernel("test1", 8, 56, 56, 8, 0, 8, 0, [(8, 4), (8,1)], generator.generateMTX(56, 56, 30), 0.0000001))
kernels.append(generator.DenseKernel("test2", 8, 40, 40, 8, 40, 8, 1, [(8, 5), (8,2)], 0.0000001))
kernels.append(generator.DenseKernel("test3", 8, 56, 56, 8, 56, 8, 0, [(8, 3), (8, 5)], 0.0000001))

kernels.append(generator.SparseKernel("arm_only_test1", 2, 3, 4, 2, 0, 2, 0, [(2, 1), (2,3)], generator.generateMTX(4, 3, 5), 0.0000001))
kernels.append(generator.SparseKernel("arm_only_test2", 2, 3, 4, 20, 0, 14, 1, [(2, 2), (2,3)], generator.generateMTX(4, 3, 5), 0.0000001))
kernels.append(generator.SparseKernel("arm_only_test3", 32, 80, 50, 32, 0, 32, 0, [(8, 5), (6,7)], generator.generateMTX(50, 80, 294), 0.0000001))
kernels.append(generator.SparseKernel("arm_only_test4", 32, 32, 32, 34, 0, 32, 0, [(4, 4), (4,3)], generator.generateMTX(32, 32, 24), 0.0000001))
kernels.append(generator.SparseKernel("arm_only_test5", 2, 1, 1, 2, 0, 8, 1, [(2, 1)], generator.generateMTX(1, 1, 1), 0.0000001))
kernels.append(generator.SparseKernel("arm_only_test6", 2, 2, 2, 2, 0, 2, 0, [(2, 1)], generator.generateMTX(2, 2, 1), 0.0000001))
kernels.append(generator.SparseKernel("arm_only_test7", 16, 5, 7, 16, 0, 16, 1, [(8, 1), (8,2)], generator.generateMTX(7, 5, 35), 0.0000001))

kernels.append(generator.DenseKernel("arm_only_test8", 2, 3, 4, 2, 4, 2, 0, [(2, 1), (2,3)], 0.0000001))
kernels.append(generator.DenseKernel("arm_only_test9", 2, 3, 4, 20, 12, 14, 1, [(2, 2), (2,3)], 0.0000001))
kernels.append(generator.DenseKernel("arm_only_test10", 32, 80, 50, 32, 50, 32, 0, [(8, 5), (6,7)], 0.0000001))
kernels.append(generator.DenseKernel("arm_only_test11", 32, 32, 32, 33, 68, 32, 1, [(4, 4), (4,3)], 0.0000001))
kernels.append(generator.DenseKernel("arm_only_test12", 2, 1, 1, 2, 1, 8, 0, [(2, 1)], 0.0000001))
kernels.append(generator.DenseKernel("arm_only_test13", 2, 3, 3, 2, 3, 2, 0, [(2, 1)], 0.0000001))
kernels.append(generator.DenseKernel("arm_only_test14", 16, 5, 7, 16, 7, 16, 1, [(8, 1), (8,2)], 0.0000001))

generator.make(kernels, "arm")