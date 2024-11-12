#!/usr/bin/env python3

import testsuite_generator as generator
from importlib import import_module

from pspamm.codegen.precision import *

import sys
import re
import random

arch = sys.argv[1]

parsedarch = re.fullmatch(r'(?P<name>[a-zA-Z_]+)(?P<prec>\d+)', arch)

archname = parsedarch.group('name')
archprec = parsedarch.group('prec')

blocksize = import_module("pspamm.codegen.architectures." + archname + ".blocksize")

scripts = {
    "arm": lambda blocksize: [blocksize.Old, blocksize.Max, blocksize.MaxK, blocksize.Cube],
    "arm_sve": lambda blocksize: [blocksize.Max, blocksize.MaxK, blocksize.Cube],
    "knl": lambda blocksize: [blocksize.Old, blocksize.Max, blocksize.MaxBn, blocksize.CubeBn],
    "hsw": lambda blocksize: [blocksize.Old, blocksize.Max, blocksize.Cube],
}

blocksize_algs = scripts[archname](blocksize) + [blocksize.Default]

bitlen = int(archprec)
v_len = bitlen // 128
v_size_fun = lambda prec: (16 // prec.size()) * v_len

# define the maximum allowed difference between elements of our solution and the reference solution for
# double and single precision
delta_hp = 1e-2
delta_sp = 1e-4 # epsilon is around e-7 => /2 ... For most cases, 1e-6 is enough
delta_dp = 1e-6 # epsilon is around e-15 => /2

kernels = []

random.seed(1)

for precision, delta in zip((Precision.SINGLE, Precision.DOUBLE), (delta_sp, delta_dp)):
    v_size = v_size_fun(precision)
    kernels.append(generator.TestKernel("testlarge", precision, 40, 100, 100, 100, 100, 100, 2.5, 1.0, [(8, 5), (8,2)] + [x.getBlocksize(10, 10, 1, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("test1", precision, 8, 56, 56, 8, 0, 8, 2.0, 0.0, [(8, 4), (8,1)] + [x.getBlocksize(8, 56, 2, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(56, 56, 30), delta))
    kernels.append(generator.TestKernel("test2", precision, 8, 40, 40, 8, 40, 8, 2.5, 1.0, [(8, 5), (8,2)] + [x.getBlocksize(8, 40, 2, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("test3", precision, 8, 56, 56, 8, 56, 8, 1.0, 5.0, [(8, 3), (8, 5)] + [x.getBlocksize(8, 56, 2, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("knl_only_test1", precision, 8, 2, 1, 8, 0, 8, 1.0, 0.0, [(8,1,2)] + [x.getBlocksize(8, 2, 2, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(1, 2, 1), delta))
    kernels.append(generator.TestKernel("knl_only_test2", precision, 24, 40, 40, 32, 0, 24, 1000, 1.0, [(8, 2,2), (16,7,2)] + [x.getBlocksize(24, 40, 2, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(40, 40, 20), delta))

    kernels.append(generator.TestKernel("knl_only_test3", precision, 8, 2, 1, 8, 0, 16, -2.0, 0.0, [(8, 1,2)] + [x.getBlocksize(8, 2, 2, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(1, 2, 2), delta))
    kernels.append(generator.TestKernel("knl_only_test4", precision, 24, 20, 10, 40, 0, 24, 35.222, 0.0, [(8, 20,2), (24,3,2)] + [x.getBlocksize(8, 20, 2, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(10, 20, 1), delta))
    kernels.append(generator.TestKernel("knl_only_test5", precision, 64, 5, 10, 64, 0, 64, 2.3, 0.0, [(32, 2,2), (8,14,2)] + [x.getBlocksize(64, 5, 2, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(10, 5, 1), delta))
    kernels.append(generator.TestKernel("knl_only_test6", precision, 8, 1, 1, 16, 0, 56, 1.0, 0.0, [(8, 1,2)] + [x.getBlocksize(8, 1, 2, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(1, 1, 1), delta))
    kernels.append(generator.TestKernel("knl_only_test7", precision, 8, 24, 40, 8, 0, 8, 1.0, 333333.2222222, [(8, 24,2), (8,1,2)] + [x.getBlocksize(8, 24, 2, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(40, 24, 1), delta))

    kernels.append(generator.TestKernel("knl_only_test8", precision, 8, 2, 1, 8, 1, 8, 2.5, 0.0, [(8,1,2)] + [x.getBlocksize(8, 2, 2, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("knl_only_test9", precision, 32, 40, 40, 32, 60, 32, 2.0, -4.33, [(8,2,2), (16,7,2)] + [x.getBlocksize(32, 40, 2, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("knl_only_test10", precision, 56, 28, 56, 56, 56, 56, 0.1, 3.0, [(8, 28,2)], None, None, delta))
    kernels.append(generator.TestKernel("knl_only_test11", precision, 8, 20, 8, 40, 10, 8, 234234.123123, 0.0, [(8, 20,2), (8,3,2)] + [x.getBlocksize(8, 20, 2, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("knl_only_test12", precision, 64, 5, 10, 64, 12, 64, 1.0, 1.0, [(32, 2,2), (8,14,2)] + [x.getBlocksize(64, 5, 2, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("knl_only_test13", precision, 8, 1, 1, 16, 1, 56, 0.0, 123.0, [(8, 1,2)] + [x.getBlocksize(8, 1, 2, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("knl_only_test14", precision, 8, 24, 40, 8, 41, 8, 2.0, 1.0, [(8, 24,2)] + [x.getBlocksize(8, 24, 2, v_size, precision) for x in blocksize_algs], None, None, delta))

    kernels.append(generator.TestKernel("hswtest1", precision, 8, 56, 56, 8, 0, 8, 2.0, 0.0, [(8, 4), (8,1)] + [x.getBlocksize(8, 56, 2, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(56, 56, 30), delta))
    kernels.append(generator.TestKernel("hswtest2", precision, 8, 40, 40, 8, 40, 8, 2.5, 1.0, [(8,2)] + [x.getBlocksize(8, 40, 2, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("hswtest3", precision, 8, 56, 56, 8, 56, 8, 1.0, 5.0, [(8, 3)] + [x.getBlocksize(8, 56, 2, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("hsw_only_test1", precision, 8, 2, 1, 8, 0, 8, 1.0, 0.0, [(8,1)] + [x.getBlocksize(8, 2, 2, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(1, 2, 1), delta))
    kernels.append(generator.TestKernel("hsw_only_test2", precision, 24, 40, 40, 32, 0, 24, 1000, 1.0, [(8, 2)] + [x.getBlocksize(24, 40, 2, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(40, 40, 20), delta))

    kernels.append(generator.TestKernel("hsw_only_test3", precision, 8, 2, 1, 8, 0, 16, -2.0, 0.0, [(8, 1)] + [x.getBlocksize(8, 2, 2, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(1, 2, 2), delta))
    kernels.append(generator.TestKernel("hsw_only_test4", precision, 24, 20, 10, 40, 0, 24, 35.222, 0.0, [] + [x.getBlocksize(8, 20, 2, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(10, 20, 1), delta))
    kernels.append(generator.TestKernel("hsw_only_test5", precision, 64, 5, 10, 64, 0, 64, 2.3, 0.0, [] + [x.getBlocksize(64, 5, 2, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(10, 5, 1), delta))
    kernels.append(generator.TestKernel("hsw_only_test6", precision, 8, 1, 1, 16, 0, 56, 1.0, 0.0, [(8, 1)] + [x.getBlocksize(8, 1, 2, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(1, 1, 1), delta))
    kernels.append(generator.TestKernel("hsw_only_test7", precision, 8, 24, 40, 8, 0, 8, 1.0, 333333.2222222, [(8,1)] + [x.getBlocksize(8, 24, 2, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(40, 24, 1), delta))

    kernels.append(generator.TestKernel("hsw_only_test8", precision, 8, 2, 1, 8, 1, 8, 2.5, 0.0, [(8,1)] + [x.getBlocksize(8, 2, 2, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("hsw_only_test9", precision, 32, 40, 40, 32, 60, 32, 2.0, -4.33, [(8,2)] + [x.getBlocksize(32, 40, 2, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("hsw_only_test10", precision, 56, 28, 56, 56, 56, 56, 0.1, 3.0, [x.getBlocksize(56, 28, 2, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("hsw_only_test11", precision, 8, 20, 8, 40, 10, 8, 234234.123123, 0.0, [(8,3)] + [x.getBlocksize(8, 20, 2, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("hsw_only_test12", precision, 64, 5, 10, 64, 12, 64, 1.0, 1.0, [] + [x.getBlocksize(64, 5, 2, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("hsw_only_test13", precision, 8, 1, 1, 16, 1, 56, 0.0, 123.0, [(8, 1)] + [x.getBlocksize(8, 1, 2, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("hsw_only_test14", precision, 8, 24, 40, 8, 41, 8, 2.0, 1.0, [] + [x.getBlocksize(8, 24, 2, v_size, precision) for x in blocksize_algs], None, None, delta))

    kernels.append(generator.TestKernel("itest4", precision, 4, 4, 4, 4, 4, 4, 2.0, 2.0, [(4, 4), (4,4,2), (4,4,4), (4,4,8)], None, None, delta))

    kernels.append(generator.TestKernel("itest1", precision, 8, 56, 56, 8, 0, 8, 1.0, 0.0, [(8, 4), (8,1)] + [x.getBlocksize(8, 56, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(56, 56, 30), delta))
    kernels.append(generator.TestKernel("itest2", precision, 8, 40, 40, 8, 40, 8, 3.0, 2.0, [(8, 5), (8,2)] + [x.getBlocksize(8, 40, 1, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("itest3", precision, 8, 56, 56, 8, 56, 8, 0.0, 0.0, [(8, 3), (8, 5)] + [x.getBlocksize(8, 56, 1, v_size, precision) for x in blocksize_algs], None, None, delta))

    kernels.append(generator.TestKernel("arm_only_test1", precision, 2, 3, 4, 2, 0, 2, 1.1233, 0.0, [(2, 1), (2,3)] + [x.getBlocksize(2, 3, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(4, 3, 5), delta))
    kernels.append(generator.TestKernel("arm_only_test2", precision, 2, 3, 4, 20, 0, 14, 1.0, 1.0, [(2, 2), (2,3)] + [x.getBlocksize(2, 3, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(4, 3, 5), delta))
    kernels.append(generator.TestKernel("arm_only_test3", precision, 32, 80, 50, 32, 0, 32, 1.0, 3.0, [(8, 5)] + [x.getBlocksize(32, 80, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(50, 80, 294), delta))
    kernels.append(generator.TestKernel("arm_only_test4", precision, 32, 32, 32, 34, 0, 32, 1.0, 0.0, [(4, 4), (4,3)] + [x.getBlocksize(32, 32, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(32, 32, 24), delta))
    kernels.append(generator.TestKernel("arm_only_test5", precision, 2, 1, 1, 2, 0, 8, 1.0, -1.0, [(2, 1)] + [x.getBlocksize(2, 1, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(1, 1, 1), delta))
    kernels.append(generator.TestKernel("arm_only_test6", precision, 2, 2, 2, 2, 0, 2, 2.0, 234234.123, [(2, 1)] + [x.getBlocksize(2, 2, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(2, 2, 1), delta))
    kernels.append(generator.TestKernel("arm_only_test7", precision, 16, 5, 7, 16, 0, 16, 0.0, -1.123, [(8, 1), (8,2)] + [x.getBlocksize(16, 5, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(7, 5, 35), delta))

    kernels.append(generator.TestKernel("arm_only_test8", precision, 2, 3, 4, 2, 4, 2, 1.0, 0.0, [(2, 1), (2,3)] + [x.getBlocksize(2, 3, 1, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("arm_only_test9", precision, 2, 3, 4, 20, 12, 14, 2.0, 1.123, [(2, 2), (2,3)] + [x.getBlocksize(2, 3, 1, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("arm_only_test10", precision, 32, 80, 50, 32, 50, 32, 0.0, 0.2, [(8, 5)] + [x.getBlocksize(32, 80, 1, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("arm_only_test11", precision, 32, 32, 32, 33, 68, 32, 1231.0, 14443.0, [(4, 4), (4,3)] + [x.getBlocksize(32, 32, 1, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("arm_only_test12", precision, 2, 1, 1, 2, 1, 8, 1.0, 3.0, [(2, 1)] + [x.getBlocksize(2, 1, 1, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("arm_only_test13", precision, 2, 3, 3, 2, 3, 2, 1.0, 0.0, [(2, 1)] + [x.getBlocksize(2, 3, 1, v_size, precision) for x in blocksize_algs], None, None, delta))
    kernels.append(generator.TestKernel("arm_only_test14", precision, 16, 5, 7, 16, 7, 16, 1.0, 1.0, [(8, 1), (8,2)] + [x.getBlocksize(16, 5, 1, v_size, precision) for x in blocksize_algs], None, None, delta))

    kernels.append(generator.TestKernel("sve_mixed_test1", precision, 9, 9, 9, 9, 9, 9, 1.0, 0.0, [(3, 3)] + [x.getBlocksize(9, 9, 1, v_size, precision) for x in blocksize_algs], None, None, delta_dp))
    kernels.append(generator.TestKernel("sve_mixed_test2", precision, 9, 9, 9, 9, 0, 9, 4.0, 2.5, [(3, 3)] + [x.getBlocksize(9, 9, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(9, 9, 20), delta_dp))
    kernels.append(generator.TestKernel("sve_mixed_test3", precision, 18, 18, 18, 18, 0, 18, 3.4, -2.5, [(1, 1), (3, 3), (6, 6)] + [x.getBlocksize(18, 18, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(18, 18, 59), delta_dp))
    kernels.append(generator.TestKernel("sve_mixed_test4", precision, 80, 80, 80, 80, 0, 80, 0.0, -2.5, [(4, 4), (8, 8)] + [x.getBlocksize(80, 80, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(80, 80, 312), delta_dp))
    kernels.append(generator.TestKernel("sve_mixed_test5", precision, 8, 8, 8, 10, 0, 8, 3.0, -0.9, [(2, 2), (4, 4)] + [x.getBlocksize(8, 8, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(8, 8, 6), delta_dp))
    kernels.append(generator.TestKernel("sve_mixed_test6", precision, 8, 8, 8, 10, 8, 8, 3.0, -0.9, [(2, 2), (4, 4)] + [x.getBlocksize(8, 8, 1, v_size, precision) for x in blocksize_algs], None, None, delta_dp))

    kernels.append(generator.TestKernel("sve_test4", precision, 4, 4, 4, 4, 4, 4, 2.0, 2.0, [(4, 4)], None, None, delta_dp))

    kernels.append(generator.TestKernel("sve_test1", precision, 8, 56, 56, 8, 0, 8, 1.0, 0.0, [(8, 4), (8,1)] + [x.getBlocksize(8, 56, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(56, 56, 30), delta_dp))
    kernels.append(generator.TestKernel("sve_test2", precision, 8, 40, 40, 8, 40, 8, 3.0, 2.0, [(8, 5), (8,2)] + [x.getBlocksize(8, 40, 1, v_size, precision) for x in blocksize_algs], None, None, delta_dp))
    kernels.append(generator.TestKernel("sve_test3", precision, 8, 56, 56, 8, 56, 8, 0.0, 0.0, [(8, 3), (8, 5)] + [x.getBlocksize(8, 56, 1, v_size, precision) for x in blocksize_algs], None, None, delta_dp))

    kernels.append(generator.TestKernel("sve_arm_only_test1", precision, 2, 3, 4, 2, 0, 2, 1.1233, 0.0, [(2, 1), (2,3)] + [x.getBlocksize(2, 3, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(4, 3, 5), delta_dp))
    kernels.append(generator.TestKernel("sve_arm_only_test2", precision, 2, 3, 4, 20, 0, 14, 1.0, 1.0, [(2, 2), (2,3)] + [x.getBlocksize(2, 3, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(4, 3, 5), delta_dp))
    kernels.append(generator.TestKernel("sve_arm_only_test3", precision, 32, 80, 50, 32, 0, 32, 1.0, 3.0, [(8, 5)] + [x.getBlocksize(32, 80, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(50, 80, 294), delta_dp))
    kernels.append(generator.TestKernel("sve_arm_only_test4", precision, 32, 32, 32, 34, 0, 32, 1.0, 0.0, [(4, 4), (4,3)] + [x.getBlocksize(32, 32, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(32, 32, 24), delta_dp))
    kernels.append(generator.TestKernel("sve_arm_only_test5", precision, 2, 1, 1, 2, 0, 8, 1.0, -1.0, [(2, 1)] + [x.getBlocksize(2, 1, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(1, 1, 1), delta_dp))
    kernels.append(generator.TestKernel("sve_arm_only_test6", precision, 2, 2, 2, 2, 0, 2, 2.0, 234234.123, [(2, 1)] + [x.getBlocksize(2, 2, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(2, 2, 1), delta_dp))
    kernels.append(generator.TestKernel("sve_arm_only_test7", precision, 16, 5, 7, 16, 0, 16, 0.0, -1.123, [(8, 1), (8,2)] + [x.getBlocksize(16, 5, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(7, 5, 35), delta_dp))

    kernels.append(generator.TestKernel("sve_arm_only_test8", precision, 2, 3, 4, 2, 4, 2, 1.0, 0.0, [(2, 1), (2,3)] + [x.getBlocksize(2, 3, 1, v_size, precision) for x in blocksize_algs], None, None, delta_dp))
    kernels.append(generator.TestKernel("sve_arm_only_test9", precision, 2, 3, 4, 20, 12, 14, 2.0, 1.123, [(2, 2), (2,3)] + [x.getBlocksize(2, 3, 1, v_size, precision) for x in blocksize_algs], None, None, delta_dp))
    kernels.append(generator.TestKernel("sve_arm_only_test10", precision, 32, 80, 50, 32, 50, 32, 0.0, 0.2, [(8, 5)] + [x.getBlocksize(32, 80, 1, v_size, precision) for x in blocksize_algs], None, None, delta_dp))
    kernels.append(generator.TestKernel("sve_arm_only_test11", precision, 32, 32, 32, 33, 68, 32, 1231.0, 14443.0, [(4, 4), (4,3)] + [x.getBlocksize(32, 32, 1, v_size, precision) for x in blocksize_algs], None, None, delta_dp))
    kernels.append(generator.TestKernel("sve_arm_only_test12", precision, 2, 1, 1, 2, 1, 8, 1.0, 3.0, [(2, 1)] + [x.getBlocksize(2, 1, 1, v_size, precision) for x in blocksize_algs], None, None, delta_dp))
    kernels.append(generator.TestKernel("sve_arm_only_test13", precision, 2, 3, 3, 2, 3, 2, 1.0, 0.0, [(2, 1)] + [x.getBlocksize(2, 3, 1, v_size, precision) for x in blocksize_algs], None, None, delta_dp))
    kernels.append(generator.TestKernel("sve_arm_only_test14", precision, 16, 5, 7, 16, 7, 16, 1.0, 1.0, [(8, 1), (8,2)] + [x.getBlocksize(16, 5, 1, v_size, precision) for x in blocksize_algs], None, None, delta_dp))

    kernels.append(generator.TestKernel("sve_arm_only_test15", precision, 23, 29, 31, 23, 31, 23, 1.32, 0.96, [x.getBlocksize(23, 29, 1, v_size, precision) for x in blocksize_algs], None, None, delta_dp))
    kernels.append(generator.TestKernel("sve_arm_only_test16", precision, 23, 29, 31, 23, 0, 23, 1.32, 0.96, [x.getBlocksize(23, 29, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(31, 29, 61), delta_dp))

    kernels.append(generator.TestKernel("sve_single_prec_test_S1", precision, 9, 9, 9, 9, 9, 9, 1.24, 0.87, [x.getBlocksize(9, 9, 1, v_size, precision) for x in blocksize_algs], None, None, delta_sp))
    kernels.append(generator.TestKernel("sve_single_prec_test_S2", precision, 15, 15, 15, 15, 15, 15, -3.14, 6.28, [x.getBlocksize(15, 15, 1, v_size, precision) for x in blocksize_algs], None, None, delta_sp))
    kernels.append(generator.TestKernel("sve_single_prec_test_S3", precision, 23, 23, 23, 23, 23, 23, 1.5, -0.66, [x.getBlocksize(23, 23, 1, v_size, precision) for x in blocksize_algs], None, None, delta_sp))
    kernels.append(generator.TestKernel("sve_single_prec_test_S4", precision, 23, 31, 13, 23, 13, 23, 2.0, 0.0, [x.getBlocksize(23, 31, 1, v_size, precision) for x in blocksize_algs], None, None, delta_sp))
    kernels.append(generator.TestKernel("sve_single_prec_test_S5", precision, 9, 9, 9, 9, 0, 9, 1.24, 0.87, [x.getBlocksize(9, 9, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(9, 9, 8), delta_sp))
    kernels.append(generator.TestKernel("sve_single_prec_test_S6", precision, 15, 15, 15, 15, 0, 15, -3.14, 6.28, [x.getBlocksize(15, 15, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(15, 15, 22), delta_sp))
    kernels.append(generator.TestKernel("sve_single_prec_test_S7", precision, 23, 23, 23, 23, 0, 23, 1.5, -0.66, [x.getBlocksize(23, 23, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(23, 23, 52), delta_sp))
    kernels.append(generator.TestKernel("sve_single_prec_test_S8", precision, 23, 31, 13, 23, 0, 23, 2.0, 0.0, [x.getBlocksize(23, 31, 1, v_size, precision) for x in blocksize_algs], None, generator.generateMTX(13, 31, 40), delta_sp))

generator.make(kernels, arch)
