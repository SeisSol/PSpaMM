#!/usr/bin/env python3

import testsuite_generator as generator

import pspamm.scripts.max_knl as max_square
import pspamm.scripts.max_bn_knl as max_bn
import pspamm.scripts.old_knl as old
from pspamm.codegen.precision import *

blocksize_algs = [max_square, max_bn, old]

kernels = []

for precision in (Precision.SINGLE, Precision.DOUBLE):
    kernels.append(generator.SparseKernel(f"test1_{precision}", precision, 8, 56, 56, 8, 0, 8, 2.0, 0.0, [(8, 4), (8,1)] + [x.getBlocksize(8, 56, 2) for x in blocksize_algs], generator.generateMTX(56, 56, 30), 0.0000001))
    kernels.append(generator.DenseKernel(f"test2_{precision}", precision, 8, 40, 40, 8, 40, 8, 2.5, 1.0, [(8, 5), (8,2)] + [x.getBlocksize(8, 40, 2) for x in blocksize_algs], 0.0000001))
    kernels.append(generator.DenseKernel(f"test3_{precision}", precision, 8, 56, 56, 8, 56, 8, 1.0, 5.0, [(8, 3), (8, 5)] + [x.getBlocksize(8, 56, 2) for x in blocksize_algs], 0.0000001))
    kernels.append(generator.SparseKernel(f"knl_only_test1_{precision}", precision, 8, 2, 1, 8, 0, 8, 1.0, 0.0, [(8,1)] + [x.getBlocksize(8, 2, 2) for x in blocksize_algs], generator.generateMTX(1, 2, 1), 0.0000001))
    kernels.append(generator.SparseKernel(f"knl_only_test2_{precision}", precision, 24, 40, 40, 32, 0, 24, 1000, 1.0, [(8, 2), (16,7)] + [x.getBlocksize(24, 40, 2) for x in blocksize_algs], generator.generateMTX(40, 40, 20), 0.0000001))

    kernels.append(generator.SparseKernel(f"knl_only_test3_{precision}", precision, 8, 2, 1, 8, 0, 16, -2.0, 0.0, [(8, 1)] + [x.getBlocksize(8, 2, 2) for x in blocksize_algs], generator.generateMTX(1, 2, 2), 0.0000001))
    kernels.append(generator.SparseKernel(f"knl_only_test4_{precision}", precision, 24, 20, 10, 40, 0, 24, 35.222, 0.0, [(8, 20), (24,3)] + [x.getBlocksize(8, 20, 2) for x in blocksize_algs], generator.generateMTX(10, 20, 1), 0.0000001))
    kernels.append(generator.SparseKernel(f"knl_only_test5_{precision}", precision, 64, 5, 10, 64, 0, 64, 2.3, 0.0, [(32, 2), (8,14)] + [x.getBlocksize(64, 5, 2) for x in blocksize_algs], generator.generateMTX(10, 5, 1), 0.0000001))
    kernels.append(generator.SparseKernel(f"knl_only_test6_{precision}", precision, 8, 1, 1, 16, 0, 56, 1.0, 0.0, [(8, 1)] + [x.getBlocksize(8, 1, 2) for x in blocksize_algs], generator.generateMTX(1, 1, 1), 0.0000001))
    kernels.append(generator.SparseKernel(f"knl_only_test7_{precision}", precision, 8, 24, 40, 8, 0, 8, 1.0, 333333.2222222, [(8, 24), (8,1)] + [x.getBlocksize(8, 24, 2) for x in blocksize_algs], generator.generateMTX(40, 24, 1), 0.0000001))

    kernels.append(generator.DenseKernel(f"knl_only_test8_{precision}", precision, 8, 2, 1, 8, 1, 8, 2.5, 0.0, [(8,1)] + [x.getBlocksize(8, 2, 2) for x in blocksize_algs], 0.0000001))
    kernels.append(generator.DenseKernel(f"knl_only_test9_{precision}", precision, 32, 40, 40, 32, 60, 32, 2.0, -4.33, [(8,2), (16,7)] + [x.getBlocksize(32, 40, 2) for x in blocksize_algs], 0.0000001))
    kernels.append(generator.DenseKernel(f"knl_only_test10_{precision}", precision, 56, 28, 56, 56, 56, 56, 0.1, 3.0, [(8, 28)], 0.0000001))
    kernels.append(generator.DenseKernel(f"knl_only_test11_{precision}", precision, 8, 20, 8, 40, 10, 8, 234234.123123, 0.0, [(8, 20), (8,3)] + [x.getBlocksize(8, 20, 2) for x in blocksize_algs], 0.0000001))
    kernels.append(generator.DenseKernel(f"knl_only_test12_{precision}", precision, 64, 5, 10, 64, 12, 64, 1.0, 1.0, [(32, 2), (8,14)] + [x.getBlocksize(64, 5, 2) for x in blocksize_algs], 0.0000001))
    kernels.append(generator.DenseKernel(f"knl_only_test13_{precision}", precision, 8, 1, 1, 16, 1, 56, 0.0, 123.0, [(8, 1)] + [x.getBlocksize(8, 1, 2) for x in blocksize_algs], 0.0000001))
    kernels.append(generator.DenseKernel(f"knl_only_test14_{precision}", precision, 8, 24, 40, 8, 41, 8, 2.0, 1.0, [(8, 24)] + [x.getBlocksize(8, 24, 2) for x in blocksize_algs], 0.0000001))

for arch in ('knl512', 'knl256', 'knl128'):
    generator.make(kernels, arch)


