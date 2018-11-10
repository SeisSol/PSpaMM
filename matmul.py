from typing import Tuple

from codegen.ast import *
from codegen.sugar import *
from codegen.forms import *

import scripts.old_arm
import scripts.max_bn_knl

from cursors import *

import architecture
import numpy

def decompose_pattern(k, n, pattern:Matrix[bool], bk:int, bn:int) -> Tuple[Matrix[int], List[Matrix[bool]]]:
    Bk,Bn = k//bk, n//bn
    patterns = []
    x = 0

    n_overhead = n % bn
    k_overhead = k % bk

    if n_overhead > 0:
        Bn += 1
    if k_overhead > 0:
        Bk += 1

    blocks = Matrix.full(Bk,Bn,-1)

    for Bni in range(Bn):
        for Bki in range(Bk):
            if Bni + 1 == Bn and n_overhead > 0 and Bki + 1 == Bk and k_overhead > 0:
                block = pattern[(Bki*bk):((Bki+1)*bk+k_overhead), (Bni*bn):((Bni)*bn+n_overhead)]
            elif Bni + 1 == Bn and n_overhead > 0:
                block = pattern[(Bki*bk):((Bki+1)*bk), (Bni*bn):((Bni)*bn+n_overhead)]
            elif Bki + 1 == Bk and k_overhead > 0:
                block = pattern[(Bki*bk):((Bki+1)*bk+k_overhead), (Bni*bn):((Bni+1)*bn)]
            else:
                block = pattern[(Bki*bk):((Bki+1)*bk), (Bni*bn):((Bni+1)*bn)]
            
            blocks[Bki,Bni] = x
            x += 1
            patterns.append(block)

    mtx_overhead = numpy.zeros(n)

    for i in range(n):
        for j in range(k, pattern.rows):
            if pattern[j, i]:
                mtx_overhead[i] += 1

    return blocks, patterns, mtx_overhead

class MatMul:
    def __init__(self,
                 m: int, 
                 n: int, 
                 k: int, 
                 lda: int, 
                 ldb: int, 
                 ldc: int,
                 beta: int,
                 mtx_filename: str,
                 mtx_format: str = 'any',
                 output_funcname: str = None,
                 output_filename: str = None,
                 bm: int = None, 
                 bn: int = None, 
                 bk: int = 2,
                 arch: str = 'knl',
                 prefetching: str = None,
                 **kwargs  # Accept and ignore args which don't belong
                 ) -> None:

        self.m = m
        self.n = n
        self.k = k

        self.lda = lda
        self.ldb = ldb
        self.ldc = ldc

        self.beta = beta

        if bm == None or bn == None:
            if arch == 'knl':
                (self.bm, self.bn) = scripts.max_bn_knl.getBlocksize(m, n, bk)
            elif arch == 'arm':
                (self.bm, self.bn) = scripts.old_arm.getBlocksize(m, n)
        else: 
            self.bm = bm
            self.bn = bn

        self.bk = bk

        self.arch = arch

        self.prefetching = prefetching

        self.mtx_filename = mtx_filename
        self.mtx_format = mtx_format

        self.output_funcname = output_funcname
        self.output_filename = output_filename

        if ldb == 0:
            pattern = Matrix.load(mtx_filename)
        else:
            mtx = numpy.zeros((k, n))
            for i in range(k):
                for j in range(n):
                    mtx[i, j] = 1
            pattern = Matrix(mtx)

        
        self.nnz = 0

        if ldb == 0:
            for i in range(n):
                for j in range(k):
                    if pattern[j,i]:
                        self.nnz += 1
        else:
            self.nnz = ldb * self.n

        self.flop = self.nnz * m * 2

        blocks,patterns,mtx_overhead = decompose_pattern(self.k, self.n, pattern, self.bk, self.bn)


        architecture.init()
        architecture.arch = arch
        architecture.Generator = architecture.get_class("codegen.architectures." + arch + ".generator.Generator")
        architecture.operands = architecture.get_class("codegen.architectures." + arch + ".operands")

        self.generator = architecture.Generator()

        self.generator.init_prefetching(self.prefetching)

        self.v_size = self.generator.get_v_size()

        assert(self.m % self.v_size == 0)

        self.A_regs, self.B_regs, self.C_regs, self.starting_regs, self.loop_reg, self.additional_regs = self.generator.make_reg_blocks(self.bm, self.bn, self.bk, self.v_size, self.nnz)

        self.A = DenseCursor("A", self.starting_regs[0], self.m, self.k, self.lda, self.bm, self.bk)
        self.B = BlockCursor("B", self.starting_regs[1], self.k, self.n, self.ldb, self.bk, self.bn, blocks, patterns,mtx_overhead)
        self.C = DenseCursor("C", self.starting_regs[2], self.m, self.n, self.ldc, self.bm, self.bn)


    def make_nk_unroll(self):

        asm = block("Unrolling over bn and bk")
        A_ptr = CursorLocation()
        B_ptr = self.B.start()
        C_ptr = CursorLocation()

        Bn = self.n // self.bn
        Bk = self.k // self.bk
        vm = self.bm // self.v_size

        n_overhead = self.n % self.bn
        k_overhead = self.k % self.bk

        if n_overhead > 0:
            Bn += 1
        if k_overhead > 0:
            Bk += 1

        asm.add(self.generator.make_b_pointers(self.starting_regs[1], self.additional_regs, self.nnz))

        for Bni in range(0,Bn):
            if Bni + 1 == Bn and n_overhead > 0:
                regs = self.C_regs[0:vm, 0:n_overhead]
            else:
                regs = self.C_regs
            if self.beta == 1:
                asm.add(self.generator.move_register_block(self.C, C_ptr, Coords(), regs, self.v_size, self.additional_regs, None, False))
            else:
                asm.add(self.generator.make_zero_block(regs, self.additional_regs))

            for Bki in range(0,Bk):

                to_A = Coords(right=Bki)
                to_B = Coords(right=Bni, down=Bki, absolute=True)

                if self.B.has_nonzero_block(B_ptr, to_B):
                    asm.add(self.generator.make_microkernel(self.A, self.B, A_ptr, B_ptr, self.A_regs, self.B_regs, regs, self.v_size, self.additional_regs, to_A, to_B))

            asm.add(self.generator.move_register_block(self.C, C_ptr, Coords(), regs, self.v_size, self.additional_regs, None, True, self.prefetching))

            if (Bni != Bn-1):
                move_C, C_ptr = self.C.move(C_ptr, Coords(right=1))
                asm.add(move_C)

        return asm



    def make(self):
        
        A_ptr = CursorLocation()
        C_ptr = CursorLocation()

        Bm = self.m // self.bm
        Bn = self.n // self.bn
        Bk = self.k // self.bk

        if self.n % self.bn != 0:
            Bn += 1

        asm = block("unrolled_{}x{}x{}".format(self.m,self.n,self.k),
            self.generator.make_scaling_offsets(self.additional_regs, self.nnz),
            loop(self.loop_reg, 0, Bm, 1).body(
                self.make_nk_unroll(),
                self.A.move(A_ptr, Coords(down=1))[0],
                self.C.move(C_ptr, Coords(down=1, right=1-Bn))[0]
            )
        )

        vm_overhead = (self.m % self.bm) // self.v_size

        if vm_overhead > 0:
            self.m = self.m % self.bm
            self.bm = self.m % self.bm
            self.A_regs = self.A_regs[0:self.bm // self.v_size, 0:self.bk]
            self.C_regs = self.C_regs[0:self.bm // self.v_size, 0:self.bn]
            self.A.r = self.m
            asm.add(self.make_nk_unroll())


        return asm
