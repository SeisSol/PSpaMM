from typing import Tuple

from codegen.ast import *
from codegen.sugar import *
from codegen.forms import *

from cursors import *

import architecture
import numpy

def decompose_pattern(pattern:Matrix[bool], bk:int, bn:int) -> Tuple[Matrix[int], List[Matrix[bool]]]:
    k,n = pattern.shape
    Bk,Bn = k//bk, n//bn
    patterns : List[Matrix[bool]] = []
    blocks = Matrix.full(Bk,Bn,-1)
    x = 0

    for Bni in range(Bn):
        for Bki in range(Bk):
            block = pattern[(Bki*bk):((Bki+1)*bk), (Bni*bn):((Bni+1)*bn)]
            found = False
            for pi in range(len(patterns)):
                if patterns[pi] == block:
                    blocks[Bki,Bni] = pi
                    found = True
            if not found:
                blocks[Bki,Bni] = x
                x += 1
                patterns.append(block)

    return blocks, patterns



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
                 bk: int = None,
                 arch: str = 'knl',
                 **kwargs  # Accept and ignore args which don't belong
                 ) -> None:

        self.m = m
        self.n = n
        self.k = k

        self.lda = lda
        self.ldb = ldb
        self.ldc = ldc

        self.beta = beta

        self.bm = bm
        self.bn = bn
        self.bk = bk

        self.arch = arch

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

        blocks,patterns = decompose_pattern(pattern, self.bk, self.bn)


        architecture.init()
        architecture.arch = arch
        architecture.generator = architecture.get_class("codegen.architectures." + arch + ".generator")
        architecture.operands = architecture.get_class("codegen.architectures." + arch + ".operands")

        self.v_size = architecture.generator.v_size

        self.A_regs, self.B_regs, self.C_regs, self.starting_regs, self.loop_reg, self.additional_regs = architecture.generator.make_reg_blocks(self.bm, self.bn, self.bk, self.v_size)

        self.A = DenseCursor("A", self.starting_regs[0], self.m, self.k, self.lda, self.bm, self.bk)
        self.B = BlockCursor("B", self.starting_regs[1], self.k, self.n, self.ldb, self.bk, self.bn, blocks, patterns)
        self.C = DenseCursor("C", self.starting_regs[2], self.m, self.n, self.ldc, self.bm, self.bn)


    def make_nk_unroll(p):

        asm = block("Unrolling over bn and bk")
        A_ptr = CursorLocation()
        B_ptr = p.B.start()
        C_ptr = CursorLocation()
        Bn = p.n // p.bn
        Bk = p.k // p.bk

        for Bni in range(0,Bn):

            if p.beta == 1:
                asm.add(architecture.generator.move_register_block(p.C, C_ptr, Coords(), p.C_regs, p.v_size, p.additional_regs, None, False))
            else:
                asm.add(architecture.generator.make_zero_block(p.C_regs, p.additional_regs))

            for Bki in range(0,Bk):

                to_A = Coords(right=Bki)
                to_B = Coords(right=Bni, down=Bki, absolute=True)

                if p.B.has_nonzero_block(B_ptr, to_B):
                    asm.add(architecture.generator.make_microkernel(p.A, p.B, A_ptr, B_ptr, p.A_regs, p.B_regs, p.C_regs, p.v_size, p.additional_regs, to_A, to_B))

            asm.add(architecture.generator.move_register_block(p.C, C_ptr, Coords(), p.C_regs, p.v_size, p.additional_regs, None, True))

            if (Bni != Bn-1):
                move_C, C_ptr = p.C.move(C_ptr, Coords(right=1))
                asm.add(move_C)

        return asm



    def make(p):
        
        A_ptr = CursorLocation()
        C_ptr = CursorLocation()

        Bm = p.m // p.bm
        Bn = p.n // p.bn
        Bk = p.k // p.bk

        asm = block(f"unrolled_{p.m}x{p.n}x{p.k}",

            loop(p.loop_reg, 0, Bm, 1).body(
                p.make_nk_unroll(),
                p.A.move(A_ptr, Coords(down=1))[0],
                p.C.move(C_ptr, Coords(down=1, right=1-Bn))[0]
            )
        )
        return asm