from typing import Tuple

from pspamm.codegen.ast import *
from pspamm.codegen.sugar import *
from pspamm.codegen.forms import *
from pspamm.codegen.precision import *

from pspamm.cursors import *

from pspamm.codegen.virtual import *
from pspamm.codegen.prune import *

import pspamm.architecture
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

    mtx_overhead = [0] * n

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
                 alpha: str,
                 beta: str,
                 mtx_filename: str,
                 amtx_filename: str,
                 bmtx_filename: str,
                 mtx_format: str = 'any',
                 output_funcname: str = None,
                 output_filename: str = None,
                 output_overwrite: bool = False,
                 bm: int = None, 
                 bn: int = None, 
                 bk: int = None,
                 arch: str = 'knl',
                 precision: str = 'd',
                 prefetching: str = None,
                 **kwargs  # Accept and ignore args which don't belong
                 ) -> None:

        self.m = m
        self.n = n
        self.k = k

        self.lda = lda
        self.ldb = ldb
        self.ldc = ldc

        try:
          self.alpha = float(alpha)
        except:
          self.alpha = 'generic'
        try:
          self.beta = float(beta)
        except:
          self.beta = 'generic'

        if arch.startswith('skx'):
          arch = 'knl' + arch[3:]

        # hacky implementation of multi-register length
        if arch.startswith('arm_sve'):
          if len(arch) == 7:
            v_len_regs = 4 # compatibility: arm_sve == arm_sve512
          else:
            v_len_bits = int(arch[7:])
            assert v_len_bits % 128 == 0 and v_len_bits <= 2048
            v_len_regs = v_len_bits // 128
          arch = 'arm_sve'
        
        if arch.startswith('knl'):
          if len(arch) == 3:
            v_len_regs = 4
          else:
            v_len_bits = int(arch[3:])
            assert v_len_bits in (128, 256, 512)
            v_len_regs = v_len_bits // 128
          arch = 'knl'
        
        if arch.startswith('hsw'):
          if len(arch) == 3:
            v_len_regs = 2
          else:
            v_len_bits = int(arch[3:])
            assert v_len_bits in (128, 256)
            v_len_regs = v_len_bits // 128
          arch = 'hsw'
        
        if arch.startswith('rvv'):
          if len(arch) == 3:
            v_len_regs = 1
          else:
            v_len_bits = int(arch[3:])
            assert v_len_bits in (128, 256, 512, 1024, 2048, 4096, 8192)
            v_len_regs = v_len_bits // 128
          arch = 'rvv'
        
        if arch.startswith('arm') and not arch.startswith('arm_sve'):
          # only 128 supported
          v_len_regs = 1
          arch = 'arm'

        self.arch = arch
        assert precision.lower() in ['bf16', 'h', 's', 'd']
        self.precision = {
            'h' : Precision.HALF,
            's' : Precision.SINGLE,
            'd' : Precision.DOUBLE,
            'bf16' : Precision.BFLOAT16
        }[precision.lower()]

        pspamm.architecture.init()
        pspamm.architecture.arch = arch
        pspamm.architecture.Generator = pspamm.architecture.get_class("pspamm.codegen.architectures." + arch + ".generator").Generator
        pspamm.architecture.operands = pspamm.architecture.get_class("pspamm.codegen.architectures." + arch + ".operands")
        pspamm.architecture.blocksize = pspamm.architecture.get_class("pspamm.codegen.architectures." + arch + ".blocksize").Default

        self.generator = pspamm.architecture.Generator(self.precision)

        # flag that determines if a matmul kernel uses sve instructions -> needed for sve predicates
        self.masks = self.generator.has_masks()
        # define which architectures need to use an explicit broadcast, necessary for alpha/beta values
        self.use_bcst = self.generator.use_broadcast()

        self.generator.v_len = v_len_regs

        self.v_size = self.generator.get_v_size()

        if bk == None:
            bk = 2 if arch == 'knl' else 1

        if bm == None or bn == None:
            (self.bm, self.bn, self.bk) = pspamm.architecture.blocksize.getBlocksize(m, n, bk, self.v_size, self.precision)
        else: 
            self.bm = bm
            self.bn = bn
            self.bk = bk

        self.prefetching = prefetching

        self.output_funcname = output_funcname
        self.output_filename = output_filename
        self.output_overwrite = output_overwrite

        if ldb == 0:
            if bmtx_filename is None or bmtx_filename == '':
                bmtx_filename = mtx_filename
            bpattern = Matrix.load(bmtx_filename)
            self.generator.set_sparse()
        else:
            bpattern = Matrix.full(k, n, True)
            assert self.k <= ldb
        
        if lda == 0:
            apattern = Matrix.load(amtx_filename)
            self.generator.set_sparse()
        else:
            apattern = Matrix.full(m, k, True)
            assert self.m <= lda
        
        self.bmtx_filename = bmtx_filename
        self.amtx_filename = amtx_filename
        self.mtx_format = mtx_format
        
        assert self.m <= ldc

        self.bnnz = bpattern.nnz()
        self.annz = apattern.nnz()

        # compute flops by splitting into outer products over k
        kannz = apattern.nnz(1)
        kbnnz = bpattern.nnz(0)
        self.flop = 2 * sum(ka * kb for ka,kb in zip(kannz, kbnnz))

        # if matrices are always padded to multiple of v_size, we can remove the if-part and execute the assert for SVE too
        if not self.masks:
            assert(self.m % self.v_size == 0)

        self.A_regs, self.B_regs, self.C_regs, self.starting_regs, self.alpha_reg, self.beta_reg, self.loop_regs, self.additional_regs, self.mask_regs, self.prefetch_reg = self.generator.make_reg_blocks(self.bm, self.bn, self.bk, self.v_size, self.bnnz, self.m, self.n, self.k, self.prefetching)

        self.A_pool = RegisterPool([self.A_regs[i,j] for i in range(self.A_regs.shape[0]) for j in range(self.A_regs.shape[1])])
        self.B_pool = RegisterPool([self.B_regs[i,j] for i in range(self.B_regs.shape[0]) for j in range(self.B_regs.shape[1])])
        self.C_pool = RegisterPool([self.C_regs[i,j] for i in range(self.C_regs.shape[0]) for j in range(self.C_regs.shape[1])])

        self.alpha_bcst_reg, self.beta_bcst_reg = self.starting_regs[3], self.starting_regs[4]

        if lda == 0:
            blocks, patterns, mtx_overhead = decompose_pattern(self.m, self.k, apattern, self.bm, self.bk)
            self.A = BlockCursor("A", self.starting_regs[0], self.m, self.k, self.lda, self.bm, self.bk, self.precision.size(), blocks, patterns, mtx_overhead)
            self.annz += sum(mtx_overhead)
        else:
            self.A = DenseCursor("A", self.starting_regs[0], self.m, self.k, self.lda, self.bm, self.bk, self.precision.size())
        if ldb == 0:
            blocks, patterns, mtx_overhead = decompose_pattern(self.k, self.n, bpattern, self.bk, self.bn)
            self.B = BlockCursor("B", self.starting_regs[1], self.k, self.n, self.ldb, self.bk, self.bn, self.precision.size(), blocks, patterns, mtx_overhead)
            self.bnnz += sum(mtx_overhead)
        else:
            self.B = DenseCursor("B", self.starting_regs[1], self.k, self.n, self.ldb, self.bk, self.bn, self.precision.size())
        self.C = DenseCursor("C", self.starting_regs[2], self.m, self.n, self.ldc, self.bm, self.bn, self.precision.size())
        self.C_pf = DenseCursor("C_pf", self.starting_regs[5], self.m, self.n, self.ldc, self.bm, self.bn, self.precision.size()) if self.prefetch_reg else None

        self.unroll_n = ldb == 0
        self.unroll_m = lda == 0

        # use unused loop registers for scaling instead
        if self.unroll_m:
            self.additional_regs += [self.loop_regs[0]]
        if self.unroll_n:
            self.additional_regs += [self.loop_regs[1]]
        if self.unroll_m or self.unroll_n:
            self.additional_regs += [self.loop_regs[2]]

    def microkernel(self, asm, Bmi, Bni, unroll, A_ptr, B_ptr, C_ptr, C_pf_ptr):
        Bn = self.n // self.bn
        Bk = self.k // self.bk
        Bm = self.m // self.bm

        vm = self.generator.ceil_div(self.bm, self.v_size)

        n_overhead = self.n % self.bn
        k_overhead = self.k % self.bk
        m_overhead = self.m % self.bm
        vm_overhead = -(m_overhead // -self.v_size)

        if n_overhead > 0:
            Bn += 1
        if k_overhead > 0:
            Bk += 1
        if m_overhead > 0:
            Bm += 1

        regs = Matrix([[VirtualRegister(self.C_regs[0,0].typeinfo, self.C_pool) for _ in range(self.C_regs.shape[1])] for _ in range(self.C_regs.shape[0])])

        BnEnd = Bni + 1 == Bn
        BmEnd = Bmi + 1 == Bm

        if BnEnd and n_overhead > 0:
            regs = regs[:, :n_overhead]
        if BmEnd and m_overhead > 0:
            regs = regs[:vm_overhead, :]
        
        C_ptr_in = CursorLocation(Coords(right=Bni, down=Bmi, absolute=True))
        to_C = Coords()
        C_ptr_pf_in = C_ptr_in

        if self.alpha in [-1.0, 1.0] and self.beta != 0.0:
            asm.add(self.generator.move_register_block(self.C, C_ptr_in, to_C, regs, self.v_size, self.additional_regs, None, False))
            if self.beta != 1.0:
                if self.use_bcst:
                    asm.add(bcst(self.beta_bcst_reg, self.beta_reg[1], "Broadcast beta"))
                for ic in range(regs.shape[1]):
                    for ir in range(regs.shape[0]):
                        pred_m = None if not self.masks else self.generator.pred_n_trues(self.bm - ir * self.v_size, self.v_size, "m")
                        asm.add(mul(regs[ir,ic], self.beta_reg[1], regs[ir,ic], "C = beta * C", pred=pred_m))
        else:
            asm.add(self.generator.make_zero_block(regs, self.additional_regs))

        def kernelK(asm, Bki):
            if unroll:
                # adjust registers if necessary for the last operation

                if BmEnd and m_overhead > 0 and not self.unroll_m:
                    A_ptr_in = CursorLocation(Coords(right=0, down=Bmi, absolute=True))
                else:
                    A_ptr_in = A_ptr
                to_A = Coords(right=Bki, down=Bmi, absolute=True) if self.unroll_m else Coords(right=Bki)

                if BnEnd and n_overhead > 0 and not self.unroll_n:
                    B_ptr_in = CursorLocation(Coords(down=0, right=Bni, absolute=True))
                else:
                    B_ptr_in = B_ptr
                to_B = Coords(right=Bni, down=Bki, absolute=True) if self.unroll_n else Coords(down=Bki)
                keep = (not self.unroll_n or self.B.has_nonzero_block(B_ptr_in, to_B)) and (not self.unroll_m or self.A.has_nonzero_block(A_ptr_in, to_A))
            else:
                # setting A_ptr, B_ptr here may be a bit too hacky...
                A_ptr_in = CursorLocation(Coords(right=Bki, down=Bmi, absolute=True))
                B_ptr_in = CursorLocation(Coords(right=Bni, down=Bki, absolute=True))
                to_A = Coords()
                to_B = Coords()
                keep = True
            
            sub = self.alpha == -1.0

            if keep:
                A_regs = Matrix([[VirtualRegister(self.A_regs[0,0].typeinfo, self.A_pool) for _ in range(self.A_regs.shape[1])] for _ in range(self.A_regs.shape[0])])
                B_regs = Matrix([[VirtualRegister(self.B_regs[0,0].typeinfo, self.B_pool) for _ in range(self.B_regs.shape[1])] for _ in range(self.B_regs.shape[0])])
                asm.add(self.generator.make_microkernel(self.A, self.B, A_ptr_in, B_ptr_in, A_regs, B_regs, regs, self.v_size, self.additional_regs, to_A, to_B, sub))

        self.loopwrap(asm, kernelK, Bk, k_overhead > 0, unroll, self.loop_regs[2], [self.A, self.B], [A_ptr, B_ptr], ['right', 'down'], loopunroll=1, overlap=True)

        if self.alpha not in [-1.0, 1.0]:
            store_block = block("")

            if self.use_bcst:
                store_block.add(bcst(self.alpha_bcst_reg, self.alpha_reg[1], "Broadcast alpha"))
                if self.beta != 0.0 and self.beta != 1.0:
                    store_block.add(bcst(self.beta_bcst_reg, self.beta_reg[1], "Broadcast beta"))

            for x in range(0, regs.shape[1], self.A_regs.shape[1]):
                A_regs = Matrix([[VirtualRegister(self.A_regs[0,0].typeinfo, self.A_pool) for _ in range(self.A_regs.shape[1])] for _ in range(self.A_regs.shape[0])])
                A_regs_cut = A_regs[0:min(self.A_regs.shape[0], regs.shape[0]), 0:regs.shape[1]-x]
                if self.beta != 0.0:
                    store_block.add(self.generator.move_register_block(self.C, C_ptr_in, to_C, A_regs_cut, self.v_size, self.additional_regs, None, False, None, self.ldc * x))

                for ir in range(A_regs_cut.shape[0]):
                    for ic in range(A_regs_cut.shape[1]):
                        pred_m = None if not self.masks else self.generator.pred_n_trues(self.bm - ir*self.v_size, self.v_size, "m")
                        if self.beta != 0.0 and self.beta != 1.0:
                            store_block.add(mul(A_regs_cut[ir,ic], self.beta_reg[1], A_regs_cut[ir,ic], "C = beta * C + alpha * AB", pred=pred_m))
                        
                        if self.beta == 0.0:
                            store_block.add(mul(regs[ir, x + ic], self.alpha_reg[1], A_regs_cut[ir, ic], "C = alpha * AB", pred=pred_m))
                        else:
                            store_block.add(fma(regs[ir, x + ic], self.alpha_reg[1], A_regs_cut[ir, ic], "C = C + alpha * AB", None, pred=pred_m))
                store_block.add(self.generator.move_register_block(self.C, C_ptr_in, to_C, A_regs_cut, self.v_size, self.additional_regs, None, True, self.prefetching, self.ldc * x, self.C_pf, C_pf_ptr))
            asm.add(store_block)
        else:
            asm.add(self.generator.move_register_block(self.C, C_ptr_in, to_C, regs, self.v_size, self.additional_regs, None, True, self.prefetching, 0, self.C_pf, C_pf_ptr))

    def blockloop(self, asm, A_ptr, B_ptr, C_ptr, C_pf_ptr):
        Bn = self.n // self.bn
        Bk = self.k // self.bk
        Bm = self.m // self.bm

        vm = self.generator.ceil_div(self.bm, self.v_size)

        n_overhead = self.n % self.bn
        k_overhead = self.k % self.bk
        m_overhead = self.m % self.bm
        vm_overhead = -(m_overhead // -self.v_size)
        
        if n_overhead > 0:
            Bn += 1
        if k_overhead > 0:
            Bk += 1
        if m_overhead > 0:
            Bm += 1

        argsA = [Bm, m_overhead > 0, self.unroll_m, self.loop_regs[0], [self.A], [A_ptr], ['down']]
        argsB = [Bn, n_overhead > 0, self.unroll_n, self.loop_regs[1], [self.B], [B_ptr], ['right']]

        if self.unroll_n and not self.unroll_m:
            # swap loops
            outerArgs, innerArgs = (argsB, argsA)
            dirC, dirC2 = ('down', 'right')
            args = lambda i,j: (j,i)
        else:
            outerArgs, innerArgs = (argsA, argsB)
            dirC, dirC2 = ('right', 'down')
            args = lambda i,j: (i,j)
        
        unroll_k = self.unroll_m | self.unroll_n

        def outerLoop(asm, i):
            def innerLoop(asm, j):
                Bmi, Bni = args(i, j)
                self.microkernel(asm, Bmi, Bni, unroll_k, A_ptr, B_ptr, C_ptr, C_pf_ptr)
                if j < innerArgs[0] - 1:
                    move_C, _ = self.C.move(C_ptr, Coords(**{dirC:1}))
                    asm.add(move_C)
                    if self.C_pf:
                        move_C_pf, _ = self.C_pf.move(C_pf_ptr, Coords(**{dirC:1}))
                        asm.add(move_C_pf)
            overhead = self.loopwrap(asm, innerLoop, *innerArgs)
            moveLength = 1-innerArgs[0] if overhead else -innerArgs[0]
            asm.add(self.C.move(C_ptr, Coords(**{dirC2:1, dirC:moveLength}))[0])
            if self.C_pf:
                asm.add(self.C_pf.move(C_pf_ptr, Coords(**{dirC2:1, dirC:moveLength}))[0])

        self.loopwrap(asm, outerLoop, *outerArgs)

    def loopwrap(self, asm, inner, length, overhead, unroll, loopreg, matrices, ptrs, directions, loopunroll=1, overlap=False):
        if unroll:
            for i in range(length):
                inner(asm, i)
            return True
        else:
            def makeMove(dist):
                asm = block(f"move by {dist}")
                for matrix, ptr, direction in zip(matrices, ptrs, directions):
                    asm.add(matrix.move(ptr, Coords(**{direction:dist}))[0])
                return asm
            def makeLoop(until):
                loopblock = block("kernel")
                inner(loopblock, 0)
                loopblock.add(makeMove(1))
                return loop(loopreg, until, unroll=loopunroll, overlap=overlap).body(loopblock)
            if length == 1:
                inner(asm, 0)
                return True
            elif overhead:
                if length > 1:
                    asm.add(makeLoop(length - 1))
                inner(asm, length - 1)
                asm.add(makeMove(1-length))
                return True
            else:
                asm.add(makeLoop(length))
                asm.add(makeMove(-length))
                return False

    def make(self):
        A_ptr = self.A.start()
        B_ptr = self.B.start()
        C_ptr = self.C.start()
        C_pf_ptr = self.C_pf.start() if self.C_pf else None

        asm = block("kernel")

        asm.add(self.generator.make_argument_load(self.starting_regs, self.C_pf is not None))

        asm.add(block("header",
            self.generator.make_scaling_offsets(self.additional_regs, self.bnnz),
            self.generator.init_mask(self.m, self.bm, self.v_size, self.loop_regs[0], self.mask_regs)
        ))

        asm.add(self.generator.init_block(self.v_size))

        self.blockloop(asm, A_ptr, B_ptr, C_ptr, C_pf_ptr)

        assignVirtualRegisters(asm, [self.A_pool, self.B_pool, self.C_pool])

        return asm
        # return block("", *prune(sched.moveLoads(list(asm.normalize()))))
