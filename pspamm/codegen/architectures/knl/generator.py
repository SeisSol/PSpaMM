from pspamm.cursors import *

from pspamm.codegen.architectures.knl.operands import *
from pspamm.codegen.ast import *
from pspamm.codegen.sugar import *
from pspamm.codegen.generator import *
from pspamm.codegen.precision import *


class Generator(AbstractGenerator):
    template = """
void {{funcName}} (const {{real_type}}* A, const {{real_type}}* B, {{real_type}}* C, {{real_type}} alpha, {{real_type}} beta, {{real_type}} const* prefetch) {{{{
  {{real_type}}* alpha_p = &alpha;
  {{real_type}}* beta_p = &beta;
  __asm__ __volatile__(
    "movq %0, %%rdi\\n\\t"
    "movq %1, %%rsi\\n\\t"
    "movq %2, %%rdx\\n\\t"
    "movq %3, %%rbx\\n\\t"
    "movq %4, %%rcx\\n\\t"
{prefetching_mov}
{{body_text}}

    : : "m"(A), "m"(B), "m"(C), "m"(alpha_p), "m"(beta_p){prefetching_decl} : {{clobbered}});

    #ifndef NDEBUG
    #ifdef _OPENMP
    #pragma omp atomic
    #endif
    pspamm_num_total_flops += {{flop}};
    #endif

}}}};
"""
    v_len = 4

    def get_v_size(self):
        return (16 // self.precision.size()) * self.v_len

    def get_template(self):
        return Generator.template

    def use_broadcast(self):
        return False

    def has_masks(self):
        return False # for now

    def pred_n_trues(self, count, v_size, mode):
        # a bit hacky at the moment (won't work for all masks)
        if count < v_size and count > 0:
            return mask(0)
        else:
            return None

    def make_reg_blocks(self, bm:int, bn:int, bk:int, v_size:int, nnz:int, m:int, n:int, k:int):
        vm = self.ceil_div(bm, v_size)
        assert((bn+bk) * vm <= 32)  # Needs to fit in AVX512 xmm/ymm/zmm registers

        vmm = {
            1: xmm,
            2: ymm,
            4: zmm
        }[self.v_len]

        A_regs = Matrix([[vmm(vm*c + r) for c in range(bk)] for r in range(vm)])
        B_regs = []
        C_regs = Matrix([[vmm(32 - vm*bn + vm*c + r) for c in range(bn)]
                                                     for r in range(vm)])

        starting_regs = [rdi, rsi, rdx, rbx, rcx]

        alpha_reg = [rbx, rbx]
        beta_reg = [rcx, rcx]

        available_regs = [r(9),r(10),r(11),r(15),rax,r(13),r(14)]

        additional_regs = [r(8)]

        mask_regs = [mask(0)]

        reg_count = 0

        for i in range(1024, min(max(nnz * self.precision.size(), m*k*self.precision.size(), m*n*self.precision.size()),8000), 2048):
            additional_regs.append(available_regs[reg_count])
            reg_count += 1

        for i in range(8192, min(nnz * self.precision.size(), 33000), 8192):
            additional_regs.append(available_regs[reg_count])
            reg_count += 1

        loop_regs = [r(12), r(13), r(14)]

        return A_regs, B_regs, C_regs, starting_regs, alpha_reg, beta_reg, loop_regs, additional_regs, mask_regs

    def init_mask(self, bm, v_size, tempreg, maskregs):
        rest = bm % v_size
        if rest == 0:
            return block("")
        else:
            asm = block("Set mask register")
            restval = (1 << rest) - 1
            asm.add(mov(restval, tempreg, False))
            asm.add(mov(tempreg, maskregs[0], False))
            return asm

    def bcst_alpha_beta(self,
                        alpha_reg: Register,
                        beta_reg: Register,
                        ) -> Block:

        asm = block("Broadcast alpha and beta using inline broadcasting")

#        asm.add(bcst(alpha_reg[0], alpha_reg[1]))
#        asm.add(bcst(beta_reg[0], beta_reg[1]))
        
        return asm

    def make_scaling_offsets(self,
                         additional_regs: List[Register],
                         nnz: int
                        ) -> Block:

        asm = block("Optimize usage of offsets when accessing B Matrix")

        for i in range(1, min(len(additional_regs), 5)):
            asm.add(mov(c(1024 + (i-1) * 2048), additional_regs[i], False))
        
        return asm

    def make_b_pointers(self,
                         B_reg: Register,
                         additional_regs: List[Register],
                         nnz: int
                        ) -> Block:

        asm = block("Optimize usage of offsets when accessing B Matrix")

        reg_count = 5

        for i in range(8192, min(nnz * self.precision.size(), 33000), 8192):
            asm.add(lea(B_reg, additional_regs[reg_count], i))
            reg_count += 1
        
        return asm


    def reg_based_scaling(self, addr: MemoryAddress, additional_regs: List[Register], with_index: bool):
        if addr.disp >= 1024 and ((addr.disp < 32768 and with_index) or addr.disp < 8192):
            scaling_and_register = {
                1: (1, 1),
                2: (2, 1),
                3: (1, 2),
                4: (4, 1),
                5: (1, 3),
                6: (2, 2),
                7: (1, 4)
            }
            if addr.disp % 8192 >= 1024:
                addr.scaling, reg = scaling_and_register[ (addr.disp % 8192) // 1024 ]
                addr.index = additional_regs[reg]

            if addr.disp >= 8192:
                addr.base = additional_regs[addr.disp // 8192 + 4]

            addr.disp = addr.disp % 1024

    def move_register_block(self,
                            cursor: Cursor,
                            cursor_ptr: CursorLocation,
                            block_offset: Coords,
                            registers: Matrix[Register],
                            v_size: int,
                            additional_regs,
                            mask: Matrix[bool] = None,
                            store: bool = False,
                            prefetching: str = None,
                            load_offset: int = 0
                           ) -> Block:

        rows, cols = registers.shape
        action = "Store" if store else "Load"
        asm = block("{} {} register block @ {}".format(action,cursor.name,block_offset))

        for ic in range(cols):
            for ir in range(rows):
                if (mask is None) or (mask[ir,ic]):
                    cell_offset = Coords(down=ir*v_size, right=ic)
                    addr, comment = cursor.look(cursor_ptr, block_offset, cell_offset)
                    addr.disp += self.precision.size() * load_offset
                    if store:
                        asm.add(mov(registers[ir,ic], addr, True, comment))
                        if prefetching == 'BL2viaC':
                            asm.add(prefetch(mem(additional_regs[0], addr.disp)))
                    else:
                        asm.add(mov(addr, registers[ir,ic], True, comment))
        return asm

    def make_zero_block(self, registers: Matrix[Register], additional_regs) -> Block:

        rows, cols = registers.shape
        asm = block("zero registers")

        for ic in range(cols):
            for ir in range(rows):
                asm.add(mov(0, registers[ir,ic], True))

        return asm


    def make_microkernel(self,
                         A: Cursor,
                         B: Cursor,
                         A_ptr: CursorLocation,
                         B_ptr: CursorLocation,
                         A_regs: Matrix[Register],
                         B_regs,
                         C_regs: Matrix[Register],
                         v_size:int,
                         additional_regs,
                         to_A_block: Coords = Coords(),
                         to_B_block: Coords = Coords()
                        ) -> Block:

        """ make_microkernel generates a GEMM microkernel for two blocks using the outer-product formulation.
            It is responsible for loading and unloading the A block,
            It does not assume that the A or B cursors point to the start of the block.
            Instead, the coordinates to the start of the block are passed separately.
            It does not modify any cursor pointers.
        """
        asm = block("Block GEMM microkernel")
        bm,bk,aidx,apattern = A.get_block(A_ptr, to_A_block)
        bk,bn,bidx,bpattern = B.get_block(B_ptr, to_B_block)

        mask = sparse_mask(A_regs, A, A_ptr, to_A_block, B, B_ptr, to_B_block, v_size, True)
        asm.add(self.move_register_block(A, A_ptr, to_A_block, A_regs, v_size, additional_regs, mask, store=False))

        for Vmi in range(bm//v_size):
            for bki in range(bk):       # inside this k-block
                for bni in range(bn):   # inside this n-block
                    to_cell = Coords(down=bki, right=bni)
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_cell):
                        B_addr, B_comment = B.look(B_ptr, to_B_block, to_cell)
                        self.reg_based_scaling(B_addr, additional_regs, True)
                        comment = "C[{}:{},{}] += A[{}:{},{}]*{}".format(Vmi*v_size,Vmi*v_size+v_size,bni,Vmi*v_size,Vmi*v_size+v_size,bki,B_comment)
                        asm.add(fma(B_addr, A_regs[Vmi, bki], C_regs[Vmi, bni], comment=comment))
        return asm

    def init_prefetching(self, prefetching):
        
        if prefetching != 'BL2viaC':
            Generator.template = Generator.template.format(prefetching_mov = "", prefetching_decl = "")    
            return None
        
        prefetchReg = r(8)
        Generator.template = Generator.template.format(prefetching_mov = '    "movq %5, {}\\n\\t"'.format(prefetchReg.ugly), prefetching_decl = ', "m"(prefetch)')
        return prefetchReg
