from cursors import *

from codegen.architectures.knl.operands import *
from codegen.ast import *
from codegen.sugar import *
from codegen.generator import *


class Generator(AbstractGenerator):

    template = """
void {{funcName}} (const double* A, const double* B, double* C, double const* prefetch_A, double const* prefetch_B, double const* prefetch_C) {{{{
  __asm__ __volatile__(
    "movq %0, %%rdi\\n\\t"
    "movq %1, %%rsi\\n\\t"
    "movq %2, %%rdx\\n\\t"
{prefetching_mov}
{{body_text}}

    : : "m"(A), "m"(B), "m"(C){prefetching_decl} : {{clobbered}});

    #ifndef NDEBUG
    #ifdef _OPENMP
    #pragma omp atomic
    #endif
    sparsemmgen_num_total_flops += {{flop}};
    #endif

}}}};
"""
    def get_v_size(self):
        return 8

    def get_template(self):
        return Generator.template

    def make_reg_blocks(self, bm:int, bn:int, bk:int, v_size:int, nnz:int):
        assert(bm % v_size == 0)
        vm = bm//v_size
        assert((bn+bk) * vm <= 32)  # Needs to fit in AVX512 zmm registers

        A_regs = Matrix([[zmm(vm*c + r) for c in range(bk)] for r in range(vm)])
        B_regs = []
        C_regs = Matrix([[zmm(32 - vm*bn + vm*c + r) for c in range(bn)]
                                                     for r in range(vm)])

        starting_regs = [rdi, rsi, rdx]

        available_regs = [r(9),r(10),r(11),r(13),r(14),r(15),rax,rbx,rcx]

        additional_regs = [r(8)]

        reg_count = 0

        for i in range(1024, min(nnz * 8, 8000), 2048):
            additional_regs.append(available_regs[reg_count])
            reg_count += 1

        for i in range(8192, min(nnz * 8, 33000), 8192):
            additional_regs.append(available_regs[reg_count])
            reg_count += 1

        loop_reg = r(12)

        return A_regs, B_regs, C_regs, starting_regs, loop_reg, additional_regs

    def make_scaling_offsets(self,
                         additional_regs: List[Register],
                         nnz: int
                        ) -> Block:

        asm = block("Optimize usage of offsets when accessing B Matrix")

        reg_count = 1

        for i in range(1024, min(nnz * 8, 8000), 2048):
            asm.add(mov(c(i), additional_regs[reg_count], False))
            reg_count += 1
        
        return asm

    def make_b_pointers(self,
                         B_reg: Register,
                         additional_regs: List[Register],
                         nnz: int
                        ) -> Block:

        asm = block("Optimize usage of offsets when accessing B Matrix")

        reg_count = 5

        for i in range(8192, min(nnz * 8, 33000), 8192):
            asm.add(lea(B_reg, additional_regs[reg_count], i))
            reg_count += 1
        
        return asm


    def reg_based_scaling(self, addr: MemoryAddress, additional_regs: List[Register]):
        if addr.disp >= 1024 and addr.disp < 32768:
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
                            prefetching: str = None
                           ) -> Block:

        rows, cols = registers.shape
        action = "Store" if store else "Load"
        asm = block("{} {} register block @ {}".format(action,cursor.name,block_offset))

        for ic in range(cols):
            for ir in range(rows):
                if (mask is None) or (mask[ir,ic]):
                    cell_offset = Coords(down=ir*8, right=ic)
                    addr, comment = cursor.look(cursor_ptr, block_offset, cell_offset)
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
        assert(bm % 8 == 0)

        mask = sparse_mask(A_regs, A, A_ptr, to_A_block, B, B_ptr, to_B_block, v_size)
        asm.add(self.move_register_block(A, A_ptr, to_A_block, A_regs, v_size, additional_regs, mask, store=False))


        for Vmi in range(bm//8):
            for bki in range(bk):       # inside this k-block
                for bni in range(bn):   # inside this n-block
                    to_cell = Coords(down=bki, right=bni)
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_cell):
                        B_addr, B_comment = B.look(B_ptr, to_B_block, to_cell)
                        self.reg_based_scaling(B_addr, additional_regs)
                        comment = "C[{}:{},{}] += A[{}:{},{}]*{}".format(Vmi*8,Vmi*8+8,bni,Vmi*8,Vmi*8+8,bki,B_comment)
                        asm.add(fma(B_addr, A_regs[Vmi, bki], C_regs[Vmi, bni], comment=comment))
        return asm

    def init_prefetching(self, prefetching):
        
        if prefetching != 'BL2viaC':
            Generator.template = Generator.template.format(prefetching_mov = "", prefetching_decl = "")    
            return
        
        Generator.template = Generator.template.format(prefetching_mov = '    "movq %3, %%r8\\n\\t"', prefetching_decl = ', "m"(prefetch_B)')
