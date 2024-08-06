from pspamm.cursors import *

from pspamm.codegen.architectures.arm.operands import *
from pspamm.codegen.ast import *
from pspamm.codegen.sugar import *
from pspamm.codegen.generator import *
from pspamm.codegen.precision import *


class Generator(AbstractGenerator):

    template = """
void {funcName} (const {real_type}* A, const {real_type}* B, {real_type}* C, {real_type} alpha, {real_type} beta, const {real_type}* prefetch) {{{{
  __asm__ __volatile__(
    "ldr x0, %0\\n\\t"
    "ldr x1, %1\\n\\t"
    "ldr x2, %2\\n\\t"
    "ldr x3, %3\\n\\t"
    "ldr x4, %4\\n\\t"
    {body_text}

    : : "m"(A), "m"(B), "m"(C), "m"(alpha), "m"(beta) : {clobbered});
    
    #ifndef NDEBUG
    #ifdef _OPENMP
    #pragma omp atomic
    #endif
    pspamm_num_total_flops += {flop};
    #endif

}}}};
"""

    def get_v_size(self):
        if self.precision == Precision.DOUBLE:
          return 2
        raise NotImplementedError

    def get_template(self):
        return Generator.template

    def use_broadcast(self):
        return True

    def has_masks(self):
        return False
    
    def init_mask(self, bm, v_size, tempreg, maskregs):
        return block("")

    def make_reg_blocks(self, bm:int, bn:int, bk:int, v_size:int, nnz:int, m:int, n:int, k:int):
        assert(bm % v_size == 0)
        vm = bm//v_size
        assert((bn+bk) * vm + bn * bk <= 32)  # Needs to fit in NEON v registers

        prec = {
            Precision.DOUBLE: "2D",
            Precision.SINGLE: "4S",
            Precision.HALF: "8H",
        }[self.get_precision()]

        A_regs = Matrix([[v(vm*c + r, prec) for c in range(bk)] for r in range(vm)])
        B_regs = Matrix([[v(vm*bk + bn * r + c, prec) for c in range(bn)] for r in range(bk)])
        C_regs = Matrix([[v(32 - vm*bn + vm*c + r, prec) for c in range(bn)]
                                                   for r in range(vm)])

        # get vector register number of the first vector in B_regs
        b_reg = vm*bk
        alpha_reg = [v(b_reg, prec), v(b_reg, prec)]
        beta_reg = [v(b_reg + 1, prec), v(b_reg + 1, prec)]


        starting_regs = [r(0), r(1), r(2), r(3), r(4)]

        additional_regs = [r(11), xzr]

        loop_reg = r(12)

        return A_regs, B_regs, C_regs, starting_regs, alpha_reg, beta_reg, loop_reg, additional_regs, []


    def bcst_alpha_beta(self,
                        alpha_reg: Register,
                        beta_reg: Register,
                        ) -> Block:

        asm = block("Broadcast alpha and beta so that efficient multiplication is possible")
        # asm.add(mov(alpha_reg[0], alpha_reg[1], True))
        # asm.add(mov(beta_reg[0], beta_reg[1], True))
        return asm

    def make_scaling_offsets(self,
                         additional_regs: List[Register],
                         nnz: int
                        ) -> Block:

        asm = block("No register based scaling")
        return asm

    def make_b_pointers(self,
                         B_reg: Register,
                         additional_regs: List[Register],
                         nnz: int
                        ) -> Block:

        asm = block("No register based scaling")
        return asm

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
        asm = block("{} {} register block @ {}".format(action, cursor.name, block_offset))

        cur11 = -1000
        skipflag = False
        for ic in range(cols):
            for ir in range(rows):
                if skipflag:
                    skipflag = False
                    continue
                if (mask is None) or (mask[ir,ic]):
                    cell_offset = Coords(down=ir*v_size, right=ic)
                    addr, comment = cursor.look(cursor_ptr, block_offset, cell_offset)
                    addr.disp += 8 * load_offset
                    next_offset = [0, 0]
                    if ir+1 < rows:
                        next_offset = [1, 0]
                    elif ic +1 < rows:
                        next_offset = [0, 1]

                    addr_next, comment_next = cursor.look(cursor_ptr, block_offset, Coords(down=(ir+next_offset[0])*v_size, right=ic+next_offset[1]))
                    addr_next.disp += 8 * load_offset
                    if addr_next.disp == addr.disp + 8 * v_size:
                        skipflag = True
                    if addr.disp > 255:
                        if(addr.disp - cur11 > 0 and addr.disp - cur11 < 256):
                            addr.disp = addr.disp - cur11
                        else:
                            asm.add(add(addr.disp, additional_regs[0], "", addr.base))
                            cur11 = addr.disp
                            addr.disp = 0
                        addr.base = additional_regs[0]
                
                if not skipflag:
                    if store:
                        asm.add(st(registers[ir,ic], addr, True, comment))
                    else:
                        asm.add(ld(addr, registers[ir,ic], True, comment))
                else:
                    if store:
                        asm.add(st(registers[ir,ic], addr, True, comment, registers[ir+next_offset[0],ic+next_offset[1]]))
                    else:
                        asm.add(ld(addr, registers[ir,ic], True, comment, registers[ir+next_offset[0],ic+next_offset[1]]))

        return asm


    def make_zero_block(self, registers: Matrix[Register], additional_regs) -> Block:

        rows, cols = registers.shape
        asm = block("zero registers")

        for ic in range(cols):
          for ir in range(rows):
              asm.add(mov(additional_regs[1], registers[ir,ic], True))

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
        assert(bm % v_size == 0)

        mask = sparse_mask(A_regs, A, A_ptr, to_A_block, B, B_ptr, to_B_block, v_size)
        asm.add(self.move_register_block(A, A_ptr, to_A_block, A_regs, v_size, additional_regs, mask, store=False))

        x = 0;
        bs = []
        cur11 = -1000
        for Vmi in range(bm//v_size):
            for bki in range(bk):       # inside this k-block
                for bni in range(bn):   # inside this n-block
                    to_cell = Coords(down=bki, right=bni)
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_cell):
                        B_cell_addr, B_comment = B.look(B_ptr, to_B_block, to_cell)
                        if B_regs[bki, bni] not in bs:
                            if B_cell_addr.disp > 255:
                                if(B_cell_addr.disp - cur11 > 0 and B_cell_addr.disp - cur11 < 256):
                                    B_cell_addr.disp = B_cell_addr.disp - cur11
                                else:
                                    asm.add(add(B_cell_addr.disp, additional_regs[0], "", B_cell_addr.base))
                                    cur11 = B_cell_addr.disp
                                    B_cell_addr.disp = 0

                                B_cell_addr.base = additional_regs[0]
                  
                            asm.add(ld(B_cell_addr, B_regs[bki, bni], True, B_comment))
                            bs.append(B_regs[bki, bni])

        for Vmi in range(bm//v_size):
            for bki in range(bk):       # inside this k-block
                for bni in range(bn):   # inside this n-block
                    to_cell = Coords(down=bki, right=bni)
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_cell):
                        B_cell_addr, B_comment = B.look(B_ptr, to_B_block, to_cell)
                        comment = "C[{}:{},{}] += A[{}:{},{}]*{}".format(Vmi*v_size, Vmi*v_size+v_size, bni, Vmi*v_size, Vmi*v_size+v_size, bki, B_comment)
                        asm.add(fma(B_regs[bki, bni], A_regs[Vmi, bki], C_regs[Vmi, bni], comment=comment))
        return asm


    def init_prefetching(self, prefetching):            
        Generator.template = Generator.template.format(prefetching_mov = "", prefetching_decl = '',
            funcName="{funcName}", body_text="{body_text}",
            clobbered="{clobbered}", real_type="{real_type}",
            init_registers="{init_registers}", flop="{flop}")
