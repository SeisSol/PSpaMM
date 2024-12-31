from pspamm.cursors import *

from pspamm.codegen.architectures.rvv.operands import *
from pspamm.codegen.ast import *
from pspamm.codegen.sugar import *
from pspamm.codegen.generator import *
from pspamm.codegen.precision import *


class Generator(AbstractGenerator):
    template = """
void {funcName} (const {real_type}* A, const {real_type}* B, {real_type}* C, const {real_type} alpha, const {real_type} beta, const {real_type}* prefetch) {{{{
  __asm__ __volatile__(
    {body_text}
    : : {args} : {clobbered});
    
    #ifndef NDEBUG
    #ifdef _OPENMP
    #pragma omp atomic
    #endif
    pspamm_num_total_flops += {flop};
    #endif
}}}};
"""

    prefetch_reg = None
    prefetch_count = 0
    is_sparse = False
    v_len = 1 # vector register length: v_len * 128 bit
    predicates = {}

    def get_v_size(self):
        return (16 // self.precision.size()) * self.v_len

    def get_precision(self):
        return self.precision

    def get_template(self):
        return self.template
    
    def use_broadcast(self):
        return False

    def has_masks(self):
        return True

    def pred_n_trues(self, num_trues: int, v_size: int, suffix: str = None) -> Register_RV:
        return None

    # is called at most one time in matmul.py
    def set_sparse(self):
        self.is_sparse = True
    
    def make_argument_load(self, starting_regs, prefetch):
        asm = block("Load arguments")
        asm.add(ld(InputOperand(f'0', 'm', 'A'), starting_regs[0], False))
        asm.add(ld(InputOperand(f'1', 'm', 'B'), starting_regs[1], False))
        asm.add(ld(InputOperand(f'2', 'm', 'C'), starting_regs[2], False))
        asm.add(ld(InputOperand(f'3', 'm', 'alpha'), starting_regs[3], False))
        asm.add(ld(InputOperand(f'4', 'm', 'beta'), starting_regs[4], False))
        if prefetch:
            asm.add(ld(InputOperand(f'5', 'm', 'prefetch'), starting_regs[5], False))
        return asm

    def make_reg_blocks(self, bm: int, bn: int, bk: int, v_size: int, nnz: int, m: int, n: int, k: int, prefetch: str):
        vm = self.ceil_div(bm, v_size)                  # vm can be 0 if bm < v_size -> makes ceil_div necessary
        
        assert bn * bk + 2 <= 32
        assert (bn + bk) * vm <= 32

        prec = {
            Precision.DOUBLE: "d",
            Precision.SINGLE: "s",
            Precision.HALF: "h",
            Precision.BFLOAT16: "h",
        }[self.get_precision()]

        A_regs = Matrix([[v(vm * c + r) for c in range(bk)] for r in range(vm)])
        B_regs = Matrix([[f(bn * r + c + 2) for c in range(bn)] for r in range(bk)])
        C_regs = Matrix([[v(32 - vm * bn + vm * c + r) for c in range(bn)] for r in range(vm)])

        b_reg = 0
        alpha_reg = [f(0), f(0)]
        beta_reg = [f(1), f(1)]

        starting_regs = [x(10), x(11), x(12), f(0), f(1)]

        additional_regs = [x(13), x(14), x(15), x(16), x(17), x(31), x(6), x(7), x(5)]

        loop_regs = [x(28), x(29), x(30)]

        mask_regs = []

        return A_regs, B_regs, C_regs, starting_regs, alpha_reg, beta_reg, loop_regs, additional_regs, mask_regs, False

    def bcst_alpha_beta(self,
                        alpha_reg: Register,
                        beta_reg: Register,
                        ) -> Block:

        asm = block("Broadcast alpha and beta when necessary")
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

    def init_mask(self,
                        m: int,
                        bm: int,
                        v_size: int,
                        tempreg,
                        maskreg
                        ) -> Block:

        asm = block("No register based scaling")
        return asm
    
    def init_block(self, size):
        return rvsetvl(x(0), size)

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
                            load_offset: int = 0,
                            is_B: bool = False
                            ) -> Block:

        rows, cols = registers.shape
        action = "Store" if store else "Load"
        asm = block("{} {} register block @ {}".format(action, cursor.name, block_offset))
        prec = self.get_precision()

        # Determine whether we use prefetching and if we are currently operating on C
        do_prefetch = self.prefetch_reg is not None and cursor.name == "C" and store

        b_row, b_col, i, _ = cursor.get_block(cursor_ptr, block_offset)

        cur11 = 0
        #TODO: figure out appropriate threshold (the 16 // self.v_len may still not be optimal; especially if 16 % self.v_len != 0, e.g. 384 bit)
        threshold = 1 if self.is_sparse else (16 // self.v_len)  # uses whole 256 byte cache line, as one SVE-512 vector = 64 bytes

        # DONE if another CPU implements SVE at VL != 64 bytes, rewrite mul_vl (maybe do this dynamically)
        mul_vl = 16 * self.v_len   # e.g. A64FX has VL of 64 bytes in memory (thus, use v_len==4)
        max_mem_ins_mult = 7  # A64FX allows a maximum positive offset of 7 in memory instructions, e.g. ld1d z1.d, p0/z, [x0, 7, MUL VL] (TODO: tune, if ever different)
        max_offset = mul_vl * max_mem_ins_mult  # ld1d/st1d instruction encodes the immediate offset using 4 bits, multiplies it with MUL VL

        prev_disp = 0
        prev_overhead = True
        prev_base = None

        process_size = min(v_size, cursor.br)

        for ic in range(cols):
            for ir in range(rows):
                if (mask is None) or (mask[ir, ic]):
                    all_coords = [Coords(down=ir*v_size+i,right=ic) for i in range(process_size)]
                    has_nonzero = [cursor.has_nonzero_cell(cursor_ptr, block_offset, offset) for offset in all_coords]
                    if not any(has_nonzero):
                        continue
                    elif any(has_nonzero) and not all(has_nonzero) and not is_B:
                        raise NotImplementedError("Element-wise sparsity in A is not yet implemented.")

                    processed = ir * process_size
                    if processed >= b_row:
                        continue
                    p = self.pred_n_trues(min(b_row - processed, process_size), v_size) if not is_B else self.pred_n_trues(process_size, v_size)
                    p_zeroing = self.pred_n_trues(min(b_row - processed, process_size), v_size, "z") if not is_B else self.pred_n_trues(process_size, v_size, "z")
                    cell_offset = Coords(down=ir * v_size, right=ic)

                    # addr = base "pointer" + relative offset in bytes
                    addr, comment = cursor.look(cursor_ptr, block_offset, cell_offset)
                    addr.disp += self.precision.size() * load_offset

                    offset = addr.disp - prev_disp

                    # count how many elements we have processed between last step and this step
                    cont_counter = (offset // mul_vl)
                    larger_max_offset = cont_counter > max_mem_ins_mult
                    non_dividing_offset = offset % mul_vl != 0

                    if larger_max_offset or (prev_overhead and addr.disp > 0) or non_dividing_offset:
                        offset_comment = f"disp > {max_offset}" if larger_max_offset else ("disp % VL != 0" if non_dividing_offset else "previous mem. instr. used p0")
                        asm.add(add(addr.disp, additional_regs[0], offset_comment, addr.base))
                        prev_disp = addr.disp
                        addr.base = additional_regs[0]
                        prev_base = addr.base

                    # adjust addr.disp to a multiple of the RVV vector length
                    if prev_base is None:
                        prev_base = addr.base
                    
                    addr.base = prev_base
                    addr.disp = (addr.disp - prev_disp) // mul_vl

                    if store:
                        asm.add(st(registers[ir, ic], addr, True, comment, pred=p, scalar_offs=False,
                                   add_reg=additional_regs[2]))
                        # perform prefetching after a store instruction, similar to KNL case
                        if do_prefetch and self.prefetch_count % threshold == 0:
                            if prev_disp > 0:
                                asm.add(add(prev_disp, additional_regs[3], "increment the prefetch register", self.prefetch_reg))
                            asm.add(prefetch(mem(additional_regs[3] if prev_disp > 0 else self.prefetch_reg, addr.disp),
                                             "", p, prec, access_type="ST"))
                            self.prefetch_count = 0
                        self.prefetch_count += 1
                    else:
                        asm.add(ld(addr, registers[ir, ic], True, comment, pred=p_zeroing, is_B=is_B, scalar_offs=False,
                                   add_reg=additional_regs[2]))

                    prev_overhead = p is None or int(p.ugly[1]) == 0  # determine if we previously used p0 (overhead predicate)

        return asm

    def make_zero_block(self, registers: Matrix[Register], additional_regs) -> Block:

        rows, cols = registers.shape
        asm = block("zero registers")

        for ic in range(cols):
            for ir in range(rows):
                asm.add(mov(0, registers[ir, ic], True))

        return asm

    def make_microkernel(self,
                         A: Cursor,
                         B: Cursor,
                         A_ptr: CursorLocation,
                         B_ptr: CursorLocation,
                         A_regs: Matrix[Register],
                         B_regs,
                         C_regs: Matrix[Register],
                         v_size: int,
                         additional_regs,
                         to_A_block: Coords = Coords(),
                         to_B_block: Coords = Coords(),
                         sub: bool = False
                         ) -> Block:

        """ make_microkernel generates a GEMM microkernel for two blocks using the outer-product formulation.
          It is responsible for loading and unloading the A block,
          It does not assume that the A or B cursors point to the start of the block.
          Instead, the coordinates to the start of the block are passed separately.
          It does not modify any cursor pointers.
        """

        asm = block("Block GEMM microkernel")
        """block_row, block_col, (start)index, pattern_matrix (true/false)"""
        bm, bk, aidx, apattern = A.get_block(A_ptr, to_A_block)
        bk, bn, bidx, bpattern = B.get_block(B_ptr, to_B_block)

        # tell sparse_mask() that we use sve
        mask = sparse_mask(A_regs, A, A_ptr, to_A_block, B, B_ptr, to_B_block, v_size, True)
        asm.add(self.move_register_block(A, A_ptr, to_A_block, A_regs, v_size, additional_regs, mask, store=False))

        # x = 0;
        bs = []
        cur11 = -10000
        Vm = max(self.ceil_div(bm, v_size), 1)

        multiple = self.precision.size()
        # for ld1rw (single prec): immediate offset is multiple of 4 in range of 0 to 252
        # for ld1rd (double prec): immediate offset is multiple of 8 in range of 0 to 504
        # in both cases: instruction encodes the immediate offset within 6 bits
        max_offs = 2047

        for Vmi in range(Vm):
            # set to all v_size predicates to true, we want to replicate a B element into a whole vector
            for bni in range(bn):  # inside this n-block
                for bki in range(bk):  # inside this k-block
                    to_bcell = Coords(down=bki, right=bni)
                    to_acell = Coords(down=Vmi*v_size, right=bki)
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_bcell) and A.has_nonzero_cell(A_ptr, to_A_block, to_acell):
                        B_cell_addr, B_comment = B.look(B_ptr, to_B_block, to_bcell)
                        if B_regs[bki, bni] not in bs:

                            # max_offs is the maximum allowed immediate offset when using ld1rd/ld1rw to broadcast a scalar value
                            if B_cell_addr.disp > max_offs:
                                moved = B_cell_addr.disp - cur11
                                if moved > 0 and moved <= max_offs:
                                    B_cell_addr.disp = moved
                                else:
                                    asm.add(add(B_cell_addr.disp, additional_regs[0], "", B_cell_addr.base))
                                    cur11 = B_cell_addr.disp
                                    B_cell_addr.disp = 0

                                B_cell_addr.base = additional_regs[0]
                            
                            asm.add(ld(B_cell_addr, B_regs[bki, bni], False, B_comment, pred=None, is_B=True))
                            bs.append(B_regs[bki, bni])

        for Vmi in range(Vm):
            p_merging = self.pred_n_trues(bm - Vmi * v_size, v_size, "m")
            end_index = bm if Vmi + 1 == Vm else Vmi * v_size + v_size  # end_index helps us print the right index ranges
            for bki in range(bk):  # inside this k-block
                for bni in range(bn):  # inside this n-block
                    to_bcell = Coords(down=bki, right=bni)
                    to_acell = Coords(down=Vmi*v_size, right=bki)
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_bcell) and A.has_nonzero_cell(A_ptr, to_A_block, to_acell):
                        B_cell_addr, B_comment = B.look(B_ptr, to_B_block, to_bcell)
                        comment = "C[{}:{},{}] += A[{}:{},{}]*{}".format(Vmi * v_size, end_index, bni, Vmi * v_size,
                                                                         end_index, bki, B_comment)
                        
                        asm.add(fma(B_regs[bki, bni], A_regs[Vmi, bki], C_regs[Vmi, bni], comment=comment, pred=p_merging, bcast=True, sub=sub))
        return asm
