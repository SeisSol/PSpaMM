from pspamm.cursors import *

from pspamm.codegen.architectures.arm_sve.operands import *
from pspamm.codegen.ast import *
from pspamm.codegen.sugar import *
from pspamm.codegen.generator import *
from pspamm.codegen.precision import *


class Generator(AbstractGenerator):
    template = """
void {funcName} (const {real_type}* A, const {real_type}* B, {real_type}* C, const {real_type} alpha, const {real_type} beta, const {real_type}* prefetch) {{{{
  __asm__ __volatile__(
{init_registers}
{body_text}
    : : {args} : {clobbered});
    
    #ifndef NDEBUG
    #ifdef _OPENMP
    #pragma omp atomic
    #endif
    pspamm_num_total_flops += {flop};
    #endif

}}}}
"""

    prefetch_count = 0
    is_sparse = False
    v_len = 4 # vector register length: v_len * 128 bit
    predicates = {}

    def get_v_size(self):
        return (16 // self.precision.size()) * self.v_len

    def get_precision(self):
        return self.precision

    def get_template(self):
        return self.template
    
    def use_broadcast(self):
        return True

    def has_masks(self):
        return True
    
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

    def pred_n_trues(self, num_trues: int, v_size: int, suffix: str = None) -> Register_ARM:
        """pred takes num_trues=num of true elements and suffix=type of predicate (m or z) for merging or zeroing
         we only use p7 as all-true predicate and p0 as overhead predicate
         e.g. pred_n_trues(n=4, v_size=8, suffix="m") returns the predicate p0/m with the first 4 elements
         set to true"""
        assert (num_trues > 0)
        assert (suffix == "m" or suffix == "z" or suffix is None)

        # we only use p7 or p0 as predicates (1 == p0, 8 == p7)
        index = 7 if num_trues >= v_size else self.predicates[num_trues]

        if suffix is None:
            s = f"p{index}"
        else:
            s = f"p{index}/{suffix}"
        return Register_ARM(AsmType.p64x8, s)

    # is called at most one time in matmul.py
    def set_sparse(self):
        self.is_sparse = True

    def make_reg_blocks(self, bm: int, bn: int, bk: int, v_size: int, nnz: int, m: int, n: int, k: int, prefetch:str):
        vm = self.ceil_div(bm, v_size)                  # vm can be 0 if bm < v_size -> makes ceil_div necessary

        # k-broadcasting only works in 128-bit lanes
        elem128 = 16 // self.get_precision().size()
        vkext = -(bk // -elem128)

        # inline broadcasting is only allowed for the lower-numbered registers
        self.inline_broadcast = False
        if bn*vkext < 16 if self.get_precision().size() == 8 else bn*vkext < 8:
            self.inline_broadcast = True
        if bk == 1:
            self.inline_broadcast = False

        if self.inline_broadcast:
            vk = vkext
        else:
            vk = bk
        
        assert ((bn + bk) * vm + bn * vk <= 32)     # Needs to fit in SVE z registers

        prec = {
            Precision.DOUBLE: "d",
            Precision.SINGLE: "s",
            Precision.HALF: "h",
            Precision.BFLOAT16: "h",
        }[self.get_precision()]

        # make place for the two broadcasting registers
        a_offset = 1 if bn * vk == 1 else 0
        assert ((bn + bk) * vm + bn * vk + a_offset <= 32)

        A_regs = Matrix([[z(vm * c + r + bn * vk + a_offset, prec) for c in range(bk)] for r in range(vm)])
        B_regs = Matrix([[z(bn * r + c, prec) for c in range(bn)] for r in range(vk)])
        C_regs = Matrix([[z(32 - vm * bn + vm * c + r, prec) for c in range(bn)] for r in range(vm)])

        b_reg = 0
        alpha_reg = [z(b_reg, prec), z(b_reg, prec)]
        beta_reg = [z(b_reg + 1, prec), z(b_reg + 1, prec)]

        starting_regs = [r(0), r(1), r(2), r(3), r(4), r(5), r(6), r(11)]  # r6 is needed for predicate creation, r5 is added in init_prefetching()

        additional_regs = [r(8), l("0.0"), r(10), r(6)]  # r10 used for scaling offsets

        loop_regs = [r(12), r(13), r(14)]

        mask_regs = [p(0), p(7)]

        self.init_registers(m, bm, k, bk, v_size, nnz)

        prefetch_reg = prefetch is not None

        return A_regs, B_regs, C_regs, starting_regs, alpha_reg, beta_reg, loop_regs, additional_regs, mask_regs, prefetch_reg

    def make_scaling_offsets(self,
                             additional_regs: List[Register],
                             nnz: int
                             ) -> Block:

        asm = block("No register based scaling")
        return asm

    def init_block(self, size):
        return block("")

    def init_mask(self,
                        m: int,
                        bm: int,
                        v_size: int,
                        tempreg,
                        maskreg
                        ) -> Block:

        asm = block("No register based scaling")
        return asm

    def init_registers(self,
                       m: int,
                       bm: int,
                       k: int,
                       bk: int,
                       v_size: int,
                       nnz: int
                       ) -> None:

        bmmod = bm % v_size
        elem128 = 16 // self.get_precision().size()
        bkmod = bk % elem128 if self.inline_broadcast else 0
        kmod = (k % bk) % elem128 if self.inline_broadcast else 0
        mmod = (m % bm) % v_size

        eol = "\\n\\t"                          # define the "end of line" sequence for easy assembly
        # determine the predicate suffix
        p_suffix = {
            Precision.DOUBLE: "d",
            Precision.SINGLE: "s",
            Precision.HALF: "h",
            Precision.BFLOAT16: "h",
        }[self.get_precision()]
        # determine length of 'dup' registers
        gen_reg = "w" if self.get_precision().size() <= 4 else "x"
        overhead_counter = 6

        comment = "// p7 denotes the 'all-true' predicate\n\t"
        comment += "// if given, p0 denotes the 'bm % v_size' predicate\n\t"
        comment += "// if given, p1 denotes the 'bk % elem128' predicate\n\t"
        comment += "// if given, p2 denotes the 'k % elem128' predicate\n\t"
        comment += "// if given, p4 denotes the 'k % v_size' predicate\n\t"

        self.has_k_overhead = kmod != 0
        self.has_bk_overhead = bkmod != 0
        self.has_nnz_overhead = nnz % elem128 != 0

        # specification for ptrue: https://developer.arm.com/documentation/ddi0596/2021-12/SVE-Instructions/PTRUE--Initialise-predicate-from-named-constraint-
        # search for 'DecodePredCount' for the explanation of how the pattern in 'ptrue p{d}.{suffix}, #pattern' is decoded:
        # https://developer.arm.com/documentation/ddi0596/2020-12/Shared-Pseudocode/AArch64-Functions?lang=en#impl-aarch64.DecodePredCount.2
        # 'ptrue' doesnt work for initialising overhead predicate when using single precision -> see valid patterns from above
        # overhead = "\"ptrue p0.{suffix}, #{overhead}{eol}\"\n\t" if bm != 0 else ""    # define overhead predicate
        overhead_bm = "\"mov {gen_reg}{overhead_counter}, #{overhead_bm}{eol}\"\n\t\"whilelo p0.{suffix}, {gen_reg}zr, {gen_reg}{overhead_counter}{eol}\"\n\t" if bmmod != 0 else ""
        overhead_bk = "\"mov {gen_reg}{overhead_counter}, #{overhead_bk}{eol}\"\n\t\"whilelo p1.{suffix}, {gen_reg}zr, {gen_reg}{overhead_counter}{eol}\"\n\t" if self.has_bk_overhead else ""
        overhead_k = "\"mov {gen_reg}{overhead_counter}, #{overhead_k}{eol}\"\n\t\"whilelo p2.{suffix}, {gen_reg}zr, {gen_reg}{overhead_counter}{eol}\"\n\t" if self.has_k_overhead else ""
        overhead_nnz = "\"mov {gen_reg}{overhead_counter}, #{overhead_nnz}{eol}\"\n\t\"whilelo p3.{suffix}, {gen_reg}zr, {gen_reg}{overhead_counter}{eol}\"\n\t" if self.has_nnz_overhead else ""
        overhead_m = "\"mov {gen_reg}{overhead_counter}, #{overhead_m}{eol}\"\n\t\"whilelo p4.{suffix}, {gen_reg}zr, {gen_reg}{overhead_counter}{eol}\"\n\t" if mmod != 0 else ""
        all_true = "\"ptrue p7.{suffix}, #31{eol}\""                             # define all true predicate
        init_registers = (comment + overhead_bm + overhead_bk + overhead_k + overhead_nnz + overhead_m + all_true).format(suffix=p_suffix,
                                                                                            gen_reg=gen_reg,
                                                                                            overhead_counter=overhead_counter,
                                                                                            v_size=v_size,
                                                                                            overhead_bm=bmmod,
                                                                                            overhead_bk=bkmod,
                                                                                            overhead_k=kmod,
                                                                                            overhead_m=mmod,
                                                                                            overhead_nnz=nnz % elem128,
                                                                                            eol=eol)

        self.predicates[v_size] = 7
        if bmmod != 0: self.predicates[bmmod] = 0
        if bkmod != 0: self.predicates[bkmod] = 1
        if kmod != 0: self.predicates[kmod] = 2
        if mmod != 0: self.predicates[mmod] = 4

        # since .format() doesn't allow partial formatting, we need to re-include the
        # placeholders that are replaced at the end of generating a kernel
        self.template = self.get_template().format(init_registers=init_registers,
                                                   funcName="{funcName}",
                                                   body_text="{body_text}",
                                                   clobbered="{clobbered}",
                                                   flop="{flop}",
                                                   real_type="{real_type}",
                                                   args="{args}")

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
                            pf_cursor: Cursor = None,
                            pf_cursor_ptr: CursorLocation = None,
                            is_B: bool = False
                            ) -> Block:

        rows, cols = registers.shape
        action = "Store" if store else "Load"
        asm = block(f"{action} {cursor.name} register block @ {block_offset}")
        prec = self.get_precision()

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
                    processed = ir * process_size
                    size = min(process_size, b_row - processed)
                    all_coords = [Coords(down=ir*v_size+i,right=ic) for i in range(size)]
                    has_nonzero = [cursor.has_nonzero_cell(cursor_ptr, block_offset, offset) for offset in all_coords]
                    if not any(has_nonzero):
                        continue
                    elif any(has_nonzero) and not all(has_nonzero) and not is_B:
                        raise NotImplementedError("Element-wise sparsity in A is not yet implemented.")

                    p = self.pred_n_trues(size, v_size) if not is_B else self.pred_n_trues(process_size, v_size)
                    p_zeroing = self.pred_n_trues(size, v_size, "z") if not is_B else self.pred_n_trues(process_size, v_size, "z")
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

                    # adjust addr.disp to a multiple of a SVE vector's length
                    if prev_base is None:
                        prev_base = addr.base
                    
                    addr.base = prev_base
                    addr.disp = (addr.disp - prev_disp) // mul_vl

                    if store:
                        asm.add(st(registers[ir, ic], addr, True, comment, pred=p, scalar_offs=False,
                                   add_reg=additional_regs[2]))
                        # perform prefetching after a store instruction, similar to KNL case
                        if prefetching:
                            addr, comment = pf_cursor.look(pf_cursor_ptr, block_offset, cell_offset)
                            addr.disp += self.precision.size() * load_offset
                            if prev_disp > 0:
                                asm.add(add(prev_disp, additional_regs[3], "increment the prefetch register", addr.base))
                            asm.add(prefetch(mem(additional_regs[3] if prev_disp > 0 else addr.base, (addr.disp - prev_disp) // mul_vl),
                                             "", p, prec, access_type="LD", closeness="L2", temporality="KEEP"))
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
                asm.add(mov(additional_regs[1], registers[ir, ic], True))

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
        cur11 = -1000
        Vm = max(self.ceil_div(bm, v_size), 1)

        multiple = self.precision.size()
        # for ld1rw (single prec): immediate offset is multiple of 4 in range of 0 to 252
        # for ld1rd (double prec): immediate offset is multiple of 8 in range of 0 to 504
        # in both cases: instruction encodes the immediate offset within 6 bits
        if not self.inline_broadcast:
            max_offs = (2 ** 6 - 1) * multiple
            divider = 1
            elem128 = 1
            vk = bk
        else:
            max_offs = 127
            divider = 16
            elem128 = 16 // self.get_precision().size()
            vk = -(bk // -elem128)

        preg = self.pred_n_trues(elem128, elem128, 'z')
        preg_last = preg if bk % elem128 == 0 else self.pred_n_trues(bk % elem128, elem128, 'z')
        firstloc = {}
        for Vmi in range(Vm):
            # set to all v_size predicates to true, we want to replicate a B element into a whole vector
            for bni in range(bn):  # inside this n-block
                for bki in range(bk):  # inside this k-block
                    bki_reg = bki // elem128
                    to_bcell = Coords(down=bki, right=bni)
                    to_acell = Coords(down=Vmi*v_size, right=bki)
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_bcell):
                        if (bki_reg, bni) not in firstloc:
                            B_cell_addr, B_comment = B.look(B_ptr, to_B_block, to_bcell)
                            firstloc[(bki_reg, bni)] = (B_cell_addr, B_comment)
                        if A.has_nonzero_cell(A_ptr, to_A_block, to_acell) and B_regs[bki_reg, bni] not in bs:
                            p_zeroing = preg_last if bki_reg + 1 == vk else preg

                            B_cell_addr = firstloc[(bki_reg, bni)][0]
                            B_comment = firstloc[(bki_reg, bni)][1]

                            # max_offs is the maximum allowed immediate offset when using ld1rd/ld1rw to broadcast a scalar value
                            if B_cell_addr.disp > max_offs or B_cell_addr.disp % divider != 0:
                                moved = B_cell_addr.disp - cur11
                                if moved > 0 and moved <= max_offs and moved % divider == 0:
                                    B_cell_addr.disp = moved
                                else:
                                    asm.add(add(B_cell_addr.disp, additional_regs[0], "", B_cell_addr.base))
                                    cur11 = B_cell_addr.disp
                                    B_cell_addr.disp = 0

                                B_cell_addr.base = additional_regs[0]
                            
                            if not self.inline_broadcast:
                                asm.add(ld(B_cell_addr, B_regs[bki_reg, bni], True, B_comment, pred=p_zeroing, is_B=True))
                            else:
                                asm.add(ld(B_cell_addr, B_regs[bki_reg, bni], True, B_comment, pred=p_zeroing, sub128=True))
                            bs.append(B_regs[bki_reg, bni])

        # TODO: refactor cell_indices into the cursors/blocks
        cell_indices = {}
        for bki in range(bk):  # inside this k-block
            for Vmi in range(Vm):
                p_merging = self.pred_n_trues(bm - Vmi * v_size, v_size, "m")
                end_index = bm if Vmi + 1 == Vm else Vmi * v_size + v_size  # end_index helps us print the right index ranges
                for bni in range(bn):  # inside this n-block
                    to_bcell = Coords(down=bki, right=bni)
                    to_acell = Coords(down=Vmi*v_size, right=bki)
                    bki_reg = bki // elem128
                    if (Vmi, bki_reg, bni) not in cell_indices:
                        cell_indices[(Vmi, bki_reg, bni)] = 0
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_bcell) and A.has_nonzero_cell(A_ptr, to_A_block, to_acell):
                        _, B_comment = B.look(B_ptr, to_B_block, to_bcell)
                        comment = f"C[{Vmi * v_size}:{end_index},{bni}] += A[{Vmi * v_size}:{end_index},{bki}]*{B_comment}"
                        
                        if not self.inline_broadcast:
                            bcast = None
                        else:
                            bcast = cell_indices[(Vmi, bki_reg, bni)]
                        asm.add(fma(B_regs[bki_reg, bni], A_regs[Vmi, bki], C_regs[Vmi, bni], comment=comment, pred=p_merging, bcast=bcast, sub=sub))
                    
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_bcell):
                        cell_indices[(Vmi, bki_reg, bni)] += 1
        return asm
