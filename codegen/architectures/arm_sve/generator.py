from cursors import *

from codegen.architectures.arm_sve.operands import *
from codegen.ast import *
from codegen.sugar import *
from codegen.generator import *
from codegen.precision import *


class Generator(AbstractGenerator):
    template = """
void {funcName} (const {real_type}* A, const {real_type}* B, {real_type}* C, {real_type} alpha, {real_type} beta, const {real_type}* prefetch) {{{{
  __asm__ __inline__ __volatile__(
    "ldr x0, %0\\n\\t"
    "ldr x1, %1\\n\\t"
    "ldr x2, %2\\n\\t"
    "ldr x3, %3\\n\\t"
    "ldr x4, %4\\n\\t"
    {init_registers}
    {body_text}

    : : "m"(A), "m"(B), "m"(C), "m"(alpha), "m"(beta) : {clobbered});
    
    #ifndef NDEBUG
    #ifdef _OPENMP
    #pragma omp atomic
    #endif
    pspamm_num_total_flops += {flop};
    #endif

}}}};"""

    def get_v_size(self):
        if self.precision == Precision.DOUBLE:
            return 8
        elif self.precision == Precision.SINGLE:
            return 16
        raise NotImplementedError

    def get_precision(self):
        return self.precision

    def get_template(self):
        return self.template

    def pred_n_trues(self, n: int, v_size: int, k: str = None) -> Register_ARM:
        """pred takes n=num of predicate and k=type of predicate,
         i.e. m or z for merging or zeroing use p7 as all-true predicate and p0 as overhead predicate"""
        # https://arxiv.org/pdf/1803.06185.pdf -> arithmetic and control of general memory is restricted to p0-p7
        assert (n > 0)
        # we only use p7 or p0 as predicates
        n = 8 if n >= v_size else 1

        if k is None:
            s = "p{}".format(n - 1)
        else:
            s = "p{}/{}".format(n - 1, k)
        return Register_ARM(AsmType.p64x8, s)

    # taken from https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    def ceil_div(self, n, d):
        return -(n // -d)

    def make_reg_blocks(self, bm: int, bn: int, bk: int, v_size: int, nnz: int, m: int, n: int, k: int):
        vm = self.ceil_div(bm, v_size)                  # can be 0 if bm < v_size
        assert ((bn + bk) * vm + bn * bk + 2 <= 32)     # Needs to fit in SVE z registers
        prec = "d" if self.get_precision() == Precision.DOUBLE else "s"

        # use max(vm, 1) in case bm < v_size, otherwise we get no A_regs/C_regs
        A_regs = Matrix([[z(max(vm, 1) * c + r + 2, prec) for c in range(bk)] for r in range(max(vm, 1))])
        B_regs = Matrix([[z(max(vm, 1) * bk + 2 + bn * r + c, prec) for c in range(bn)] for r in range(bk)])
        C_regs = Matrix(
            [[z(32 - max(vm, 1) * bn + max(vm, 1) * c + r, prec) for c in range(bn)] for r in range(max(vm, 1))])

        alpha_reg = [z(0, prec), z(0, prec)]
        beta_reg = [z(1, prec), z(1, prec)]

        starting_regs = [r(0), r(1), r(2)]

        additional_regs = [r(11), l("0.0"), r(10)]  # r10 used for scaling offsets

        loop_reg = r(12)

        self.init_registers(bm, v_size)

        return A_regs, B_regs, C_regs, starting_regs, alpha_reg, beta_reg, loop_reg, additional_regs

    def bcst_alpha_beta(self,
                        alpha_reg: Register,
                        beta_reg: Register,
                        ) -> Block:

        asm = block("Broadcast alpha and beta so that efficient multiplication is possible")
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

    def init_registers(self,
                       bm: int,
                       v_size: int
                       ) -> None:

        # TODO: why not just modulo operator? bm = bm % v_size should do the trick
        # while bm > v_size:
        #    bm -= v_size  # calc bm overhead
        bm %= v_size

        eol = "\\n\\t"                          # define the "end of line" sequence for easy assembly
        p_suffix = "d" if v_size == 8 else "s"  # determine whether predicate suffix is '.d' or '.s
        gen_reg = "x" if v_size == 8 else "w"   # determine if 'dup' registers are 64 bit or 32 bit

        # https://developer.arm.com/documentation/102374/0101/Registers-in-AArch64---general-purpose-registers
        # Wn adresses the lower 32 bits of Xn
        # generally: Xn/Wn "[...] are two separate ways of looking at the same register"
        # this means we can load alpha/beta into x3/x4 even if they are floats (32 bits)
        dup_alpha = "\"dup z0.{suffix}, {gen_reg}3{eol}\"\n\t"  # define broadcasting of alpha
        dup_beta = "\"dup z1.{suffix}, {gen_reg}4{eol}\"\n\t"   # define broadcasting of beta

        comment = "//p7 denotes the 'all-true' predicate and, if given, p0 denotes the 'bm % v_size' predicate\n\t"
        overhead = "\"ptrue p0.{suffix}, #{overhead}{eol}\"\n\t" if bm != 0 else ""    # define overhead predicate
        all_true = "\"ptrue p7.{suffix}, #{v_size}{eol}\""                             # define all true predicate
        init_registers = (dup_alpha + dup_beta + comment + overhead + all_true).format(suffix=p_suffix,
                                                                                       gen_reg=gen_reg,
                                                                                       v_size=v_size,
                                                                                       overhead=bm,
                                                                                       eol=eol)

        # since .format() doesn't allow partial formatting, we need to re-include the
        # placeholders that are replaced at the end of generating a kernel
        self.template = self.get_template().format(init_registers=init_registers,
                                                   funcName="{funcName}",
                                                   body_text="{body_text}",
                                                   clobbered="{clobbered}",
                                                   flop="{flop}",
                                                   real_type="{real_type}")

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

        b_row, b_col, i, _ = cursor.get_block(cursor_ptr, block_offset)

        cur11 = 0

        # TODO: if another CPU implements SVE at VL != 64 bytes, rewrite mul_vl (maybe do this dynamically)
        mul_vl = 64     # A64FX has VL of 64 bytes in memory
        max_mem_ins_mult = 7
        max_offset = mul_vl * max_mem_ins_mult  # ld1d/st1d instruction encodes the immediate offset within 4 bits, multiplies it with MUL VL

        init_prev_base = True

        prev_disp = 0
        prev_overhead = True

        for ic in range(cols):
            for ir in range(rows):
                if (mask is None) or (mask[ir, ic]):
                    processed = ir * v_size
                    p = pred_n_trues(b_row - processed, v_size) if not is_B else pred_n_trues(v_size, v_size)
                    p_zeroing = pred_n_trues(b_row - processed, v_size, "z") if not is_B else pred_n_trues(v_size,
                                                                                                           v_size, "z")
                    cell_offset = Coords(down=ir * v_size, right=ic)

                    # addr = base "pointer" + relative offset in bytes
                    addr, comment = cursor.look(cursor_ptr, block_offset, cell_offset)
                    # need to initialise prev_base once to a valid addr.base (especially for sparse kernels)
                    if init_prev_base:
                        # TODO: move initialization outside of loop with dummy address
                        #  cursor.look(cursor_ptr, block_offset, Coords(0,0)), test if this works with sparse kernels
                        prev_base = addr.base
                        init_prev_base = False
                    addr.disp += self.precision.value * load_offset
                    # count how many elements we have processed between last step and this step
                    cont_counter = ((addr.disp - prev_disp) // mul_vl)
                    larger_max_offset = cont_counter > max_mem_ins_mult

                    if larger_max_offset or (prev_overhead and addr.disp > 0):
                        offset_comment = "disp > {}".format(max_offset) if larger_max_offset else "previous mem. instr. used p0"
                        asm.add(add(addr.disp, additional_regs[0], offset_comment, addr.base))
                        prev_disp = addr.disp
                        addr.base = additional_regs[0]
                        prev_base = addr.base

                    # adjust addr.disp to a multiple of a SVE vector's length
                    addr.base = prev_base
                    addr.disp = (addr.disp - prev_disp) // mul_vl

                if store:
                    asm.add(st(registers[ir, ic], addr, True, comment, pred=p, scalar_offs=False,
                               add_reg=additional_regs[2]))
                else:
                    asm.add(ld(addr, registers[ir, ic], True, comment, pred=p_zeroing, is_B=is_B, scalar_offs=False,
                               add_reg=additional_regs[2]))

                prev_overhead = int(p.ugly[1]) == 0  # determine if we previously used p0 (overhead predicate)

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
                         is_B: bool = True
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
        mask = sparse_mask(A_regs, A, A_ptr, to_A_block, B, B_ptr, to_B_block, v_size, is_sve=True)
        asm.add(self.move_register_block(A, A_ptr, to_A_block, A_regs, v_size, additional_regs, mask, store=False))

        # x = 0;
        bs = []
        cur11 = -1000
        Vm = max(self.ceil_div(bm, v_size), 1)

        multiple = self.precision.value
        # for ld1rw: imm offset is multiple of 4 in range of 0 to 252
        # for ld1rd: imm offset is multiple of 8 in range of 0 to 504
        # in both cases: instruction encodes the immediate offset within 6 bits
        max_offs = (2 ** 6 - 1) * multiple
        for Vmi in range(Vm):
            # set to all v_size predicates to true, we want to replicate a B element into a whole vector
            p_zeroing = pred_n_trues(v_size, v_size, "z")
            for bki in range(bk):  # inside this k-block
                for bni in range(bn):  # inside this n-block
                    to_cell = Coords(down=bki, right=bni)
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_cell):
                        B_cell_addr, B_comment = B.look(B_ptr, to_B_block, to_cell)
                        if B_regs[bki, bni] not in bs:
                            # 504 is the maximum allowed immediate offset when using ld1rd to broadcast scalar elements
                            if B_cell_addr.disp > max_offs:
                                if B_cell_addr.disp - cur11 > 0 and B_cell_addr.disp - cur11 <= max_offs:
                                    B_cell_addr.disp = B_cell_addr.disp - cur11
                                else:
                                    asm.add(add(B_cell_addr.disp, additional_regs[0], "", B_cell_addr.base))
                                    cur11 = B_cell_addr.disp
                                    B_cell_addr.disp = 0

                                B_cell_addr.base = additional_regs[0]
                            asm.add(ld(B_cell_addr, B_regs[bki, bni], True, B_comment, pred=p_zeroing, is_B=is_B))
                            bs.append(B_regs[bki, bni])

        for Vmi in range(Vm):
            p_merging = pred_n_trues(bm - Vmi * v_size, v_size, "m")
            end_index = bm if Vmi + 1 == Vm else Vmi * v_size + v_size  # end_index helps us print the right index ranges
            for bki in range(bk):  # inside this k-block
                for bni in range(bn):  # inside this n-block
                    to_cell = Coords(down=bki, right=bni)
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_cell):
                        B_cell_addr, B_comment = B.look(B_ptr, to_B_block, to_cell)
                        comment = "C[{}:{},{}] += A[{}:{},{}]*{}".format(Vmi * v_size, end_index, bni, Vmi * v_size,
                                                                         end_index, bki, B_comment)
                        asm.add(fma(B_regs[bki, bni], A_regs[Vmi, bki], C_regs[Vmi, bni], comment=comment, pred=p_merging))
        return asm

    def init_prefetching(self, prefetching):
        Generator.template = Generator.template.format(prefetching_mov="", prefetching_decl='',
                                                       funcName="{funcName}", body_text="{body_text}",
                                                       clobbered="{clobbered}")
