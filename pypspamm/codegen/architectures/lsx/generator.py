from pypspamm.codegen.address import ScratchBase
from pypspamm.codegen.architectures.lsx.operands import *
from pypspamm.codegen.ast import *
from pypspamm.codegen.generator import *
from pypspamm.codegen.microkernel import LoadedOnce, cells
from pypspamm.codegen.precision import *
from pypspamm.codegen.regcache import *
from pypspamm.codegen.sugar import *
from pypspamm.codegen.target import TARGETS
from pypspamm.cursors import *


class Generator(AbstractGenerator):
    target = TARGETS["lsx"]

    template = """
void {funcName} (const {real_type}* A, const {real_type}* B, {real_type}* C, {real_type} alpha, {real_type} beta, {real_type} const* prefetch) {{
  __asm__ __volatile__(
{body_text}
    : : {args} : {clobbered});

    #ifndef NDEBUG
    #ifdef _OPENMP
    #pragma omp atomic
    #endif
    pspamm_num_total_flops += {flop};
    #endif
}}
"""
    v_len = 2

    def get_v_size(self):
        return (16 // self.precision.size()) * self.v_len

    def get_template(self):
        return Generator.template

    def init_mask(self, m, bm, v_size, tempreg, maskregs):
        return block("")

    def make_argument_load(self, starting_regs, prefetch):
        asm = block("Load arguments")
        asm.add(ld(InputOperand(f"0", "m", "A"), starting_regs[0], False))
        asm.add(ld(InputOperand(f"1", "m", "B"), starting_regs[1], False))
        asm.add(ld(InputOperand(f"2", "m", "C"), starting_regs[2], False))
        asm.add(ld(InputOperand(f"3", "m", "alpha"), starting_regs[3], False))
        asm.add(ld(InputOperand(f"4", "m", "beta"), starting_regs[4], False))
        if prefetch:
            asm.add(ld(InputOperand(f"5", "m", "prefetch"), starting_regs[5], False))
        return asm

    def make_reg_blocks(
        self,
        bm: int,
        bn: int,
        bk: int,
        v_size: int,
        nnz: int,
        m: int,
        n: int,
        k: int,
        prefetch: str,
        copies: int = 1,
    ):
        assert copies == 1, "lsx does not lay out a second copy of the operands yet"
        assert bm % v_size == 0
        vm = self.ceil_div(bm, v_size)

        assert self.target.fits(bn, bk, vm)

        vmm = {1: vr, 2: xr}[self.v_len]

        A_regs = Matrix([[vmm(vm * c + r) for c in range(bk)] for r in range(vm)])
        Aoffset = vm * bk

        B_regs = Matrix(
            [[vmm(Aoffset + bn * r + c) for c in range(bn)] for r in range(bk)]
        )
        C_regs = Matrix(
            [[vmm(32 - vm * bn + vm * c + r) for c in range(bn)] for r in range(vm)]
        )

        b_reg = Aoffset
        alpha_reg = [vmm(b_reg)] * 2
        beta_reg = [vmm(b_reg + 1)] * 2

        starting_regs = [r(10), r(11), r(12), r(13), r(14), r(6), r(5)]

        additional_regs = [r(15), r(16), r(17), r(31), r(7)]

        loop_regs = [r(28), r(29), r(30)]

        prefetch_reg = prefetch == "BL2viaC"

        return (
            A_regs,
            B_regs,
            C_regs,
            starting_regs,
            alpha_reg,
            beta_reg,
            loop_regs,
            additional_regs,
            [],
            prefetch_reg,
        )

    def make_scaling_offsets(self, additional_regs: List[Register], nnz: int) -> Block:
        return block("")

    def init_block(self, size):
        return block("")

    def move_register_block(
        self,
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
        temp=None,
    ) -> Block:

        rows, cols = registers.shape
        action = "Store" if store else "Load"
        asm = block(f"{action} {cursor.name} register block @ {block_offset}")

        max_offs, _ = self.target.memory_offset_limit(1)
        base = ScratchBase(
            additional_regs[0],
            immediate=self.target.scalar_immediate,
            prefer_original=self.target.prefer_original_base,
        )

        for ic in range(cols):
            for ir in range(rows):
                if (mask is None) or (mask[ir, ic]):
                    all_coords = [
                        Coords(down=ir * v_size + i, right=ic) for i in range(v_size)
                    ]
                    has_nonzero = [
                        cursor.has_nonzero_cell(cursor_ptr, block_offset, offset)
                        for offset in all_coords
                    ]
                    if all(has_nonzero):
                        cell_offset = all_coords[0]
                        addr, comment = cursor.look(
                            cursor_ptr, block_offset, cell_offset
                        )
                        addr.disp += self.precision.size() * load_offset
                        needsmove = base.place(asm, addr, max_offs, 1)
                        if store:
                            asm.add(st(registers[ir, ic], addr, True, comment))
                            if prefetching == "BL2viaC" and pf_cursor is not None:
                                addr, comment = pf_cursor.look(
                                    pf_cursor_ptr, block_offset, cell_offset
                                )
                                addr.disp += self.precision.size() * load_offset
                                if addr.disp > max_offs:
                                    moved = addr.disp - base.offset
                                    if needsmove:
                                        asm.add(
                                            add(
                                                addr.disp,
                                                additional_regs[3],
                                                "",
                                                addr.base,
                                            )
                                        )
                                        addr.disp = 0
                                    else:
                                        addr.disp = moved
                                    addr.base = additional_regs[3]
                                asm.add(prefetch(addr, closeness="L2"))
                        else:
                            asm.add(ld(addr, registers[ir, ic], True, comment))
                    elif any(has_nonzero):
                        raise NotImplementedError(
                            "Element-wise sparsity in A is not yet fully implemented."
                        )
        return asm

    def make_zero_block(self, registers: Matrix[Register], additional_regs) -> Block:

        rows, cols = registers.shape
        asm = block("zero registers")

        for ic in range(cols):
            for ir in range(rows):
                asm.add(mov(0, registers[ir, ic], True))

        return asm

    def make_microkernel(
        self,
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
        sub: bool = False,
    ) -> Block:
        """make_microkernel generates a GEMM microkernel for two blocks using the outer-product formulation.
        It is responsible for loading and unloading the A block,
        It does not assume that the A or B cursors point to the start of the block.
        Instead, the coordinates to the start of the block are passed separately.
        It does not modify any cursor pointers.
        """
        asm = block("Block GEMM microkernel")
        bm, bk, aidx, apattern = A.get_block(A_ptr, to_A_block)
        bk, bn, bidx, bpattern = B.get_block(B_ptr, to_B_block)
        assert bm % v_size == 0

        mask = sparse_mask(A_regs, A, A_ptr, to_A_block, B, B_ptr, to_B_block, v_size)
        asm.add(
            self.move_register_block(
                A,
                A_ptr,
                to_A_block,
                A_regs,
                v_size,
                additional_regs,
                mask,
                store=False,
                temp=B_regs[0, 0],
            )
        )

        Vm = self.ceil_div(bm, v_size)
        base = ScratchBase(
            additional_regs[0],
            immediate=self.target.scalar_immediate,
            prefer_original=self.target.prefer_original_base,
        )
        max_offs, _ = self.target.memory_offset_limit(1)

        loaded = LoadedOnce()
        for Vmi, bni, bki, to_acell, to_bcell in cells(
            A, B, A_ptr, B_ptr, to_A_block, to_B_block, Vm, bn, bk, v_size
        ):
            B_cell_addr, B_comment = B.look(B_ptr, to_B_block, to_bcell)
            if loaded.first(B_regs[bki, bni], B_cell_addr):
                # max_offs is the maximum allowed immediate offset when using ld1rd/ld1rw to broadcast a scalar value
                base.place(asm, B_cell_addr, max_offs, 1)

                asm.add(bcst(B_cell_addr, B_regs[bki, bni], B_comment))

        for bki in range(bk):  # inside this k-block
            for Vmi in range(Vm):
                for bni in range(bn):  # inside this n-block
                    to_bcell = Coords(down=bki, right=bni)
                    to_acell = Coords(down=Vmi * v_size, right=bki)
                    if B.has_nonzero_cell(
                        B_ptr, to_B_block, to_bcell
                    ) and A.has_nonzero_cell(A_ptr, to_A_block, to_acell):
                        _, B_comment = B.look(B_ptr, to_B_block, to_bcell)
                        comment = f"C[{Vmi*v_size}:{Vmi*v_size+v_size},{bni}] += A[{Vmi*v_size}:{Vmi*v_size+v_size},{bki}]*{B_comment}"
                        asm.add(
                            fma(
                                B_regs[bki, bni],
                                A_regs[Vmi, bki],
                                C_regs[Vmi, bni],
                                comment=comment,
                                bcast=None,
                                sub=sub,
                            )
                        )
        return asm
