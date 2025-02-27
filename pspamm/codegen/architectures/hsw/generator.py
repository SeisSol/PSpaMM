from pspamm.cursors import *

from pspamm.codegen.architectures.hsw.operands import *
from pspamm.codegen.ast import *
from pspamm.codegen.sugar import *
from pspamm.codegen.generator import *
from pspamm.codegen.precision import *
from pspamm.codegen.regcache import *

class Generator(AbstractGenerator):
    template = """
void {funcName} (const {real_type}* A, const {real_type}* B, {real_type}* C, {real_type} alpha, {real_type} beta, {real_type} const* prefetch) {{
  {real_type}* alpha_p = &alpha;
  {real_type}* beta_p = &beta;
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

    def use_broadcast(self):
        return True

    def has_masks(self):
        return False

    def init_mask(self, m, bm, v_size, tempreg, maskregs):
        return block("")
    
    def scale_base(self):
        return 256

    def pred_n_trues(self, count, v_size, mode):
        # hacked in right now: we set a number as predicate if we need it
        if count < v_size:
            return (1 << count) - 1
        else:
            return None

    def make_argument_load(self, starting_regs, prefetch):
        asm = block("Load arguments")
        asm.add(mov(InputOperand(f'0', 'm', 'A'), starting_regs[0], False))
        asm.add(mov(InputOperand(f'1', 'm', 'B'), starting_regs[1], False))
        asm.add(mov(InputOperand(f'2', 'm', 'C'), starting_regs[2], False))
        asm.add(mov(InputOperand(f'3', 'm', 'alpha_p'), starting_regs[3], False))
        asm.add(mov(InputOperand(f'4', 'm', 'beta_p'), starting_regs[4], False))
        if prefetch:
            asm.add(mov(InputOperand(f'5', 'm', 'prefetch'), starting_regs[5], False))
        return asm

    def make_expand_predicate(self, mask):
        combined = 0
        offset = 0
        for i, value in enumerate(mask):
            if value:
                combined |= offset << (8*i)
                offset += 1
            else:
                combined |= 255 << (8*i)
        return combined

    def make_reg_blocks(self, bm:int, bn:int, bk:int, v_size:int, nnz:int, m:int, n:int, k:int, prefetch: str):
        assert(bm % v_size == 0)
        vm = self.ceil_div(bm, v_size)

        # Needs to fit in AVX/AVX2 ymm registers
        if (bn + bk) * vm + bn * bk <= 16:
            self.preloadA = True
        else:
            self.preloadA = False
            assert(bn * vm + bn * bk + 1 <= 16)

        vmm = {
            1: xmm,
            2: ymm
        }[self.v_len]

        if self.preloadA:
            A_regs = Matrix([[vmm(vm*c + r) for c in range(bk)] for r in range(vm)])
            Aoffset = vm*bk
        else:
            A_regs = Matrix([[vmm(0) for c in range(bk)] for r in range(vm)])
            Aoffset = 1
        
        B_regs = Matrix([[vmm(Aoffset + bn * r + c) for c in range(bn)] for r in range(bk)])
        C_regs = Matrix([[vmm(16 - vm*bn + vm*c + r) for c in range(bn)]
                                                     for r in range(vm)])
        starting_regs = [rdi, rsi, rdx, rbx, rcx]

        b_reg = Aoffset
        alpha_reg = [xmm(b_reg), vmm(b_reg)]
        beta_reg = [xmm(b_reg + 1), vmm(b_reg + 1)]

        additional_regs = [r(9),r(10),r(11),r(15),rax] # ,r(13),r(14)

        prefetch_reg = prefetch == 'BL2viaC'
        if prefetch_reg:
            starting_regs += [r(8)]
        else:
            additional_regs += [r(8)]

        loop_regs = [r(12), r(13), r(14)]

        return A_regs, B_regs, C_regs, starting_regs, alpha_reg, beta_reg, loop_regs, additional_regs, [], prefetch_reg

    def make_scaling_offsets(self,
                         additional_regs: List[Register],
                         nnz: int
                        ) -> Block:

        asm = block("Optimize usage of offsets when accessing B Matrix")

        for i in range(1, len(additional_regs)):
            asm.add(mov(c(self.scale_base() * (2*i - 1)), additional_regs[i], False))
        
        return asm

    def init_block(self, size):
        return block("")

    def reg_based_scaling(self, asm, addr: MemoryAddress, additional_regs: List[Register]):
        halfscale = self.scale_base() // 2
        if addr.disp >= halfscale:
            base = (addr.disp + halfscale) // self.scale_base()
            scaling = 1
            while base % 2 == 0:
                base //= 2
                scaling *= 2
            register = base // 2 + 1

            if register < len(additional_regs) and scaling <= 8:
                addr.index = additional_regs[register]
                addr.scaling = scaling
                addr.disp = ((addr.disp + halfscale) % self.scale_base()) - halfscale

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
                            temp = None
                           ) -> Block:

        rows, cols = registers.shape
        action = "Store" if store else "Load"
        asm = block(f"{action} {cursor.name} register block @ {block_offset}")

        for ic in range(cols):
            for ir in range(rows):
                if (mask is None) or (mask[ir,ic]):
                    all_coords = [Coords(down=ir*v_size+i,right=ic) for i in range(v_size)]
                    has_nonzero = [cursor.has_nonzero_cell(cursor_ptr, block_offset, offset) for offset in all_coords]
                    if all(has_nonzero):
                        cell_offset = all_coords[0]
                        addr, comment = cursor.look(cursor_ptr, block_offset, cell_offset)
                        addr.disp += self.precision.size() * load_offset
                        self.reg_based_scaling(asm, addr, additional_regs)
                        if store:
                            asm.add(mov(registers[ir,ic], addr, True, comment))
                            if prefetching == 'BL2viaC' and pf_cursor is not None:
                                addr, comment = pf_cursor.look(pf_cursor_ptr, block_offset, cell_offset)
                                addr.disp += self.precision.size() * load_offset
                                self.reg_based_scaling(asm, addr, additional_regs)
                                asm.add(prefetch(addr, closeness="L2"))
                        else:
                            asm.add(mov(addr, registers[ir,ic], True, comment))
                    elif any(has_nonzero):
                        raise NotImplementedError("Element-wise sparsity in A is not yet fully implemented.")
                        firsti = 0
                        for i in range(v_size):
                            if has_nonzero[i]:
                                firsti = i
                                break
                        addr, comment = cursor.look(cursor_ptr, block_offset, all_coords[firsti])
                        # assume contiguous memory here

                        asm.add(mov(self.make_expand_predicate(all_coords), additional_regs[0], False))
        return asm

    def move_register_single(self,
                            cursor: Cursor,
                            cursor_ptr: CursorLocation,
                            block_offset: Coords,
                            registers: Matrix[Register],
                            v_size: int,
                            additional_regs,
                            ir,
                            ic,
                            mask: Matrix[bool] = None,
                            store: bool = False,
                            prefetching: str = None,
                            load_offset: int = 0
                           ) -> Block:

        asm = block("")

        if (mask is None) or (mask[ir,ic]):
            cell_offset = Coords(down=ir*v_size, right=ic)
            addr, comment = cursor.look(cursor_ptr, block_offset, cell_offset)
            addr.disp += self.precision.size() * load_offset
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
        bm,bk,aidx,apattern = A.get_block(A_ptr, to_A_block)
        bk,bn,bidx,bpattern = B.get_block(B_ptr, to_B_block)
        assert(bm % v_size == 0)

        mask = sparse_mask(A_regs, A, A_ptr, to_A_block, B, B_ptr, to_B_block, v_size)
        if self.preloadA:
            asm.add(self.move_register_block(A, A_ptr, to_A_block, A_regs, v_size, additional_regs, mask, store=False, temp=B_regs[0,0]))
        else:
            asm.add(self.move_register_single(A, A_ptr, to_A_block, A_regs, v_size, additional_regs, 0, 0, mask, store=False))

        Vm = self.ceil_div(bm, v_size)

        bs = []
        bsv = []
        for Vmi in range(Vm):
            for bni in range(bn):   # inside this n-block
                for bki in range(bk):       # inside this k-block
                    to_bcell = Coords(down=bki, right=bni)
                    to_acell = Coords(down=Vmi*v_size, right=bki)
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_bcell) and A.has_nonzero_cell(A_ptr, to_A_block, to_acell):
                        B_addr, B_comment = B.look(B_ptr, to_B_block, to_bcell)
                        self.reg_based_scaling(asm, B_addr, additional_regs)
                        if B_regs[bki, bni] not in bs:
                            asm.add(bcst(B_addr, B_regs[bki, bni], comment=B_comment))
                            bs.append(B_regs[bki, bni])
                            bsv.append(B_addr)
                        else:
                            # just to make sure we do not use registers differently in a block
                            assert bsv[bs.index(B_regs[bki, bni])].ugly == B_addr.ugly

        for bki in range(bk):       # inside this k-block
            for Vmi in range(Vm):
                if not self.preloadA and not (Vmi, bki) == (0,0):
                    asm.add(self.move_register_single(A, A_ptr, to_A_block, A_regs, v_size, additional_regs, Vmi, bki, mask, store=False))
                for bni in range(bn):   # inside this n-block
                    to_bcell = Coords(down=bki, right=bni)
                    to_acell = Coords(down=Vmi*v_size, right=bki)
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_bcell) and A.has_nonzero_cell(A_ptr, to_A_block, to_acell):
                        _, B_comment = B.look(B_ptr, to_B_block, to_bcell)
                        comment = f"C[{Vmi*v_size}:{Vmi*v_size+v_size},{bni}] += A[{Vmi*v_size}:{Vmi*v_size+v_size},{bki}]*{B_comment}"
                        asm.add(fma(B_regs[bki, bni], A_regs[Vmi, bki], C_regs[Vmi, bni], comment=comment, bcast=None, sub=sub))
        return asm
