from pspamm.cursors import *

from pspamm.codegen.architectures.knl.operands import *
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
    v_len = 4
    predicates = {0:kmask(0)}

    def get_v_size(self):
        return (16 // self.precision.size()) * self.v_len

    def get_template(self):
        return Generator.template

    def use_broadcast(self):
        return False

    def has_masks(self):
        return True

    def pred_n_trues(self, count, v_size, mode):
        # a bit hacky at the moment (won't work for all masks)
        if count < v_size:
            return Predicate(self.predicates[count], mode=='z')
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

    def make_reg_blocks(self, bm:int, bn:int, bk:int, v_size:int, nnz:int, m:int, n:int, k:int, prefetch: str):
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

        available_regs = [r(9),r(10),r(11),r(15),rax] # ,r(13),r(14)

        prefetch_reg = prefetch == 'BL2viaC'
        if prefetch_reg:
            starting_regs += [r(8)]
            additional_regs = []
        else:
            additional_regs = [r(8)]

        mask_regs = [kmask(1), kmask(2)]

        reg_count = 0

        self.spontaneous_scaling = False
        for i in range(1024, min(max(nnz * self.precision.size(), m*k*self.precision.size(), m*n*self.precision.size()),8192), 2048):
            additional_regs.append(available_regs[reg_count])
            reg_count += 1

        for i in range(8192, min(nnz * self.precision.size(), 32768), 8192):
            if reg_count == len(available_regs):
                self.spontaneous_scaling = True
                break
            additional_regs.append(available_regs[reg_count])
            reg_count += 1

        loop_regs = [r(12), r(13), r(14)]

        # FIXME: a bit hacky to have the mask setup here
        rest = bm % v_size
        rest2 = (m % bm) % v_size
        self.predicates[rest] = kmask(1)
        self.predicates[rest2] = kmask(2)
        self.predicates[0] = kmask(0)

        return A_regs, B_regs, C_regs, starting_regs, alpha_reg, beta_reg, loop_regs, additional_regs, mask_regs, prefetch_reg

    def init_mask(self, m, bm, v_size, tempreg, maskregs):
        rest = bm % v_size
        rest2 = (m % bm) % v_size
        asm = block("Set mask registers")
        if rest > 0:
            restval = (1 << rest) - 1
            asm.add(mov(restval, tempreg, False))
            asm.add(mov(tempreg, maskregs[0], False))
        if rest2 > 0:
            restval2 = (1 << rest2) - 1
            asm.add(mov(restval2, tempreg, False))
            asm.add(mov(tempreg, maskregs[1], False))
        return asm

    def bcst_alpha_beta(self,
                        alpha_reg: Register,
                        beta_reg: Register,
                        ) -> Block:

        asm = block("Broadcast alpha and beta using inline broadcasting")
        
        return asm

    def make_scaling_offsets(self,
                         additional_regs: List[Register],
                         nnz: int
                        ) -> Block:

        asm = block("Optimize usage of offsets when accessing B Matrix")

        if not self.spontaneous_scaling:
            for i in range(1, min(len(additional_regs), 5)):
                asm.add(mov(c(1024 + (i-1) * 2048), additional_regs[i], False))
        
        return asm

    def make_b_pointers(self,
                         B_reg: Register,
                         additional_regs: List[Register],
                         nnz: int
                        ) -> Block:

        asm = block("Optimize usage of offsets when accessing B Matrix")

        if not self.spontaneous_scaling:
            reg_count = 5

            for i in range(8192, min(nnz * self.precision.size(), 33000), 8192):
                asm.add(lea(B_reg, additional_regs[reg_count], i))
                reg_count += 1
        
        return asm

    def init_block(self, size):
        return block("")

    def reg_based_scaling(self, regcache, asm, addr: MemoryAddress, additional_regs: List[Register], with_index: bool):
        if addr.disp >= 1024:
            if ((addr.disp < 32768 and with_index) or addr.disp < 8192) and not self.spontaneous_scaling:
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

                if addr.disp >= 8192 and not self.spontaneous_scaling:
                    addr.base = additional_regs[addr.disp // 8192 + 4]

                addr.disp = addr.disp % 1024
            else:
                large_offset = addr.disp // 1024

                basereg, load = regcache.get(large_offset)
                if load:
                    asm.add(mov(addr.base, basereg, False))
                    asm.add(add(c(large_offset * 1024), basereg))

                addr.base = basereg
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
        asm = block(f"{action} {cursor.name} register block @ {block_offset}")

        b_row, _, _, _ = cursor.get_block(cursor_ptr, block_offset)

        process_size = min(v_size, cursor.br)

        for ic in range(cols):
            for ir in range(rows):
                if (mask is None) or (mask[ir,ic]):
                    all_coords = [Coords(down=ir*v_size+i,right=ic) for i in range(process_size)]
                    has_nonzero = [cursor.has_nonzero_cell(cursor_ptr, block_offset, offset) for offset in all_coords]
                    if all(has_nonzero):
                        cell_offset = all_coords[0]
                        addr, comment = cursor.look(cursor_ptr, block_offset, cell_offset)
                        addr.disp += self.precision.size() * load_offset

                        processed = ir * process_size
                        if processed >= b_row:
                            continue
                        p = self.pred_n_trues(min(process_size, b_row - processed), v_size, 'm')
                        if store:
                            asm.add(mov(registers[ir,ic], addr, True, comment, pred=p))
                            if prefetching == 'BL2viaC':
                                asm.add(prefetch(mem(additional_regs[0], addr.disp)))
                        else:
                            asm.add(mov(addr, registers[ir,ic], True, comment, pred=p))
                    elif any(has_nonzero):
                        contiguous = True
                        firsti = 0
                        lasti = None
                        for i in range(process_size):
                            if has_nonzero[i]:
                                firsti = i
                                break
                        bitmask = 0
                        for i in range(process_size):
                            if has_nonzero[i]:
                                bitmask |= 1 << i
                                if lasti is not None:
                                    contiguous = False
                            elif i > firsti:
                                if lasti is None:
                                    lasti = i - 1
                        if lasti is None:
                            lasti = process_size
                        addr, comment = cursor.look(cursor_ptr, block_offset, all_coords[firsti])
                        # assume contiguous memory here

                        maskFound = False
                        if firsti == 0 and contiguous:
                            self.predicates[lasti]

                        if not maskFound:
                            maskreg = kmask(3)

                            asm.add(mov(maskreg, additional_regs[0], False))
                            asm.add(mov(additional_regs[0], maskreg, False))
                            pred = Predicate(maskreg, True)

                        needsExpand = not contiguous

                        if store:
                            asm.add(mov(registers[ir,ic], addr, True, comment, pred=pred, expand=needsExpand))
                            if prefetching == 'BL2viaC':
                                asm.add(prefetch(mem(additional_regs[0], addr.disp)))
                        else:
                            asm.add(mov(addr, registers[ir,ic], True, comment, pred=pred, expand=needsExpand))
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

        regcache = RegisterCache(additional_regs)

        mask = sparse_mask(A_regs, A, A_ptr, to_A_block, B, B_ptr, to_B_block, v_size, True)
        asm.add(self.move_register_block(A, A_ptr, to_A_block, A_regs, v_size, additional_regs, mask, store=False))

        Vm = max(self.ceil_div(bm, v_size), 1)

        for Vmi in range(Vm):
            for bki in range(bk):       # inside this k-block
                for bni in range(bn):   # inside this n-block
                    to_bcell = Coords(down=bki, right=bni)
                    to_acell = Coords(down=Vmi*v_size, right=bki)
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_bcell) and A.has_nonzero_cell(A_ptr, to_A_block, to_acell):
                        B_addr, B_comment = B.look(B_ptr, to_B_block, to_bcell)
                        self.reg_based_scaling(regcache, asm, B_addr, additional_regs, True)
                        comment = f"C[{Vmi*v_size}:{Vmi*v_size+v_size},{bni}] += A[{Vmi*v_size}:{Vmi*v_size+v_size},{bki}]*{B_comment}"
                        asm.add(fma(B_addr, A_regs[Vmi, bki], C_regs[Vmi, bni], comment=comment, bcast=0, sub=sub))
        return asm
