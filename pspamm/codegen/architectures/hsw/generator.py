from pspamm.cursors import *

from pspamm.codegen.architectures.hsw.operands import *
from pspamm.codegen.ast import *
from pspamm.codegen.sugar import *
from pspamm.codegen.generator import *
from pspamm.codegen.precision import *
from pspamm.codegen.regcache import *

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
    
    def pred_n_trues(self, count, v_size, mode):
        # hacked in right now: we set a number as predicate if we need it
        if count < v_size:
            return (1 << count) - 1
        else:
            return None

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

    def make_reg_blocks(self, bm:int, bn:int, bk:int, v_size:int, nnz:int, m:int, n:int, k:int):
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

        available_regs = [r(9),r(10),r(11),r(15),rax] # ,r(13),r(14)

        additional_regs = [r(8)]

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

        return A_regs, B_regs, C_regs, starting_regs, alpha_reg, beta_reg, loop_regs, additional_regs, []


    def bcst_alpha_beta(self,
                        alpha_reg: Register,
                        beta_reg: Register,
                        ) -> Block:

        asm = block("Broadcast alpha and beta when needed")

#        asm.add(bcst(alpha_reg[0], alpha_reg[1]))
#        asm.add(bcst(beta_reg[0], beta_reg[1]))
        
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
                # TODO: not 100%ly sure about this code here...
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
                            load_offset: int = 0,
                            temp = None
                           ) -> Block:

        rows, cols = registers.shape
        action = "Store" if store else "Load"
        asm = block("{} {} register block @ {}".format(action,cursor.name,block_offset))

        for ic in range(cols):
            for ir in range(rows):
                if (mask is None) or (mask[ir,ic]):
                    all_coords = [Coords(down=ir*v_size+i,right=ic) for i in range(v_size)]
                    has_nonzero = [cursor.has_nonzero_cell(cursor_ptr, block_offset, offset) for offset in all_coords]
                    if all(has_nonzero):
                        cell_offset = all_coords[0]
                        addr, comment = cursor.look(cursor_ptr, block_offset, cell_offset)
                        addr.disp += self.precision.size() * load_offset
                        if store:
                            asm.add(mov(registers[ir,ic], addr, True, comment))
                            if prefetching == 'BL2viaC':
                                asm.add(prefetch(mem(additional_regs[0], addr.disp)))
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
                        if store:
                            asm.add(mov(registers[ir,ic], addr, True, comment, pred=additional_regs[0], expand=True, temp=temp))
                            if prefetching == 'BL2viaC':
                                asm.add(prefetch(mem(additional_regs[0], addr.disp)))
                        else:
                            asm.add(mov(addr, registers[ir,ic], True, comment, pred=additional_regs[0], expand=True, temp=temp))
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

        regcache = RegisterCache(additional_regs)

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
                        self.reg_based_scaling(regcache, asm, B_addr, additional_regs, True)
                        if B_regs[bki, bni] not in bs:
                            asm.add(bcst(B_addr, B_regs[bki, bni], comment=B_comment))
                            bs.append(B_regs[bki, bni])
                            bsv.append(B_addr)
                        else:
                            # just to make sure we do not use registers differently in a block
                            assert bsv[bs.index(B_regs[bki, bni])].ugly == B_addr.ugly

        for Vmi in range(Vm):
            for bki in range(bk):       # inside this k-block
                if not self.preloadA and not (Vmi, bki) == (0,0):
                    asm.add(self.move_register_single(A, A_ptr, to_A_block, A_regs, v_size, additional_regs, Vmi, bki, mask, store=False))
                for bni in range(bn):   # inside this n-block
                    to_bcell = Coords(down=bki, right=bni)
                    to_acell = Coords(down=Vmi*v_size, right=bki)
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_bcell) and A.has_nonzero_cell(A_ptr, to_A_block, to_acell):
                        B_addr, B_comment = B.look(B_ptr, to_B_block, to_bcell)
                        comment = "C[{}:{},{}] += A[{}:{},{}]*{}".format(Vmi*v_size,Vmi*v_size+v_size,bni,Vmi*v_size,Vmi*v_size+v_size,bki,B_comment)
                        asm.add(fma(B_regs[bki, bni], A_regs[Vmi, bki], C_regs[Vmi, bni], comment=comment, bcast=None, sub=sub))
        return asm

    def init_prefetching(self, prefetching):
        
        if prefetching != 'BL2viaC':
            Generator.template = Generator.template.format(prefetching_mov = "", prefetching_decl = "")    
            return None
        
        prefetchReg = r(8)
        Generator.template = Generator.template.format(prefetching_mov = '    "movq %5, {}\\n\\t"'.format(prefetchReg.ugly), prefetching_decl = ', "m"(prefetch)')
        return prefetchReg
