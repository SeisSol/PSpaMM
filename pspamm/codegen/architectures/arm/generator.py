from pspamm.cursors import *

from pspamm.codegen.architectures.arm.operands import *
from pspamm.codegen.ast import *
from pspamm.codegen.sugar import *
from pspamm.codegen.generator import *
from pspamm.codegen.precision import *


class Generator(AbstractGenerator):

    template = """
void {funcName} (const {real_type}* A, const {real_type}* B, {real_type}* C, {real_type} alpha, {real_type} beta, const {real_type}* prefetch) {{
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

    def get_v_size(self):
        return 16 // self.precision.size()

    def get_template(self):
        return Generator.template

    def use_broadcast(self):
        return True

    def has_masks(self):
        return False
    
    def init_mask(self, m, bm, v_size, tempreg, maskregs):
        return block("")
    
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

    def make_reg_blocks(self, bm:int, bn:int, bk:int, v_size:int, nnz:int, m:int, n:int, k:int, prefetch: str):
        assert(bm % v_size == 0)
        vm = bm//v_size
        elem128 = 16 // self.get_precision().size()
        vk = -(bk // -elem128)
        assert((bn+bk) * vm + bn * vk <= 32)  # Needs to fit in NEON v registers

        prec = {
            Precision.DOUBLE: "2d",
            Precision.SINGLE: "4s",
            Precision.HALF: "8h",
        }[self.get_precision()]

        A_regs = Matrix([[v(vm*c + r, prec) for c in range(bk)] for r in range(vm)])
        B_regs = Matrix([[v(vm*bk + bn * r + c, prec) for c in range(bn)] for r in range(vk)])
        C_regs = Matrix([[v(32 - vm*bn + vm*c + r, prec) for c in range(bn)]
                                                   for r in range(vm)])

        # get vector register number of the first vector in B_regs
        b_reg = vm*bk
        alpha_reg = [v(b_reg, prec), v(b_reg, prec)]
        beta_reg = [v(b_reg + 1, prec), v(b_reg + 1, prec)]


        starting_regs = [r(0), r(1), r(2), r(3), r(4), r(5), r(11)]

        additional_regs = [r(8), xzr, r(10)]

        loop_regs = [r(12), r(13), r(14)]

        prefetch_reg = prefetch is not None

        return A_regs, B_regs, C_regs, starting_regs, alpha_reg, beta_reg, loop_regs, additional_regs, [], prefetch_reg

    def make_scaling_offsets(self,
                         additional_regs: List[Register],
                         nnz: int
                        ) -> Block:

        asm = block("No register based scaling")
        return asm

    def init_block(self, size):
        return block("")
    
    class LoadStoreLocation:
        def __init__(self, addr, register, comment, pfaddr=None):
            self.addr = addr
            self.register = register
            self.comment = comment
            self.pfaddr = pfaddr

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
                            pf_cursor_ptr: CursorLocation = None
                            ) -> Block:

        rows, cols = registers.shape

        locations = []
        for ic in range(cols):
            for ir in range(rows):
                if (mask is None) or (mask[ir,ic]):
                    all_coords = [Coords(down=ir*v_size+i,right=ic) for i in range(v_size)]
                    has_nonzero = [cursor.has_nonzero_cell(cursor_ptr, block_offset, offset) for offset in all_coords]
                    if not any(has_nonzero):
                        continue
                    elif any(has_nonzero) and not all(has_nonzero):
                        raise NotImplementedError("Element-wise sparsity in A is not yet implemented.")

                    cell_offset = Coords(down=ir*v_size, right=ic)
                    addr, comment = cursor.look(cursor_ptr, block_offset, cell_offset)
                    addr.disp += self.precision.size() * load_offset

                    if prefetching:
                        pfaddr, _ = pf_cursor.look(pf_cursor_ptr, block_offset, cell_offset)
                        pfaddr.disp += self.precision.size() * load_offset
                    else:
                        pfaddr = None
                    locations += [self.LoadStoreLocation(addr, registers[ir,ic], comment, pfaddr)]

        return self.fuse_loadstore_block(locations, store, cursor.name, block_offset, additional_regs)

    def fuse_loadstore_block(self, locations, store, name, block_offset, additional_regs):
        offsets = list(sorted([(location.addr.disp,location) for location in locations]))

        action = "Store" if store else "Load"
        asm = block(f"{action} {name} register block @ {block_offset}")

        curpf = 0
        cur11 = -1000
        fuse_cache = []
        def try_flush_cache(force, cur11):
            if len(fuse_cache) == 0:
                return

            if force:
                op1 = fuse_cache[0]
                op2 = fuse_cache[1] if len(fuse_cache) > 1 else None
                op3 = fuse_cache[2] if len(fuse_cache) > 2 else None
                op4 = fuse_cache[3] if len(fuse_cache) > 3 else None

                max_offset = [65520, 1008, 48, 64][len(fuse_cache) - 1]
                div_offset = [16, 16, 24, 32][len(fuse_cache) - 1]

                comment = f'{op1.comment}'
                if op2 is not None: comment += f', {op2.comment}'
                if op3 is not None: comment += f', {op3.comment}'
                if op4 is not None: comment += f', {op4.comment}'

                offset = op1.addr.disp - cur11 if cur11 >= 0 else op1.addr.disp

                if cur11 >= 0:
                    op1.addr.disp = offset
                    op1.addr.base = additional_regs[0]

                if offset > max_offset or offset % div_offset != 0:
                    if cur11 < 0:
                        asm.add(add(offset, additional_regs[0], "", op1.addr.base))
                        cur11 = offset
                    else:
                        asm.add(add(offset, additional_regs[0], ""))
                        cur11 += offset
                    op1.addr.disp = 0
                    op1.addr.base = additional_regs[0]
                
                op1r = op1.register
                op2r = op2.register if op2 is not None else None
                op3r = op3.register if op3 is not None else None
                op4r = op4.register if op4 is not None else None

                if store:
                    asm.add(st(op1r, op1.addr, True, comment, src2=op2r, src3=op3r, src4=op4r))
                else:
                    asm.add(ld(op1.addr, op1r, True, comment, dest2=op2r, dest3=op3r, dest4=op4r))
                
                fuse_cache.clear()
            
            return cur11

        for _,location in offsets:
            if len(fuse_cache) > 0:
                can_fuse = location.addr.disp == fuse_cache[-1].addr.disp + 16

                # TODO: extend to 4?
                max_length = len(fuse_cache) == 2

                cur11 = try_flush_cache(not can_fuse or max_length, cur11)

            fuse_cache += [location]

            if location.pfaddr is not None:
                if location.pfaddr.disp - curpf >= 32768:
                    asm.add(add(location.pfaddr.disp, additional_regs[2], "increment the prefetch register", location.pfaddr.base))
                    curpf = location.pfaddr.disp
                if curpf > 0:
                    reg = additional_regs[2]
                    disp = location.pfaddr.disp - curpf
                else:
                    reg = location.pfaddr.base
                    disp = location.pfaddr.disp
                asm.add(prefetch(mem(reg, disp), "", access_type="LD", closeness="L2", temporality="KEEP"))

        cur11 = try_flush_cache(True, cur11)
        
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
        asm.add(self.move_register_block(A, A_ptr, to_A_block, A_regs, v_size, additional_regs, mask, store=False))

        elem128 = 16 // self.get_precision().size()
        vk = -(bk // -elem128)

        # TODO: fuse loads here as well
        bs = []
        firstloc = {}
        locations = []
        for Vmi in range(bm//v_size):
            for bni in range(bn):   # inside this n-block
                for bki in range(bk):       # inside this k-block
                    bki_reg = bki // elem128
                    to_bcell = Coords(down=bki, right=bni)
                    to_acell = Coords(down=Vmi*v_size, right=bki)
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_bcell):
                        if (bki_reg, bni) not in firstloc:
                            B_cell_addr, B_comment = B.look(B_ptr, to_B_block, to_bcell)
                            firstloc[(bki_reg, bni)] = self.LoadStoreLocation(B_cell_addr, B_regs[bki_reg, bni], B_comment)
                        if A.has_nonzero_cell(A_ptr, to_A_block, to_acell) and B_regs[bki_reg, bni] not in bs:
                            locations += [firstloc[(bki_reg, bni)]]
                            bs.append(B_regs[bki_reg, bni])
        asm.add(self.fuse_loadstore_block(locations, False, B.name, to_B_block, additional_regs))

        cell_indices = {}
        for bki in range(bk):       # inside this k-block
            # TODO: refactor cell_indices into the cursors/blocks
            for Vmi in range(bm//v_size):
                for bni in range(bn):   # inside this n-block
                    to_bcell = Coords(down=bki, right=bni)
                    to_acell = Coords(down=Vmi*v_size, right=bki)

                    bki_reg = bki // elem128
                    if (Vmi, bki_reg, bni) not in cell_indices:
                        cell_indices[(Vmi, bki_reg, bni)] = 0
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_bcell) and A.has_nonzero_cell(A_ptr, to_A_block, to_acell):
                        _, B_comment = B.look(B_ptr, to_B_block, to_bcell)
                        comment = f"C[{Vmi*v_size}:{Vmi*v_size+v_size},{bni}] += A[{Vmi*v_size}:{Vmi*v_size+v_size},{bki}]*{B_comment}"
                        asm.add(fma(B_regs[bki_reg, bni], A_regs[Vmi, bki], C_regs[Vmi, bni], comment=comment, bcast=cell_indices[(Vmi, bki_reg, bni)], sub=sub))
                    
                    if B.has_nonzero_cell(B_ptr, to_B_block, to_bcell):
                        cell_indices[(Vmi, bki_reg, bni)] += 1

        return asm
