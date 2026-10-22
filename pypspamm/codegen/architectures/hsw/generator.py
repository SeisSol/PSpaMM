from pypspamm.codegen.architectures.hsw.operands import *
from pypspamm.codegen.ast import *
from pypspamm.codegen.forms import loop
from pypspamm.codegen.generator import *
from pypspamm.codegen.microkernel import LoadedOnce, cells, contribution
from pypspamm.codegen.precision import *
from pypspamm.codegen.regcache import *
from pypspamm.codegen.sugar import *
from pypspamm.codegen.target import TARGETS
from pypspamm.cursors import *


class Generator(AbstractGenerator):
    target = TARGETS["hsw"]

    template = """
void {funcName} (const {real_type}* A, const {real_type}* B, {real_type}* C, {real_type} alpha, {real_type} beta, {real_type} const* prefetch) {{
  {real_type}* alpha_p = &alpha;
  {real_type}* beta_p = &beta;
{prelude}  __asm__ __volatile__(
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
        asm.add(mov(InputOperand(f"0", "m", "A"), starting_regs[0], False))
        asm.add(mov(InputOperand(f"1", "m", "B"), starting_regs[1], False))
        asm.add(mov(InputOperand(f"2", "m", "C"), starting_regs[2], False))
        asm.add(mov(InputOperand(f"3", "m", "alpha_p"), starting_regs[3], False))
        asm.add(mov(InputOperand(f"4", "m", "beta_p"), starting_regs[4], False))
        if prefetch:
            asm.add(mov(InputOperand(f"5", "m", "prefetch"), starting_regs[5], False))
        return asm

    def make_expand_predicate(self, mask):
        combined = 0
        offset = 0
        for i, value in enumerate(mask):
            if value:
                combined |= offset << (8 * i)
                offset += 1
            else:
                combined |= 255 << (8 * i)
        return combined

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
        assert bm % v_size == 0
        vm = self.ceil_div(bm, v_size)

        # A is kept in registers whenever the whole block fits
        if (bn + bk) * vm + bn * bk <= self.target.vector_registers:
            self.preloadA = True
        else:
            self.preloadA = False
            assert copies == 1, "a rotated loop needs A in registers"
            assert bn * vm + bn * bk + 1 <= self.target.vector_registers

        vmm = {1: xmm, 2: ymm}[self.v_len]

        if self.preloadA:
            A_regs = Matrix(
                [[vmm(vm * c + r) for c in range(bk * copies)] for r in range(vm)]
            )
            Aoffset = vm * bk * copies
        else:
            A_regs = Matrix([[vmm(0) for c in range(bk)] for r in range(vm)])
            Aoffset = 1

        B_regs = Matrix(
            [[vmm(Aoffset + bn * r + c) for c in range(bn)] for r in range(bk * copies)]
        )
        C_regs = Matrix(
            [[vmm(16 - vm * bn + vm * c + r) for c in range(bn)] for r in range(vm)]
        )
        starting_regs = [rdi, rsi, rdx, rbx, rcx]

        b_reg = Aoffset
        alpha_reg = [xmm(b_reg), vmm(b_reg)]
        beta_reg = [xmm(b_reg + 1), vmm(b_reg + 1)]

        additional_regs = [r(9), r(10), r(11), r(15), rax]  # ,r(13),r(14)

        prefetch_reg = prefetch == "BL2viaC"
        if prefetch_reg:
            starting_regs += [r(8)]
        else:
            additional_regs += [r(8)]

        loop_regs = [r(12), r(13), r(14)]

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

        asm = block("Optimize usage of offsets when accessing B Matrix")

        for i in range(1, len(additional_regs)):
            asm.add(mov(c(self.scale_base() * (2 * i - 1)), additional_regs[i], False))

        return asm

    def init_block(self, size):
        return block("")

    def reg_based_scaling(
        self, asm, addr: MemoryAddress, additional_regs: List[Register]
    ):
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
                        self.reg_based_scaling(asm, addr, additional_regs)
                        if store:
                            asm.add(mov(registers[ir, ic], addr, True, comment))
                            if prefetching == "BL2viaC" and pf_cursor is not None:
                                addr, comment = pf_cursor.look(
                                    pf_cursor_ptr, block_offset, cell_offset
                                )
                                addr.disp += self.precision.size() * load_offset
                                self.reg_based_scaling(asm, addr, additional_regs)
                                asm.add(prefetch(addr, closeness="L2"))
                        else:
                            asm.add(mov(addr, registers[ir, ic], True, comment))
                    elif any(has_nonzero):
                        raise NotImplementedError(
                            "Element-wise sparsity in A is not yet fully implemented."
                        )
                        firsti = 0
                        for i in range(v_size):
                            if has_nonzero[i]:
                                firsti = i
                                break
                        addr, comment = cursor.look(
                            cursor_ptr, block_offset, all_coords[firsti]
                        )
                        # assume contiguous memory here

                        asm.add(
                            mov(
                                self.make_expand_predicate(all_coords),
                                additional_regs[0],
                                False,
                            )
                        )
        return asm

    def move_register_single(
        self,
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
        load_offset: int = 0,
    ) -> Block:

        asm = block("")

        if (mask is None) or (mask[ir, ic]):
            cell_offset = Coords(down=ir * v_size, right=ic)
            addr, comment = cursor.look(cursor_ptr, block_offset, cell_offset)
            addr.disp += self.precision.size() * load_offset
            asm.add(mov(addr, registers[ir, ic], True, comment))
        return asm

    def make_zero_block(self, registers: Matrix[Register], additional_regs) -> Block:

        rows, cols = registers.shape
        asm = block("zero registers")

        for ic in range(cols):
            for ir in range(rows):
                asm.add(mov(0, registers[ir, ic], True))

        return asm

    #: temporaries an ELLPACK kernel cycles through for its permutations
    ellpack_temps = 4

    def ellpack_registers(self, k):
        """How many vector registers a column of B with k rows fills in an
        ELLPACK kernel: none, as the kernel reads B from memory where it
        permutes it. Single and double precision only."""

        if self.precision not in (Precision.SINGLE, Precision.DOUBLE):
            return None
        return 0

    def make_ellpack(self, layout, n, ldb, ldc, alpha, beta, types, encode):
        """The kernel for an A in sliced ELLPACK (see codegen/ellpack.py).

        Sixteen registers do not hold a column of B, so a step permutes the
        registers of B it reads straight from memory: by an immediate, but for
        single precision on 256 bit, which takes its positions from the table.
        A step that reads two registers permutes both by the same positions,
        those of its lanes within their register, and blends the two. A step
        whose lanes read their own positions of one register takes it as it
        is. The vector of B at the end of a column, which would read past it,
        is loaded once per column under a mask and permuted in its register.

        A loop over the columns of B and C takes the full slices, each in an
        accumulator that starts from C as beta says; a second loop takes the
        slice at the end of the rows, whose loads and stores of C are masked,
        for the reason given with the AVX-512 kernel. alpha other than one or
        minus one is applied at the end; minus one is in the values already.

        The register A is read through points at the table, which starts with
        the masks; encode turns a value into a word of it. Returns the
        statements, the table and the registers the arguments are loaded into.
        """

        v_size = self.get_v_size()
        size = types.a.size()
        vmm = {1: xmm, 2: ymm}[self.v_len]
        table, B, C, alpha_p, beta_p = rdi, rsi, rdx, rbx, rcx
        Bcol, Ccol, counter = r(8), r(9), r(12)
        assert self.ellpack_registers(layout.k) == 0

        ktail = layout.k % v_size
        mtail = layout.m % v_size
        sign = -1.0 if alpha == -1.0 else 1.0
        # alpha other than one or minus one is known at run time only, and AB
        # is scaled by it before beta C is added
        scaled = alpha not in (1.0, -1.0)
        by_register = self.precision == Precision.SINGLE and v_size == 8

        free = iter(vmm(i) for i in range(self.target.vector_registers))
        kmask = next(free) if ktail > 0 else None
        mmask = next(free) if mtail > 0 else None
        btail = next(free) if ktail > 0 else None
        alpha_reg = next(free) if scaled else None
        beta_reg = next(free) if beta not in (0.0, 1.0) else None
        tregs = [next(free) for _ in range(self.ellpack_temps)]
        accs = list(free)

        words = []
        asm = block("ELLPACK kernel")
        ones = (1 << (8 * size)) - 1
        for mask, lanes in ((kmask, ktail), (mmask, mtail)):
            if mask is not None:
                asm.add(mov(mem(table, len(words) * size), mask, True, "mask"))
                words.extend(ones if lane < lanes else 0 for lane in range(v_size))
        if alpha_reg is not None:
            asm.add(bcst(alpha_p, alpha_reg, "alpha"))
        if beta_reg is not None:
            asm.add(bcst(beta_p, beta_reg, "beta"))

        permutations = [0]

        def temp():
            register = tregs[permutations[0] % len(tregs)]
            permutations[0] += 1
            return register

        def source(reg):
            if ktail > 0 and reg == layout.registers - 1:
                return btail
            return mem(Bcol, reg * v_size * size)

        def immediate(index):
            # vpermilpd takes one bit per lane, the others two
            bits = 1 if v_size == 2 else 2
            return sum((p % v_size) << (bits * lane) for lane, p in enumerate(index))

        def gather(column, step):
            """A register with the entries of B the lanes of the step need."""

            index = step.index(v_size)
            if by_register:
                # vpermps reads the low three bits of a position, its place
                # within its register
                control = temp()
                column.add(
                    mov(mem(table, len(words) * size), control, True, "positions")
                )
                words.extend(index)
                first = control
            else:
                control = c(immediate(index))
                first = temp()
            if len(step.sources) == 2:
                second = temp()
                column.add(permute(control, source(step.sources[1]), second))
            column.add(permute(control, source(step.sources[0]), first))
            if len(step.sources) == 2:
                mask = sum(1 << lane for lane, p in enumerate(index) if p >= v_size)
                column.add(blend(first, second, first, mask))
            return first

        def take(column, piece, acc):
            pred = Predicate(mmask) if piece.rows < v_size else None
            target = mem(Ccol, piece.start * size)
            if beta == 0.0 or scaled:
                column.add(mov(0, acc, True))
            else:
                column.add(mov(target, acc, True, "C", pred=pred))
                if beta != 1.0:
                    column.add(mul(acc, beta_reg, acc, "C = beta * C"))
            for step in piece.steps:
                if not step.in_place(v_size):
                    gathered = gather(column, step)
                elif isinstance(source(step.sources[0]), Register):
                    gathered = source(step.sources[0])
                else:
                    gathered = temp()
                    column.add(mov(source(step.sources[0]), gathered, True, "B"))
                column.add(
                    fma(
                        mem(table, len(words) * size),
                        gathered,
                        acc,
                        f"C[{piece.start}:{piece.start + piece.rows}] += A B[{step.sources}]",
                    )
                )
                words.extend(encode(sign * value) for value in step.values)
            result = acc
            if scaled and beta == 0.0:
                column.add(mul(acc, alpha_reg, acc, "C = alpha * AB"))
            elif scaled:
                result = temp()
                column.add(mov(target, result, True, "C", pred=pred))
                if beta != 1.0:
                    column.add(mul(result, beta_reg, result, "beta * C"))
                column.add(fma(alpha_reg, acc, result, "C = alpha * AB + beta * C"))
            column.add(mov(result, target, True, "store", pred=pred))

        def columns(description, pieces):
            column = block(description)
            if ktail > 0 and any(
                layout.registers - 1 in step.sources
                for piece in pieces
                for step in piece.steps
            ):
                column.add(
                    mov(
                        mem(Bcol, (layout.registers - 1) * v_size * size),
                        btail,
                        True,
                        "B at the end of the column",
                        pred=Predicate(kmask),
                    )
                )
            for index, piece in enumerate(pieces):
                take(column, piece, accs[index % len(accs)])
            column.add(add(ldb * size, Bcol, "next column of B"))
            column.add(add(ldc * size, Ccol, "next column of C"))
            asm.add(mov(B, Bcol, False))
            asm.add(mov(C, Ccol, False))
            asm.add(loop(counter, n).body(*column.contents))

        full = [piece for piece in layout.slices if piece.rows == v_size]
        rest = [piece for piece in layout.slices if piece.rows < v_size]
        if full:
            columns("full slices, one column", full)
        if rest:
            columns("the slice at the end, one column", rest)

        return asm, words, [table, B, C, alpha_p, beta_p]

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
        if self.preloadA:
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
        else:
            asm.add(
                self.move_register_single(
                    A,
                    A_ptr,
                    to_A_block,
                    A_regs,
                    v_size,
                    additional_regs,
                    0,
                    0,
                    mask,
                    store=False,
                )
            )

        Vm = self.ceil_div(bm, v_size)

        loaded = LoadedOnce()
        for Vmi, bni, bki, to_acell, to_bcell in cells(
            A, B, A_ptr, B_ptr, to_A_block, to_B_block, Vm, bn, bk, v_size
        ):
            B_addr, B_comment = B.look(B_ptr, to_B_block, to_bcell)
            self.reg_based_scaling(asm, B_addr, additional_regs)
            if loaded.first(B_regs[bki, bni], B_addr):
                asm.add(bcst(B_addr, B_regs[bki, bni], comment=B_comment))

        for bki in range(bk):  # inside this k-block
            for Vmi in range(Vm):
                if not self.preloadA and not (Vmi, bki) == (0, 0):
                    asm.add(
                        self.move_register_single(
                            A,
                            A_ptr,
                            to_A_block,
                            A_regs,
                            v_size,
                            additional_regs,
                            Vmi,
                            bki,
                            mask,
                            store=False,
                        )
                    )
                for bni in range(bn):  # inside this n-block
                    to_bcell = Coords(down=bki, right=bni)
                    to_acell = Coords(down=Vmi * v_size, right=bki)
                    if B.has_nonzero_cell(
                        B_ptr, to_B_block, to_bcell
                    ) and A.has_nonzero_cell(A_ptr, to_A_block, to_acell):
                        _, B_comment = B.look(B_ptr, to_B_block, to_bcell)
                        comment = contribution(Vmi, bni, bki, v_size, B_comment)
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
