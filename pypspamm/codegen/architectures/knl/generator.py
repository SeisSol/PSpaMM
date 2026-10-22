from pypspamm.codegen.architectures.knl.operands import *
from pypspamm.codegen.ast import *
from pypspamm.codegen.forms import loop
from pypspamm.codegen.generator import *
from pypspamm.codegen.microkernel import cells, contribution
from pypspamm.codegen.precision import *
from pypspamm.codegen.regcache import *
from pypspamm.codegen.sugar import *
from pypspamm.codegen.target import TARGETS
from pypspamm.cursors import *


class Generator(AbstractGenerator):
    target = TARGETS["knl"]

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
    v_len = 4
    predicates = {0: kmask(0)}

    def get_v_size(self):
        return (16 // self.precision.size()) * self.v_len

    def get_template(self):
        return Generator.template

    def scale_base(self):
        # larger scaling range for B inline broadcasts
        return self.precision.size() * 256

    def pred_n_trues(self, count, v_size, mode):
        # a bit hacky at the moment (won't work for all masks)
        if count < v_size:
            return Predicate(self.predicates[count], mode == "z")
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
        vm = self.ceil_div(bm, v_size)
        assert self.target.fits(bn, bk, vm)

        vmm = {1: xmm, 2: ymm, 4: zmm}[self.v_len]

        A_regs = Matrix(
            [[vmm(vm * c + r) for c in range(bk * copies)] for r in range(vm)]
        )
        B_regs = Matrix([[]])
        C_regs = Matrix(
            [[vmm(32 - vm * bn + vm * c + r) for c in range(bn)] for r in range(vm)]
        )

        starting_regs = [rdi, rsi, rdx, rbx, rcx]

        alpha_reg = [rbx, rbx]
        beta_reg = [rcx, rcx]

        additional_regs = [r(9), r(10), r(11), r(15), rax]  # ,r(13),r(14)

        prefetch_reg = prefetch == "BL2viaC"
        if prefetch_reg:
            starting_regs += [r(8)]
        else:
            additional_regs += [r(8)]

        mask_regs = [kmask(1), kmask(2)]

        loop_regs = [r(12), r(13), r(14)]

        # FIXME: a bit hacky to have the mask setup here
        rest = bm % v_size
        rest2 = (m % bm) % v_size
        self.predicates[rest] = kmask(1)
        self.predicates[rest2] = kmask(2)
        self.predicates[0] = kmask(0)

        return (
            A_regs,
            B_regs,
            C_regs,
            starting_regs,
            alpha_reg,
            beta_reg,
            loop_regs,
            additional_regs,
            mask_regs,
            prefetch_reg,
        )

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

    def make_scaling_offsets(self, additional_regs: List[Register], nnz: int) -> Block:

        asm = block("Optimize usage of offsets when accessing B Matrix")

        scale = self.scale_base()
        for i in range(1, len(additional_regs)):
            asm.add(mov(c((2 * i - 1) * scale), additional_regs[i], False))

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
                base >>= 1
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
    ) -> Block:

        rows, cols = registers.shape
        action = "Store" if store else "Load"
        asm = block(f"{action} {cursor.name} register block @ {block_offset}")

        b_row, _, _, _ = cursor.get_block(cursor_ptr, block_offset)

        process_size = min(v_size, cursor.br)

        maskcache = RegisterCache([kmask(3), kmask(4), kmask(5), kmask(6), kmask(7)])

        for ic in range(cols):
            for ir in range(rows):
                if (mask is None) or (mask[ir, ic]):
                    # no register-based scaling here (for now)

                    processed = ir * process_size

                    size = min(process_size, b_row - processed)

                    all_coords = [
                        Coords(down=ir * process_size + i, right=ic)
                        for i in range(size)
                    ]
                    has_nonzero = [
                        cursor.has_nonzero_cell(cursor_ptr, block_offset, offset)
                        for offset in all_coords
                    ]
                    if any(has_nonzero):
                        contiguous = True
                        firsti = 0
                        lasti = None
                        for i in range(size):
                            if has_nonzero[i]:
                                firsti = i
                                break
                        bitmask = 0
                        for i in range(size):
                            if has_nonzero[i]:
                                bitmask |= 1 << i
                                if lasti is not None:
                                    contiguous = False
                            elif i > firsti:
                                if lasti is None:
                                    lasti = i
                        if lasti is None:
                            lasti = size
                        addr, comment = cursor.look(
                            cursor_ptr, block_offset, all_coords[firsti]
                        )
                        addr.disp += self.precision.size() * load_offset
                        # assume contiguous memory here

                        maskFound = False
                        needsExpand = not (firsti == 0 and contiguous)
                        # TODO: if firsti > 0, set addr.disp -= firsti
                        if not needsExpand:
                            if lasti == v_size:
                                pred = None
                                maskFound = True
                            elif lasti in self.predicates:
                                pred = Predicate(self.predicates[lasti], True)
                                maskFound = True
                            else:
                                # mostly implemented, but there are still bugs
                                raise NotImplementedError(
                                    "Element-wise sparsity in A is not yet implemented"
                                )
                        else:
                            raise NotImplementedError(
                                "Element-wise sparsity in A is not yet implemented"
                            )

                        if not maskFound:
                            maskreg, needsAssign = maskcache.get(bitmask)
                            if needsAssign:
                                asm.add(mov(bitmask, additional_regs[0], False))
                                asm.add(mov(additional_regs[0], maskreg, False))
                            pred = Predicate(maskreg, True)

                        if store:
                            asm.add(
                                mov(
                                    registers[ir, ic],
                                    addr,
                                    True,
                                    comment,
                                    pred=pred,
                                    expand=needsExpand,
                                )
                            )
                            if prefetching == "BL2viaC" and pf_cursor is not None:
                                addr, comment = pf_cursor.look(
                                    pf_cursor_ptr, block_offset, all_coords[firsti]
                                )
                                addr.disp += self.precision.size() * load_offset
                                asm.add(prefetch(addr, closeness="L2"))
                        else:
                            asm.add(
                                mov(
                                    addr,
                                    registers[ir, ic],
                                    True,
                                    comment,
                                    pred=pred,
                                    expand=needsExpand,
                                )
                            )
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
        ELLPACK kernel, or None where that leaves no room for the rest: the
        temporaries and the accumulator. Half precision would need the FP16
        extension, which the kernel does not use."""

        if self.precision.size() < 4:
            return None
        registers = -(k // -self.get_v_size())
        if registers + self.ellpack_temps + 1 > self.target.vector_registers:
            return None
        return registers

    def make_ellpack(self, layout, n, ldb, ldc, alpha, beta, types, encode):
        """The kernel for an A in sliced ELLPACK (see codegen/ellpack.py).

        A loop over the columns of B and C takes the full slices: the column of
        B in registers, the last one loaded with the end of the column masked
        to zero, and per slice the accumulator, which starts from C as beta
        says and takes the steps of the slice. A step gathers the entries of
        its lanes by a permutation of the registers of B it reads, by its
        positions, and multiplies them with its values. A step whose lanes read
        their own positions of one register takes that register as it is.
        alpha other than one or minus one is applied at the end; minus one is
        in the values already. Each slice starts the accumulator anew, from
        zero or from C, so renaming keeps the slices apart.

        Positions and values are the same for every column, so the registers
        left over hold as many of them as there is room for, loaded once ahead
        of the loop: the values first, then the positions of the steps that
        read one register of B, which the permutation of one register leaves
        as they are, then the positions of the steps that read two, which
        vpermi2 overwrites and so takes a copy of. The rest comes from the
        table at every column, as a load of the positions and as the memory
        operand of the multiplication.

        A slice at the end of the rows, which a masked store finishes, comes
        in a loop over the columns of its own, with the registers of B it
        reads. The masked store at the end of a column of C covers the start
        of the next one as far as the memory order of the core is concerned,
        and a load there waits until the store has gone; in the loop of the
        full slices that would serialise the columns.

        The register A is read through points at the table; encode turns a
        value into a word of it. Returns the statements, the table and the
        registers the arguments are loaded into.
        """

        v_size = self.get_v_size()
        size = types.a.size()
        vmm = {1: xmm, 2: ymm, 4: zmm}[self.v_len]
        table, B, C, alpha_p, beta_p = rdi, rsi, rdx, rbx, rcx
        Bcol, Ccol, counter = r(8), r(9), r(12)

        temps = self.ellpack_temps
        assert self.ellpack_registers(layout.k) == layout.registers
        bregs = [vmm(i) for i in range(layout.registers)]
        tregs = [vmm(layout.registers + i) for i in range(temps)]
        acc = vmm(layout.registers + temps)
        free = [
            vmm(i)
            for i in range(layout.registers + temps + 1, self.target.vector_registers)
        ]
        # a permutation of one register by positions in a register of their
        # own exists from 256 bit on
        single = self.v_len >= 2

        # the table: per step its positions, unless it takes a register as it
        # is, and its values; where the words of a step start, by the first
        # row of its slice and its number there
        sign = -1.0 if alpha == -1.0 else 1.0
        words = []
        where = {}
        for piece in layout.slices:
            for number, step in enumerate(piece.steps):
                positions = None
                if not step.in_place(v_size):
                    positions = len(words)
                    words.extend(step.index(v_size))
                where[piece.start, number] = (positions, len(words))
                words.extend(encode(sign * value) for value in step.values)

        asm = block("ELLPACK kernel")
        ktail = layout.k % v_size
        mtail = layout.m % v_size
        if mtail > 0:
            asm.add(mov((1 << mtail) - 1, rax, False))
            asm.add(mov(rax, kmask(1), False))
        if ktail > 0:
            asm.add(mov((1 << ktail) - 1, rax, False))
            asm.add(mov(rax, kmask(2), False))

        # alpha other than one or minus one is known at run time only, and AB
        # is scaled by it before beta C is added
        scaled = alpha not in (1.0, -1.0)
        permutations = [0]

        def temp():
            register = tregs[permutations[0] % temps]
            permutations[0] += 1
            return register

        def load_b(column, registers):
            for reg in registers:
                pred = None
                if reg == len(bregs) - 1 and ktail > 0:
                    pred = Predicate(kmask(2), True)
                column.add(
                    mov(
                        mem(Bcol, reg * v_size * size),
                        bregs[reg],
                        True,
                        f"B[{reg}]",
                        pred=pred,
                    )
                )

        def kept(pieces):
            """The registers the words of these pieces go into ahead of their
            loop, by kind, first row of the slice and number of the step."""
            steps = [
                (piece.start, number, step)
                for piece in pieces
                for number, step in enumerate(piece.steps)
            ]
            values = [("values", start, number) for start, number, _ in steps]
            permuted = [
                (start, number, step)
                for start, number, step in steps
                if not step.in_place(v_size)
            ]
            unchanged = [
                ("positions", start, number)
                for start, number, step in permuted
                if len(step.sources) == 1 and single
            ]
            overwritten = [
                ("positions", start, number)
                for start, number, step in permuted
                if len(step.sources) == 2 or not single
            ]
            return dict(zip(values + unchanged + overwritten, free))

        def take(column, piece, registers):
            tail = piece.rows < v_size
            load = Predicate(kmask(1), True) if tail else None
            store = Predicate(kmask(1), False) if tail else None
            target = mem(Ccol, piece.start * size)
            if beta == 0.0 or scaled:
                column.add(mov(0, acc, True))
            else:
                column.add(mov(target, acc, True, "C", pred=load))
                if beta != 1.0:
                    column.add(mul(acc, beta_p, acc, "C = beta * C"))
            for number, step in enumerate(piece.steps):
                positions, values = where[piece.start, number]
                index = registers.get(("positions", piece.start, number))
                if step.in_place(v_size):
                    source = bregs[step.sources[0]]
                elif index is not None and len(step.sources) == 1 and single:
                    source = temp()
                    column.add(permute(index, bregs[step.sources[0]], source))
                else:
                    source = temp()
                    if index is None:
                        index = mem(table, positions * size)
                    column.add(mov(index, source, True, "positions"))
                    lo = bregs[step.sources[0]]
                    hi = bregs[step.sources[-1]]
                    column.add(permute(source, lo, source, table2=hi))
                value = registers.get(("values", piece.start, number))
                if value is None:
                    value = mem(table, values * size)
                column.add(
                    fma(
                        value,
                        source,
                        acc,
                        f"C[{piece.start}:{piece.start + piece.rows}] += A B[{step.sources}]",
                    )
                )
            result = acc
            if scaled and beta == 0.0:
                column.add(mul(acc, alpha_p, acc, "C = alpha * AB"))
            elif scaled:
                result = temp()
                column.add(mov(target, result, True, "C", pred=load))
                if beta != 1.0:
                    column.add(mul(result, beta_p, result, "beta * C"))
                column.add(
                    fma(mem(alpha_p, 0), acc, result, "C = alpha * AB + beta * C", 0)
                )
            column.add(mov(result, target, True, "store", pred=store))

        def columns(description, pieces):
            registers = kept(pieces)
            for (kind, start, number), register in registers.items():
                positions, values = where[start, number]
                word = positions if kind == "positions" else values
                asm.add(mov(mem(table, word * size), register, True, kind))
            column = block(description)
            used = sorted(
                {r for piece in pieces for step in piece.steps for r in step.sources}
            )
            load_b(column, used)
            for piece in pieces:
                take(column, piece, registers)
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

        mask = sparse_mask(
            A_regs, A, A_ptr, to_A_block, B, B_ptr, to_B_block, v_size, True
        )
        asm.add(
            self.move_register_block(
                A, A_ptr, to_A_block, A_regs, v_size, additional_regs, mask, store=False
            )
        )

        Vm = max(self.ceil_div(bm, v_size), 1)

        for Vmi, bni, bki, to_acell, to_bcell in cells(
            A, B, A_ptr, B_ptr, to_A_block, to_B_block, Vm, bn, bk, v_size, order="kmn"
        ):
            B_addr, B_comment = B.look(B_ptr, to_B_block, to_bcell)
            self.reg_based_scaling(asm, B_addr, additional_regs)
            comment = contribution(Vmi, bni, bki, v_size, B_comment)
            asm.add(
                fma(
                    B_addr,
                    A_regs[Vmi, bki],
                    C_regs[Vmi, bni],
                    comment=comment,
                    bcast=0,
                    sub=sub,
                )
            )
        return asm
