from .operands import Register


class VirtualRegister(Register):
    _created = 0

    def __init__(self, typeinfo, pool):
        super().__init__(typeinfo, "")
        self.register = None
        self.pool = pool

        # registers are handed out in creation order, so that the assignment
        # does not depend on the iteration order of a set
        self.serial = VirtualRegister._created
        VirtualRegister._created += 1

        self.usage = []

    def setRegister(self, register: Register):
        assert not isinstance(register, VirtualRegister)
        assert self.typeinfo == register.typeinfo

        self.register = register

    @property
    def ugly(self):
        return self.register.ugly if self.register is not None else f"vreg{id(self)}"

    @property
    def ugly_scalar_1d(self):
        return (
            self.register.ugly_scalar_1d
            if self.register is not None
            else f"vreg{id(self)}"
        )

    @property
    def ugly_scalar(self):
        return (
            self.register.ugly_scalar
            if self.register is not None
            else f"vreg{id(self)}"
        )

    @property
    def ugly_xmm(self):
        return (
            self.register.ugly_xmm if self.register is not None else f"vreg{id(self)}"
        )

    @property
    def clobbered(self):
        return (
            self.register.clobbered if self.register is not None else f"vreg{id(self)}"
        )

    def firstUsage(self):
        return None if len(self.usage) == 0 else self.usage[0]

    def lastUsage(self):
        return None if len(self.usage) == 0 else self.usage[-1]


class RegisterPool:
    def __init__(self, registers, name="register"):
        self.registers = registers
        self.name = name

    def assign(self, asm):
        unlive = list(self.registers)
        for instr in asm.flatten():
            mine = sorted(
                (
                    vreg
                    for vreg in instr.regs()
                    if isinstance(vreg, VirtualRegister) and vreg.pool is self
                ),
                key=lambda vreg: vreg.serial,
            )

            # an instruction reads its operands before it writes its result, so
            # a register that dies here is available to one that starts here
            for vreg in mine:
                if vreg.lastUsage() is instr and vreg.register is not None:
                    unlive.append(vreg.register)

            for vreg in mine:
                if vreg.firstUsage() is instr:
                    assert vreg.register is None, "Register assigned twice"
                    assert len(unlive) > 0, (
                        f"{self.name} pool exhausted: "
                        f"{len(self.registers)} registers are not enough"
                    )
                    vreg.register = unlive.pop(0)
                    if vreg.lastUsage() is instr:
                        unlive.append(vreg.register)


def usagePass(asm):
    for instruction in asm.flatten():
        for reg in instruction.regs():
            if isinstance(reg, VirtualRegister):
                reg.usage += [instruction]


def assignVirtualRegisters(asm, pools):
    usagePass(asm)
    for pool in pools:
        pool.assign(asm)
