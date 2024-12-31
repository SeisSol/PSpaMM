from .operands import Register

class VirtualRegister(Register):
    def __init__(self, typeinfo, pool):
        super().__init__(typeinfo, '')
        self.register = None
        self.pool = pool

        self.usage = []

    def setRegister(register: Register):
        assert not isinstance(register, VirtualRegister)
        assert self.typeinfo == register.typeinfo

        self.register = register

    @property
    def ugly(self):
        return self.register.ugly if self.register is not None else f'vreg{id(self)}'
    
    @property
    def clobbered(self):
        return self.register.clobbered if self.register is not None else f'vreg{id(self)}'
    
    def firstUsage(self):
        return None if len(self.usage) == 0 else self.usage[0]
    
    def lastUsage(self):
        return None if len(self.usage) == 0 else self.usage[-1]

class RegisterPool:
    def __init__(self, registers):
        self.registers = registers

    def assign(self, asm):
        unlive = list(self.registers)
        for instr in asm.flatten():
            for vreg in instr.regs():
                if isinstance(vreg, VirtualRegister) and vreg.pool is self:
                    if vreg.firstUsage() is instr:
                        assert vreg.register is None, "Register assigned twice"
                        assert len(unlive) > 0, "No free registers in the register pool"
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
