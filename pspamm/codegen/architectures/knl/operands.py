from pspamm.codegen.operands import *


class Operand_KNL:
    @property
    def ugly(self):
        raise NotImplementedError()


# TODO: Rename this 'Immediate'
class Constant_KNL(Constant):

    @property
    def ugly(self):
        return "${}".format(self.value)


def c(n):
    """Sugar for conveniently defining integer constants"""
    return Constant_KNL(value=int(n))



class Label_KNL(Label):

    @property
    def ugly(self):
        #return self.ordinal
        return self.value.upper() + "_%="

def l(label: str):
    return Label_KNL(label)


class Register_KNL(Register):

    @property
    def ugly(self):
        return "%%" + self.value


class MemoryAddress_KNL(MemoryAddress):
    
    def __init__(self,
                 base: Register,
                 disp: int,
                 index: Register = None,
                 scaling: int = None) -> None:
        self.base = base
        self.disp = disp
        self.index = index
        self.scaling = scaling

    @property
    def ugly(self):
        if self.index is None:
            return "{}({})".format(self.disp,self.base.ugly)
        return "{}({},{},{})".format(self.disp,self.base.ugly,self.index.ugly,self.scaling)

    @property
    def clobbered(self):
        return self.base.clobbered

def mem(base, offset, index=None, scaling=None):
    return MemoryAddress_KNL(base, offset, index, scaling)


rax = Register_KNL(AsmType.i64, "rax")
rbx = Register_KNL(AsmType.i64, "rbx")
rcx = Register_KNL(AsmType.i64, "rcx")
rdx = Register_KNL(AsmType.i64, "rdx")
rdi = Register_KNL(AsmType.i64, "rdi")
rsi = Register_KNL(AsmType.i64, "rsi")

r   = lambda n: Register_KNL(AsmType.i64, "r"+str(n)) if n > 7 else gen_regs[n]
xmm = lambda n: Register_KNL(AsmType.f64x2, "xmm"+str(n))
ymm = lambda n: Register_KNL(AsmType.f64x4, "ymm"+str(n))
zmm = lambda n: Register_KNL(AsmType.f64x8, "zmm"+str(n))

