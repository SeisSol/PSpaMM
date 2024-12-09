from pspamm.codegen.operands import *


class Operand_HSW:
    @property
    def ugly(self):
        raise NotImplementedError()


# TODO: Rename this 'Immediate'
class Constant_HSW(Constant):

    @property
    def ugly(self):
        return "${}".format(self.value)


def c(n):
    """Sugar for conveniently defining integer constants"""
    return Constant_HSW(value=int(n))



class Label_HSW(Label):

    @property
    def ugly(self):
        #return self.ordinal
        return self.value.upper() + "_%="

def l(label: str):
    return Label_HSW(label)


class Register_HSW(Register):

    @property
    def ugly(self):
        return "%%" + self.value
    
    @property
    def ugly_xmm(self):
        return "%%x" + self.value[1:]


rax = Register_HSW(AsmType.i64, "rax")
rbx = Register_HSW(AsmType.i64, "rbx")
rcx = Register_HSW(AsmType.i64, "rcx")
rdx = Register_HSW(AsmType.i64, "rdx")
rdi = Register_HSW(AsmType.i64, "rdi")
rsi = Register_HSW(AsmType.i64, "rsi")

r   = lambda n: Register_HSW(AsmType.i64, "r"+str(n)) if n > 7 else gen_regs[n]
xmm = lambda n: Register_HSW(AsmType.f64x2, "xmm"+str(n))
ymm = lambda n: Register_HSW(AsmType.f64x4, "ymm"+str(n))




class MemoryAddress_HSW(MemoryAddress):
    
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

def mem(base, offset, index=None, scaling=None):
    return MemoryAddress_HSW(base, offset, index, scaling)






