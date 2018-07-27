from codegen.operands import *


class Operand_KNL:
    @property
    def ugly(self):
        raise NotImplementedError()


# TODO: Rename this 'Immediate'
class Constant_KNL(Constant):

    @property
    def ugly(self):
        return f"${self.value}"


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


rdx = Register_KNL(AsmType.i64, "rdx")
rdi = Register_KNL(AsmType.i64, "rdi")
rsi = Register_KNL(AsmType.i64, "rsi")

r   = lambda n: Register_KNL(AsmType.i64, "r"+str(n))
ymm = lambda n: Register_KNL(AsmType.f64x4, "ymm"+str(n))
zmm = lambda n: Register_KNL(AsmType.f64x8, "zmm"+str(n))




class MemoryAddress_KNL(MemoryAddress):

    def __init__(self,
                 base: Register,
                 index: Register,
                 scale: int,
                 disp: int) -> None:
        self.base = base
        self.index = index
        self.scale = scale
        self.disp = disp

    @property
    def ugly(self):
        if self.index is not None and self.scale is not None:
            return f"{self.disp}({self.base.ugly}, {self.index.ugly}, {self.scale})"
        else:
            return f"{self.disp}({self.base.ugly})"

    def __repr__(self):
        return "MemoryAddress: " + self.nice


def mem(base, index, scale, offset):
    return MemoryAddress_KNL(base, index, scale, offset)






