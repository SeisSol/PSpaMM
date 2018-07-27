from codegen.operands import *


class Operand_ARM:
    @property
    def ugly(self):
        raise NotImplementedError()


# TODO: Rename this 'Immediate'
class Constant_ARM(Constant):

    @property
    def ugly(self):
        return f"#{self.value}"


def c(n):
    """Sugar for conveniently defining integer constants"""
    return Constant_ARM(value=int(n))



class Label_ARM(Label):

    @property
    def ugly(self):
        #return self.ordinal
        return self.value.upper() + "_%="

def l(label: str):
    return Label_ARM(label)


class Register_ARM(Register):
    @property
    def ugly(self):
        return self.value

    @property
    def clobbered(self):
        return (self.value.split(".")[0]).replace("x", "r")

    @property
    def ugly_scalar(self):
        return (self.value.split(".")[0]).replace("v", "q")

    @property
    def ugly_scalar_1d(self):
        return (self.value.split(".")[0]).replace("v", "d")
    @property

    def ugly_1d(self):
        return self.value.replace("2d", "1d")

r   = lambda n: Register_ARM(AsmType.i64, "x"+str(n))
xzr = Register_ARM(AsmType.i64, "xzr")
ymm = lambda n: Register_ARM(AsmType.f64x4, "ymm"+str(n))
zmm = lambda n: Register_ARM(AsmType.f64x8, "v"+str(n) + ".2d")




class MemoryAddress_ARM(MemoryAddress):

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
            return f"[{self.base.ugly}, {self.disp}]"

    def __repr__(self):
        return "MemoryAddress: " + self.nice


def mem(base, index, scale, offset):
    return MemoryAddress_ARM(base, index, scale, offset)






