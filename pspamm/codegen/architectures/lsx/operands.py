from pspamm.codegen.operands import *


class Operand_LSX:
    @property
    def ugly(self):
        raise NotImplementedError()


# TODO: Rename this 'Immediate'
class Constant_LSX(Constant):

    @property
    def ugly(self):
        return f"{self.value}"


def c(n):
    """Sugar for conveniently defining integer constants"""
    return Constant_LSX(value=int(n))



class Label_LSX(Label):

    @property
    def ugly(self):
        #return self.ordinal
        return self.value.upper() + "_%="

def l(label: str):
    return Label_LSX(label)


class Register_LSX(Register):

    @property
    def ugly(self):
        return "$" + self.value

r   = lambda n: Register_LSX(AsmType.i64, "r"+str(n))
vr = lambda n: Register_LSX(AsmType.f64x2, "vr"+str(n))
xr = lambda n: Register_LSX(AsmType.f64x4, "xr"+str(n))




class MemoryAddress_LSX(MemoryAddress):
    
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
        #if self.index is None:
        #    return f"{self.disp}({self.base.ugly})"
        #return f"{self.disp}({self.base.ugly},{self.index.ugly},{self.scaling})"
        return f"{self.base.ugly},{self.disp}"
    
    def registers(self):
        return [self.base, self.index]

def mem(base, offset, index=None, scaling=None):
    return MemoryAddress_LSX(base, offset, index, scaling)






