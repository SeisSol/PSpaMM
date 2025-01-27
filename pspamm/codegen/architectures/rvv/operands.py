from pspamm.codegen.operands import *


class Operand_RV:
    @property
    def ugly(self):
        raise NotImplementedError()


class Constant_RV(Constant):
    @property
    def ugly(self):
        return str(self.value)

def c(n):
    """Sugar for conveniently defining integer constants"""
    return Constant_RV(value=int(n))


class Label_RV(Label):
    @property
    def ugly(self):
        return self.value.upper() + "_%="


def l(label: str):
    return Label_RV(label)


class Register_RV(Register):
    @property
    def ugly(self):
        return self.value

    @property
    def ugly_precision(self):
        return self.value.split(".")[1]

    @property
    def ugly_lsl_shift(self):
        return {
            "d": 3,
            "s": 2,
            "h": 1
        }[self.ugly_precision]

    @property
    def clobbered(self):
        # removed [this comment should stay here for now---in case there's some compiler expecting it]: .replace("x", "r")
        return (self.value.split(".")[0])

    @property
    def ugly_scalar(self):
        return (self.value.split(".")[0]).replace("v", "q")

    @property
    def ugly_scalar_1d(self):
        #turns "Vn.2d" into "Dn"
        return (self.value.split(".")[0]).replace("v", "d")


x = lambda n: Register_RV(AsmType.i64, "x" + str(n))
f = lambda n: Register_RV(AsmType.f64, "f" + str(n))
v = lambda n: Register_RV(AsmType.f64x8, "v" + str(n))

class MemoryAddress_RV(MemoryAddress):
    @property
    def ugly(self):
        if self.disp == 0:
            return f'({self.base.ugly})'
        else:
            return f'{self.disp}({self.base.ugly})'

    @property
    def clobbered(self):
        return self.base

    @property
    def ugly_base(self):
        return str(self.base.ugly)

    @property
    def ugly_offset(self):
        return str(self.disp)


def mem(base, offset):
    return MemoryAddress_RV(base, offset)
