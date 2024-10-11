from pspamm.codegen.operands import *


class Operand_ARM:
    @property
    def ugly(self):
        raise NotImplementedError()


# TODO: Rename this 'Immediate'
class Constant_ARM(Constant):

    @property
    def ugly(self):
        return "#{}".format(self.value)

    @property
    def ugly_large(self):
        return "={}".format(self.value)

    @property
    def ugly_lower16(self):
        return "#:lower16:{}".format(self.value)

    @property
    def ugly_upper16(self):
        return "#:upper16:{}".format(self.value)


def c(n):
    """Sugar for conveniently defining integer constants"""
    return Constant_ARM(value=int(n))


class Label_ARM(Label):

    @property
    def ugly(self):
        # return self.ordinal
        return self.value.upper() + "_%="


def l(label: str):
    return Label_ARM(label)


class Register_ARM(Register):
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
        return (self.value.split(".")[0]).replace("v", "d")
    
    @property
    def ugly_b32(self):
        return (self.value.split(".")[0]).replace("x", "w")


r = lambda n: Register_ARM(AsmType.i64, "x" + str(n))
xzr = Register_ARM(AsmType.i64, "xzr")
v = lambda n, prec: Register_ARM(AsmType.f64x8, "v" + str(n) + "." + prec)


class MemoryAddress_ARM(MemoryAddress):

    @property
    def ugly(self):
        return "[{}, {}]".format(self.base.ugly, self.disp)


def mem(base, offset):
    return MemoryAddress_ARM(base, offset)
