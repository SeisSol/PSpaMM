from pspamm.codegen.operands import *


class Operand_ARM:
    @property
    def ugly(self):
        raise NotImplementedError()


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
        #turns "Vn.2d" into "Dn"
        return (self.value.split(".")[0]).replace("v", "d")


r = lambda n: Register_ARM(AsmType.i64, "x" + str(n))
xzr = Register_ARM(AsmType.i64, "xzr")
z = lambda n, prec: Register_ARM(AsmType.f64x8, "z" + str(n) + "." + prec)
p = lambda n: Register_ARM(AsmType.i64, "p" + str(n))

class MemoryAddress_ARM(MemoryAddress):
    @property
    def ugly(self):
        # from the specifications:
        # Contiguous load of doublewords to elements of a vector register from the memory address generated by
        # a 64-bit scalar base and immediate index in the range -8 to 7 which is multiplied by the vector's
        # in-memory size, irrespective of predication, and added to the base address. Inactive elements will not
        # not cause a read from Device memory or signal a fault, and are set to zero in the destination vector.
        # MUL VL should multiply 64 on top of the immediate offset?
        return "[{}, {}, MUL VL]".format(self.base.ugly, self.disp)

    @property
    def clobbered(self):
        return self.base

    @property
    def ugly_no_vl_scaling(self):
        return "[{}, {}]".format(self.base.ugly, self.disp)

    @property
    def ugly_base(self):
        return "{}".format(self.base.ugly)

    @property
    def ugly_offset(self):
        # TODO: is this already dynamic? -> if precision is single, we need LSL #2
        return "{}".format(self.disp)


def mem(base, offset):
    return MemoryAddress_ARM(base, offset)
