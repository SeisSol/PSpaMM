from enum import Enum
from typing import List, Dict


AsmType = Enum('AsmType', ['unknown','i8','i16','i32','i64','f32','f64',
                           'f32x4','f32x8','f32x16','f64x2','f64x4','f64x8',
                           'p64x8'])

class Operand:
    @property
    def ugly(self):
        raise NotImplementedError()


# TODO: Rename this 'Immediate'
class Constant(Operand):
    def __init__(self, value:int) -> None:
        self.value = value

    @property
    def ugly(self):
        raise NotImplementedError()

class Label(Operand):
    _interns = {}
    _last = -1
    def __init__(self, value) -> None:
        assert(isinstance(value, str))
        self.value = value
        if value in Label._interns:
            self.ordinal = Label._interns[value]
        else:
            self.ordinal = Label._last + 1
            Label._last += 1
            Label._interns[value] = self.ordinal

    @property
    def ugly(self):
        raise NotImplementedError()

class Register(Operand):

    def __init__(self, typeinfo, value) -> None:
        self.typeinfo = typeinfo
        self.value = str(value)
    
    def size(self):
        if self.typeinfo == AsmType.i8:
            return 1
        if self.typeinfo == AsmType.i16:
            return 2
        if self.typeinfo == AsmType.i32:
            return 4
        if self.typeinfo == AsmType.i64:
            return 8
        if self.typeinfo == AsmType.f32:
            return 4
        if self.typeinfo == AsmType.f64:
            return 8
        if self.typeinfo == AsmType.f32x4:
            return 16
        if self.typeinfo == AsmType.f32x8:
            return 32
        if self.typeinfo == AsmType.f32x16:
            return 64
        if self.typeinfo == AsmType.f64x2:
            return 16
        if self.typeinfo == AsmType.f64x4:
            return 32
        if self.typeinfo == AsmType.f64x8:
            return 64

    @property
    def ugly(self):
        raise NotImplementedError()

    @property
    def clobbered(self):
        return self.value

class MemoryAddress(Operand):

    def __init__(self,
                 base,
                 disp) -> None:
        self.base = base
        self.disp = disp

    @property
    def ugly(self):
        raise NotImplementedError()
