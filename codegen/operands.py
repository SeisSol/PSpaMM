from enum import Enum
from typing import List, Dict


AsmType = Enum('AsmType', ['unknown','i8','i16','i32','i64','f32','f64',
                           'f32x4','f32x8','f32x16','f64x2','f64x4','f64x8'])

AsmType.__doc__ = """Enum of different concrete types, the useful subset of
    the cross product {Int,Float} x {size} x {vector length}.
    Each Operand has exactly one concrete type."""

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
    _interns: Dict[str, int] = {}
    _last: int = -1
    def __init__(self, value: str) -> None:
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

    def __init__(self, typeinfo:AsmType, value:str) -> None:
        self.typeinfo = typeinfo
        self.value = str(value)

    @property
    def ugly(self):
        raise NotImplementedError()

    @property
    def clobbered(self):
        return self.value

class MemoryAddress(Operand):

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
        raise NotImplementedError()
