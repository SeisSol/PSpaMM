from enum import Enum

class Precision(Enum):
  DOUBLE = 8
  SINGLE = 4

  @classmethod
  def getCType(cls, precision):
    ctype = {cls.DOUBLE: 'double', cls.SINGLE: 'float'}
    return ctype[precision]

