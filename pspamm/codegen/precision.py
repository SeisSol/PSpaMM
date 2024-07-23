from enum import Enum

class Precision(Enum):
  DOUBLE = 8
  SINGLE = 4
  HALF = 2

  @classmethod
  def getCType(cls, precision):
    ctype = {cls.DOUBLE: 'double', cls.SINGLE: 'float', cls.HALF: 'half'}
    return ctype[precision]

