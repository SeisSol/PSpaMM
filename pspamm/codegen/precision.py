from enum import Enum

class Precision(Enum):
  DOUBLE = 8
  SINGLE = 4
  HALF = 2
  BFLOAT16 = 2.1

  @classmethod
  def getCType(cls, precision):
    ctype = {cls.DOUBLE: 'double', cls.SINGLE: 'float', cls.HALF: 'uint16_t', cls.BFLOAT16: 'uint16_t'}
    return ctype[precision]
  
  def ctype(self):
    return self.getCType(self)

  def size(self):
    return {
      self.DOUBLE: 8,
      self.SINGLE: 4,
      self.HALF: 2,
      self.BFLOAT16: 2
    }[self]
    raise NotImplementedError()
  
  def __repr__(self):
    return self.getCType(self)

  def __str__(self):
    return self.getCType(self)

