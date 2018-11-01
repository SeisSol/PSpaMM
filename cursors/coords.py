from collections import namedtuple

# The relationship between blocks and cells has become too complicated to
# safely bake into a coordinate system. The semantics has been moved to the
# cursor movement commands. A NewCoords object may represent a logical cell,
# a logical block start, or a physical block start depending on context.
# We are including a {relative|absolute} flag in order to reduce the number of methods.

C = namedtuple('C', 'down right absolute')
C.__new__.__defaults__ = (0, 0, False)

class Coords(C):

    def copy(self):
        return Coords(self.down, self.right, self.absolute)
    
    def __add__(self, other):
        absolute = self.absolute | other.absolute
        return Coords(self.down+other.down, self.right+other.right, absolute)

    def __sub__(self, other):
        absolute = self.absolute != other.absolute  # TODO: What is the math behind this?
        return Coords(self.down-other.down, self.right-other.right, absolute)

    def __neg__(self, other):
        return Coords(-self.down, -self.right, self.absolute)

    def __eq__(self, other):
        return self.down == other.down and \
               self.right == other.right and \
               self.absolute == other.absolute

    def __repr__(self):
        if self.absolute:
            absolute = ", absolute"
        else:
            absolute = ""
        return "(d={},r={}{})".format(self.down, self.right,absolute) 