from typing import NamedTuple

# The relationship between blocks and cells has become too complicated to
# safely bake into a coordinate system. The semantics has been moved to the
# cursor movement commands. A NewCoords object may represent a logical cell,
# a logical block start, or a physical block start depending on context.
# We are including a {relative|absolute} flag in order to reduce the number of methods.


class Coords(NamedTuple):
    down:     int  = 0
    right:    int  = 0
    absolute: bool = False

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
        return f"(d={self.down},r={self.right}{absolute})"




class OldCoords:
    """ Coords is a container for relative logical matrix coordinates.
    """
    def __init__(self, down:int=0, right:int=0, units:str="cells") -> None:
        self.down_cells = 0
        self.right_cells = 0
        self.down_vecs = 0
        self.right_vecs = 0
        self.down_blocks = 0
        self.right_blocks = 0
        self.update(down,right,units)


    def go(self, down:int=0, right:int=0, units:str="cells"):
        return self + OldCoords(down, right, units)


    def update(self, down:int=0, right:int=0, units:str="cells"):
        if units=="cells":
            self.down_cells += down
            self.right_cells += right

        elif units=="vectors":
            self.down_vecs += down
            self.right_vecs += right

        elif units=="blocks":
            self.down_blocks += down
            self.right_blocks += right

        else:
            raise Exception("Units must be cells, vectors, or blocks")


    def __add__(self, other:"OldCoords"):
        result = OldCoords()
        result.down_cells += self.down_cells + other.down_cells
        result.right_cells += self.right_cells + other.right_cells
        result.down_vecs += self.down_vecs + other.down_vecs
        result.right_vecs += self.right_vecs + other.right_vecs
        result.down_blocks += self.down_blocks + other.down_blocks
        result.right_blocks += self.right_blocks + other.right_blocks
        return result

    def __neg__(self):
        result = OldCoords()
        result.down_cells =  -self.down_cells
        result.right_cells = -self.right_cells
        result.down_vecs = -self.down_vecs
        result.right_vecs = -self.right_vecs
        result.down_blocks = -self.down_blocks
        result.right_blocks = -self.right_blocks
        return result

    def __sub__(lhs,rhs):
        result = OldCoords()
        result.down_cells =  lhs.down_cells - rhs.down_cells
        result.right_cells = lhs.right_cells - rhs.right_cells
        result.down_vecs = lhs.down_vecs - rhs.down_vecs
        result.right_vecs = lhs.right_vecs - rhs.right_vecs
        result.down_blocks = lhs.down_blocks - rhs.down_blocks
        result.right_blocks = lhs.right_blocks - rhs.right_blocks
        return result

    def __eq__(self, other):
        return self.down_cells == other.down_cells and \
            self.down_vecs == other.down_vecs and \
            self.down_blocks == other.down_blocks and \
            self.right_cells == other.right_cells and \
            self.right_vecs == other.right_vecs and \
            self.right_blocks == other.right_blocks

    def __repr__(self):

        down = []
        if self.down_blocks != 0:
            down.append(f"{self.down_blocks}b")
        if self.down_vecs != 0:
            down.append(f"{self.down_vecs}v")
        if self.down_cells != 0:
            down.append(f"{self.down_cells}c")

        right = []
        if self.right_blocks != 0:
            right.append(f"{self.right_blocks}b")
        if self.right_vecs != 0:
            right.append(f"{self.right_vecs}v")
        if self.right_cells != 0:
            right.append(f"{self.right_cells}c")

        downstr = ",".join(down)
        rightstr = ",".join(right)
        return f"(down={downstr}; right={rightstr})"


down1block = OldCoords(down=1,units="blocks")
