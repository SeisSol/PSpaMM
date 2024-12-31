from .ast import *
from .operands import *
from .forms import *

def prune(block, toplevel=True):
    pruned = []
    cached = []

    for instr in block:
        if isinstance(instr, AddStmt) and isinstance(instr.src, Constant) and instr.additional is None:
            combinedValue = instr.src.value
            for i, cinstr in enumerate(cached):
                if cinstr.dest == instr.dest:
                    combinedValue += cinstr.src.value
                    cached.pop(i)
                    break
            cached += [add(combinedValue, instr.dest)]
        else:
            pruned += cached
            cached = []
            if isinstance(instr, Loop):
                instr.body_contents.contents = prune(instr.body_contents.contents, False)
            pruned += [instr]
    
    if not toplevel:
        pruned += cached
    return pruned
