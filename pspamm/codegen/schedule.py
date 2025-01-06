from .ast import *
from .operands import *
from .forms import *

def isStore(instr):
    return isinstance(instr, StoreStmt) or (isinstance(instr, MovStmt) and isinstance(instr.dest, MemoryAddress))

def isScalar(instr):
    return isinstance(instr, AddStmt) or (isinstance(instr, MovStmt) and isinstance(instr.dest, Register) and instr.typ == AsmType.i64)

def isLoad(instr):
    return isinstance(instr, LoadStmt) or (isinstance(instr, MovStmt) and isinstance(instr.src, MemoryAddress)) or isinstance(instr, BcstStmt)

def hasDependency(instr1, instr2):
    ww = instr1.regs_out() & instr2.regs_out()
    wr = instr1.regs_out() & instr2.regs_in()
    rw = instr1.regs_in() & instr2.regs_out()
    return len(ww) > 0 or len(wr) > 0 or len(rw) > 0

def moveLoads(block, isLoop=False):
    has_subloops = any(isinstance(instr, Loop) for instr in block)

    preprocessed = []
    for instr in block:
        if isinstance(instr, Loop):
            if True:
                # only unroll the innermost loop at most
                prelude, inner, postlude = moveLoads(instr.body_contents.contents, True)
                if instr.final_val == 1:
                    preprocessed += prelude + postlude
                elif instr.final_val > 1:
                    preprocessed += prelude + [loop(instr.iteration_var, instr.final_val - 1, instr.unroll).body(*inner)] + postlude
            else:
                inner = moveLoads(instr.body_contents.contents, False)
                preprocessed += [loop(instr.iteration_var, instr.final_val, instr.unroll).body(*inner)]
        else:
            preprocessed += [instr]
    
    return moveLoadsBlock(preprocessed, isLoop)

def moveLoadsBlock(block, isLoop):
    reordered = []
    currentLoads = []
    previousLoad = []
    insertCounter = []

    # TODO: merge prune.py into this one here
    def addReorderedLoad(j):
        loadInstr = currentLoads.pop(j)
        prevLoadInstr = previousLoad.pop(j)
        insertCounter.pop(j)
        delta = 0
        if prevLoadInstr is not None:
            k = j
            for instr in reversed(currentLoads[:j]):
                k -= 1
                if instr is prevLoadInstr:
                    delta = addReorderedLoad(k)
                    break
        reordered.append(loadInstr)
        return 1 + delta
    def addDependentLoads(instr, i):
        j = 0
        while j < len(currentLoads):
            loadInstr = currentLoads[j]
            if hasDependency(instr, loadInstr):
                j -= addReorderedLoad(j)
            j += 1
    def preponeLoad(instr, i):
        maxI = None
        for loadInstr in reversed(currentLoads):
            if hasDependency(loadInstr, instr) or instr.barrier():
                maxI = loadInstr
                break
        previousLoad.append(maxI)
        currentLoads.append(instr)
        insertCounter.append(i)

    for i, instr in reversed(list(enumerate(block))):
        if isLoad(instr) or isScalar(instr):
            preponeLoad(instr, i)
        else:
            addDependentLoads(instr, i)
            reordered.append(instr)
    
    if isLoop:
        # pass again, but ignore loads
        postlude = list(reversed(reordered))
        reordered = []
        prelude = list(reversed(currentLoads))
        # TODO: tag loads to have a different iteration
        for i, instr in reversed(list(enumerate(block))):
            if isLoad(instr) or isScalar(instr):
                preponeLoad(instr, i + len(block))
            else:
                addDependentLoads(instr, i + len(block))
                reordered.append(instr)
        
        return prelude, list(reversed(reordered)), postlude
    else:
        for loadInstr in currentLoads:
            reordered.append(loadInstr)

        return list(reversed(reordered))
