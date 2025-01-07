from .ast import *
from .operands import *
from .forms import *

def isStore(instr):
    return isinstance(instr, StoreStmt) or (isinstance(instr, MovStmt) and isinstance(instr.dest, MemoryAddress))

def isScalar(instr):
    return isinstance(instr, AddStmt) or (isinstance(instr, MovStmt) and isinstance(instr.dest, Register) and instr.typ == AsmType.i64)

def isLoad(instr):
    return isinstance(instr, LoadStmt) or (isinstance(instr, MovStmt) and isinstance(instr.src, MemoryAddress)) or isinstance(instr, BcstStmt)

def hasDependency(instr1, instr2, rrt=False):
    ww = instr1.regs_out() & instr2.regs_out()
    wr = instr1.regs_out() & instr2.regs_in()
    rw = instr1.regs_in() & instr2.regs_out()
    rr = instr1.regs_in() & instr2.regs_in()
    return len(ww) > 0 or len(wr) > 0 or len(rw) > 0 or (rrt and len(rr) > 0)

def moveLoads(block, isLoop=False):
    preprocessed = []
    for instr in block:
        if isinstance(instr, Loop):
            if instr.may_overlap:
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
        prevLoadInstrs = previousLoad.pop(j)
        insertCounter.pop(j)
        delta = 0
        k = j
        while k > 0:
            k -= 1
            instr = currentLoads[k]
            for prevLoadInstr in prevLoadInstrs:
                if instr is prevLoadInstr:
                    newdelta = addReorderedLoad(k)
                    k -= newdelta
                    delta += newdelta
        reordered.append(loadInstr)
        return 1 + delta
    def addDependentLoads(instr, i):
        j = 0
        while j < len(currentLoads):
            loadInstr = currentLoads[j]
            # for now, include read-read dependencies here
            if hasDependency(instr, loadInstr): # or too far away insertCounter[j] < i + 4
                j -= addReorderedLoad(j)
            j += 1
    def preponeLoad(instr, i):
        maxI = []
        for loadInstr in reversed(currentLoads):
            if hasDependency(loadInstr, instr) or instr.barrier():
                maxI.append(loadInstr)
                # TODO: use a more imaginative dependency tracking here? (e.g. by register, write-write-breaking and so on)
                # right now, it's order-enforcing (which should be enough however)
        previousLoad.append(maxI)
        currentLoads.append(instr)
        insertCounter.append(i)

    for i, instr in reversed(list(enumerate(block))):
        if isLoad(instr) or isScalar(instr):
            preponeLoad(instr, len(reordered))
        else:
            addDependentLoads(instr, len(reordered))
            reordered.append(instr)
    
    if isLoop:
        # pass again, but ignore loads
        postlude = list(reversed(reordered))
        reordered = []
        prelude = list(reversed(currentLoads))

        # TODO: tag loads to have a different iteration?
        for i, instr in reversed(list(enumerate(block))):
            if isLoad(instr) or isScalar(instr):
                preponeLoad(instr, len(reordered) + len(postlude))
            else:
                addDependentLoads(instr, len(reordered) + len(postlude))
                reordered.append(instr)
        
        # add loads/scalar instructions that would be materialized over 2 iterations only (?)
        for i,instr in enumerate(currentLoads):
            if i + len(prelude) >= len(currentLoads):
                break
            reordered.append(instr)
        
        return prelude, list(reversed(reordered)), postlude
    else:
        for loadInstr in currentLoads:
            reordered.append(loadInstr)

        return list(reversed(reordered))
