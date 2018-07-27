
from cursors import *
from codegen.operands import *

def C_mask(C_regs: Matrix[Register],
           C: CursorDef,
           C_ptr: CursorLocation,
           C_block_offset: Coords,
           B: CursorDef,
           B_ptr: CursorLocation,
           B_block_offset: Coords,
           v_size: int,
           tiled: bool = True
          ) -> Matrix[bool]:

    Vr, Vc = C_regs.shape
    mask = Matrix.full(Vr, Vc, False)
    C_br, C_bc, C_idx, C_pat = C.get_block(C_ptr, C_block_offset)
    B_br, B_bc, B_idx, B_pat = B.get_block(B_ptr, B_block_offset)

    assert(Vr*v_size == C_br)   # bm must tile m exactly for now
    assert(Vc >= C_bc)     # Matrix block must fit in register block
    assert(C_bc == B_bc)   # Matrix blocks are compatible

    # Mask out registers not used in current block, including zero-cols of B
    if tiled:
        for Vci in range(C_bc):
            if B_pat[:,Vci].any(axis=0):
                mask[:,Vci] = True
    else:
        mask[:, :C_bc] = True


    return mask


def C_mask_untiled(C_regs: Matrix[Register],
           C: CursorDef,
           C_ptr: CursorLocation,
           C_block_offset: Coords,
           v_size: int
          ) -> Matrix[bool]:

    Vr, Vc = C_regs.shape
    mask = Matrix.full(Vr, Vc, False)
    C_br, C_bc, C_idx, C_pat = C.get_block(C_ptr, C_block_offset)

    assert(Vr*v_size == C_br)   # bm must tile m exactly for now
    assert(Vc >= C_bc)     # Matrix block must fit in register block

    mask[:, :C_bc] = True
    return mask



def A_mask(A_regs: Matrix[Register],
           A: CursorDef,
           A_ptr: CursorLocation,
           A_block_offset: Coords,
           B: CursorDef,
           B_ptr: CursorLocation,
           B_block_offset: Coords,
           v_size: int
          ) -> Matrix[bool]:

    Vr, Vc = A_regs.shape
    mask = Matrix.full(Vr, Vc, False)
    A_br, A_bc, A_idx, A_pat = A.get_block(A_ptr, A_block_offset)
    B_br, B_bc, B_idx, B_pat = B.get_block(B_ptr, B_block_offset)

    assert(Vr*v_size == A_br)   # bm must tile m exactly for now
    assert(Vc >= A_bc)     # Matrix block must fit in register block
    assert(A_bc == B_br)   # Matrix blocks are compatible

    # Mask out registers not used in current block, including zero-rows of B
    for Vci in range(A_bc):
        if B_pat[Vci,:].any(axis=1):
            mask[:,Vci] = True

    return mask
