def getBlocksize(m, n, bk, v_size=2):
    # v_size default is 2, however for SVE that parameter will always be larger
    bm = 2
    bn = 1
    maxval = 0

    for i in range(2, m + 1, 1):
        next_multiple = i
        while next_multiple % v_size != 0:
            next_multiple += 1
        for j in range(1, n + 1):
            if ARM_condition(next_multiple, j, bk, v_size) and tileable(m, i):
                if i * j >= maxval:
                    maxval = i * j
                    bm = i
                    bn = j

    if maxval == 0:
        raise RuntimeError("Could not find an appropriate block size. We suggest padding the matrix dimensions")

    return (bm, bn)


def ARM_condition(bm, bn, bk, v_size):
    return (bn + bk) * (bm / v_size) + bn + 2 <= 32


def tileable(m, bm):
    return m % bm == 0
