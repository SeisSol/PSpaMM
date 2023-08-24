def getBlocksize(m, n, bk, v_size=4):

	bm = v_size
	bn = 1

	for j in range(1, n+1):
		if HSW_condition(bm, j, bk, v_size):
			bn = j

	return (bm, bn)


def HSW_condition(bm, bn, bk, v_size):
    return (bn + bk) * (bm / v_size) + bn * bk + 2 <= 16
