def getBlocksize(m, n, bk, v_size=8):

	bm = v_size
	bn = 1

	for j in range(1, n+1):
		if KNL_condition(bm, j, bk, v_size):
			bn = j

	return (bm, bn)


def KNL_condition(bm, bn, bk, v_size):
    return (bn+bk) * (bm / v_size) + 2 <= 32
