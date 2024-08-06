def getBlocksize(m, n, bk, v_size=8):

	bm = v_size
	bn = 1

	for j in range(1, n+1):
		if KNL_condition(bm, j, bk, v_size):
			bn = j

	while KNL_condition(bm, bn, bk+1, v_size):
		bk += 1

	return (bm, bn, bk)


def KNL_condition(bm, bn, bk, v_size):
    # ceiling division
    vm = -(bm // -v_size)
    return (bn+bk) * vm <= 32
