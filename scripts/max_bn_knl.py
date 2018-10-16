def getBlocksize(m, n, bk):

	bm = 8
	bn = 1

	for j in range(1, n+1):
		if KNL_condition(bm, j, bk):
			bn = j

	return (bm, bn)


def KNL_condition(bm, bn, bk):
    return (bn+bk) * (bm / 8) <= 32
