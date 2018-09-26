def getBlocksize(m , n):

	bm = 8
	bn = 1

	for j in range(1, n+1):
		if KNL_condition(bm, j):
			bn = j

	return (bm, bn)


def KNL_condition(bm, bn):
    return (bn+1) * (bm / 8) <= 32