def getBlocksize(m , n):

	bm = 8
	bn = 1
	maxval = 0

	for i in range(8, m+1, 8):
		for j in range(1, n+1):
			if KNL_condition(i, j):
				if i*j > maxval:
					maxval = i*j
					bm = i
					bn = j 

	return (bm, bn)


def KNL_condition(bm, bn):
    return (bn+1) * (bm / 8) <= 32