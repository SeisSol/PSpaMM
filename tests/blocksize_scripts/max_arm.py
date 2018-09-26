def getBlocksize(m , n):

	bm = 2
	bn = 1
	maxval = 0

	for i in range(2, m+1, 2):
		for j in range(1, n+1):
			if ARM_condition(i, j):
				if i*j > maxval:
					maxval = i*j
					bm = i
					bn = j

	return (bm, bn)


def ARM_condition(bm, bn):
    return (bn+1) * (bm / 2) + bn <= 32