def getBlocksize(m , n, bk):

	bm = 4
	bn = 1
	maxval = 0

	for i in range(4, m+1, 4):
		for j in range(1, n+1):
			if HSW_condition(i, j, bk):
				if i*j > maxval:
					maxval = i*j
					bm = i
					bn = j 

	return (bm, bn)


def HSW_condition(bm, bn, bk):
	v_size = 4
	# ceiling division
	vm = -(bm // -v_size)
	return (bn + bk) * vm + bn * bk <= 16
