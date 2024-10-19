def getBlocksize(m , n, bk, v_size, prec):

	bm = 8
	bn = 1
	maxval = 0

	for i in range(1, m+1):
		next_multiple = -(bm // -v_size)
		for j in range(1, n+1):
			if KNL_condition(next_multiple, j, bk, v_size) and tileable(m, bm):
				if i*j > maxval:
					maxval = i*j
					bm = i
					bn = j 
	
	while KNL_condition(bm, bn, bk+1, v_size):
		bk += 1

	return (bm, bn, bk)


def KNL_condition(bm, bn, bk, v_size):
    # ceiling division
    vm = -(bm // -v_size)
    return (bn+bk) * vm <= 32

def tileable(m, bm):
    return m % bm == 0

