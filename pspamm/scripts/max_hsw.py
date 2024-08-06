def getBlocksize(m , n, bk, v_size=4):

	bm = 4
	bn = 1
	maxval = 0

	for i in range(v_size, m+1, v_size):
		for j in range(1, n+1):
			# can be replaced by HSW_condition_extended here
			if HSW_condition(i, j, bk, v_size):
				if i*j > maxval and (HSW_condition(i, j, bk, v_size) or j > 1):
					maxval = i*j
					bm = i
					bn = j 

	while HSW_condition(bm, bn, bk+1, v_size):
		bk += 1

	return (bm, bn, bk)

def HSW_condition(bm, bn, bk, v_size):
	# ceiling division
	vm = -(bm // -v_size)
	return (bn + bk) * vm + bn * bk <= 16

def HSW_condition_extended(bm, bn, bk, v_size):
	# ceiling division
	vm = -(bm // -v_size)
	return bn * vm + bn * bk + 1 <= 16
