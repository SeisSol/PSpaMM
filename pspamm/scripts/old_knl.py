def getBlocksize(m , n, bk, v_size=8):

	bm = m
	bn = n
	
	if KNL_condition(bm, bn, bk, v_size):
		return (bm, bn)

	while not KNL_condition(bm, bn, bk, v_size):
		bm, bn = lowerToNextDiv(m, n, bm, bn, v_size)

	while KNL_condition(bm, bn, bk+1, v_size):
		bk += 1

	return (bm, bn)


def lowerToNextDiv(m, n, bm, bn, v_size):
	if bm > bn and bm > v_size:
		bm -= v_size
		while m % bm != 0:
			bm -= v_size
	else:
		bn -= 1
		while n % bn != 0:
			bn -= 1

	return bm, bn


def KNL_condition(bm, bn, bk, v_size):
	# ceiling division
	vm = -(bm // -v_size)
	return (bn+bk) * vm  <= 32
