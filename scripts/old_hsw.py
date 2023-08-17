def getBlocksize(m , n, bk):

	bm = m
	bn = n
	
	if HSW_condition(bm, bn, bk):
		return (bm, bn)

	while not HSW_condition(bm, bn, bk):
		bm, bn = lowerToNextDiv(m, n, bm, bn)

	return (bm, bn)


def lowerToNextDiv(m, n, bm, bn):
	if bm > bn and bm > 4:
		bm -= 4
		while m % bm != 0:
			bm -= 4
	else:
		bn -= 1
		while n % bn != 0:
			bn -= 1

	return bm, bn


def HSW_condition(bm, bn, bk):
	v_size = 4
	return (bn + bk) * (bm / v_size) + bn * bk + 2 <= 16
