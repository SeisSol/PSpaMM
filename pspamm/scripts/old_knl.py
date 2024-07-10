def getBlocksize(m , n, bk):

	bm = m
	bn = n
	
	if KNL_condition(bm, bn, bk):
		return (bm, bn)

	while not KNL_condition(bm, bn, bk):
		bm, bn = lowerToNextDiv(m, n, bm, bn)

	return (bm, bn)


def lowerToNextDiv(m, n, bm, bn):
	if bm > bn and bm > 8:
		bm -= 8
		while m % bm != 0:
			bm -= 8
	else:
		bn -= 1
		while n % bn != 0:
			bn -= 1

	return bm, bn


def KNL_condition(bm, bn, bk):
	return (bn+bk) * (bm / 8)  <= 32
