def getBlocksize(m , n):

	bm = m
	bn = n

	if ARM_condition(bm, bn):
		return (bm, bn)

	while not ARM_condition(bm, bn):
		bm, bn = lowerToNextDiv(m, n, bm, bn)

	return (bm, bn)


def lowerToNextDiv(m, n, bm, bn):
	if bm > bn and bm > 2:
		bm -= 2
		while m % bm != 0:
			bm -= 2
	else:
		bn -= 1
		while n % bn != 0:
			bn -= 1

	return bm, bn

def ARM_condition(bm, bn):
	return (bn+1) * (bm / 2) + bn <= 32