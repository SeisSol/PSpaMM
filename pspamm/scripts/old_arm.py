def getBlocksize(m , n, bk, v_size=2):

	bm = m
	bn = n
	
	if ARM_condition(bm, bn, bk, v_size):
		return (bm, bn)

	while not ARM_condition(bm, bn, bk, v_size):
		bm, bn = lowerToNextDiv(m, n, bm, bn, v_size)

	while ARM_condition(bm, bn, bk+1, v_size):
		bk += 1

	return (bm, bn, bk)


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


def ARM_condition(bm, bn, bk, v_size):
  # ceiling division
  vm = -(bm // -v_size)
  return (bn+bk) * vm + bn <= 32
