def getBlocksize(m , n, bk, v_size, prec):

	bm = 2
	bn = 1
	maxval = 0

	for i in range(v_size, m+1, v_size):
		for j in range(1, n+1):
			if ARM_condition(i, j, bk, v_size):
				if i*j > maxval:
					maxval = i*j
					bm = i
					bn = j

	while ARM_condition(bm, bn, bk+1, v_size):
		bk += 1

	return (bm, bn, bk)


def ARM_condition(bm, bn, bk, v_size):
  # ceiling division
  vm = -(bm // -v_size)
  return (bn+bk) * vm + bn*bk <= 32
