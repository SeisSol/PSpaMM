#!/usr/bin/env python


import sys

m=int(sys.argv[1])
n=int(sys.argv[2])


def condition(bm, bn):
    return (bn+1) * (bm / 2) + bn <= 32

bm = 2
bn = 1

maxval = 0

for i in xrange(2, m+1, 2):
	for j in range(1, n+1):
		if condition(i, j):
			if i*j > maxval:
				maxval = i*j
				bm = i
				bn = j 


print(str(bm) + "-" + str(bn))

