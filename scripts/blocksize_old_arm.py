#!/usr/bin/env python

import sys

m=int(sys.argv[1])
n=int(sys.argv[2])

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



def KNL_condition(bm, bn):
	return (bn+1) * (bm / 2) <= 32
bm = 96
bn = 100

while not KNL_condition(bm, bn):
	bm, bn = lowerToNextDiv(m, n, bm, bn)

print(str(bm) + "-" + str(bn))

