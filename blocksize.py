#!/usr/bin/env python


import sys

m=int(sys.argv[1])
n=int(sys.argv[2])
v=int(sys.argv[3])

def lowerToNextDiv(m, n, bm, bn):
	if bm > bn and bm > v:
		bm -= v
		while m % bm != 0:
			bm -= v
	else:
		bn -= 1
		while n % bn != 0:
			bn -= 1

	return bm, bn



def ARM_condition(bm, bn):
	return (bn+1) * (bm / v) + bn <= 32

bm = 96
bn = 100

while not ARM_condition(bm, bn):
	bm, bn = lowerToNextDiv(m, n, bm, bn)

print(str(bm) + "-" + str(bn))

