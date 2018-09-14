#!/usr/bin/env python


import sys

m=int(sys.argv[1])
n=int(sys.argv[2])

def lowerToNextDiv(m, n, bm, bn):
	if bm > bn and bm > v:
		bm -= v
	else:
		bn -= 1

	return bm, bn



def condition(bm, bn):
	return (bn+1) * (bm / 2) + bn <= 32

bm = 96
bn = 100

while not ARM_condition(bm, bn):
	bm, bn = lowerToNextDiv(m, n, bm, bn)

print(str(bm) + "-" + str(bn))

