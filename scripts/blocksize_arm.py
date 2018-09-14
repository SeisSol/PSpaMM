#!/usr/bin/env python


import sys

m=int(sys.argv[1])
n=int(sys.argv[2])
bm=int(sys.argv[3])
bn=int(sys.argv[4])

def lowerToNextDiv(m, n, bm, bn):
	if bm > bn:
		bm -= 2
	else:
		bn -= 1

	return bm, bn



def condition(bm, bn):
	return (bn+1) * (bm / 2) + bn <= 32

bm, bn = lowerToNextDiv(m,n,bm,bn)

while not condition(bm, bn):
	bm, bn = lowerToNextDiv(m, n, bm, bn)

print(str(bm) + "-" + str(bn))

