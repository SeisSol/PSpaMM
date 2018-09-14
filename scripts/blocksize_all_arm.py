#!/usr/bin/env python


import sys

m=int(sys.argv[1])
n=int(sys.argv[2])
bm=int(sys.argv[3])
bn=int(sys.argv[4])


def condition(bm, bn):
    return (bn+1) * (bm / 2) + bn <= 32

bm -= 2

if bm == 0:
    bn -=1
    bm = m

while not condition(bm, bn) and bn > 0:

    while not condition(bm, bn) and bm > 0 and bn > 0:
        bm -= 2

    if bm == 0:
        bn -= 1
        bm = m

print(str(bm) + "-" + str(bn))

