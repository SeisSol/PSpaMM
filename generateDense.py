#!/usr/bin/env python


import sys

m=int(sys.argv[1])
n=int(sys.argv[2])
ld=int(sys.argv[3])

print('%%MatrixMarket matrix coordinate integer general')
print('%')
print('{} {} {}'.format(ld,n,n*m))
for col in range(n):
  for row in range(m):
    print('{} {} {}'.format(row+1, col+1, 1))
