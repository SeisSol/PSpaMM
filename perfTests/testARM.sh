#TEST 1

M=8
N=56
K=56
LDA=8
LDB_DENSE=56
LDB_SPARSE=0
LDC=8
BETA=0
BK=1
MTX="mtx/56x56.mtx"
NNZ=294
ITER=131072


BM=$(($M+2))
BN=$N

while [ $BN -gt 1 ] || [ $BM -gt 1 ]; do

	BS=$(./../scripts/blocksize_arm.py $M $N $BM $BN)

	BM=$(echo $BS | cut -f1 -d-)
	BN=$(echo $BS | cut -f2 -d-)

	> arm/gemm_sparse.h
	> arm/gemm_dense.h

	python3.6 ../sparsemmgen.py $M $N $K $LDA $LDB_SPARSE $LDC $BETA --bm $BM --bn $BN --bk $BK --arch arm --output_funcname gemm_sparse --output_filename arm/gemm_sparse.h --mtx_filename $MTX
	python3.6 ../sparsemmgen.py $M $N $K $LDA $LDB_DENSE $LDC $BETA --bm $BM --bn $BN --bk $BK --arch arm --output_funcname gemm_dense --output_filename arm/gemm_dense.h

	g++ -std=c++11 -fopenmp testARM.cpp

	export OMP_NUM_THREADS=4

	./a.out $M $N $K $BETA $NNZ $ITER $MTX

	printf " "
	printf $BM
	printf " "
	printf $BN
done
