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
MTXCSC="mtx/56x56csc.mtx"
NNZ=294
ITER=12400000

BM=$(($M+8))
BN=$N

while [ $BN -gt 1 ] || [ $BM -gt 1 ]; do

	BS=$(./../scripts/blocksize_all_knl.py $M $N $BM $BN)

	BM=$(echo $BS | cut -f1 -d-)
	BN=$(echo $BS | cut -f2 -d-)

	> knl/gemm_sparse.h
	> knl/gemm_dense.h
	> knl/gemm_libxsmm_sparse.h
	> knl/gemm_libxsmm_dense.h

	python3.6 ../sparsemmgen.py $M $N $K $LDA $LDB_SPARSE $LDC $BETA --bm $BM --bn $BN --bk $BK --arch knl --output_funcname gemm_sparse --output_filename arm/gemm_sparse.h --mtx_filename $MTX
	python3.6 ../sparsemmgen.py $M $N $K $LDA $LDB_DENSE $LDC $BETA --bm $BM --bn $BN --bk $BK --arch knl --output_funcname gemm_dense --output_filename arm/gemm_dense.h
	./../../../libxsmm/bin/libxsmm_gemm_generator sparse knl/gemms_libxsmm_sparse.h gemm_libxsmm_sparse $M $N $K $LDA $LDB_SPARSE $LDC 1 $BETA 1 1 knl nopf DP $MTXCSC
	./../../../libxsmm/bin/libxsmm_gemm_generator dense knl/gemms_libxsmm_dense.h gemm_libxsmm_dense $M $N $K $LDA $LDB_DENSE $LDC 1 $BETA 1 1 knl nopf DP

	g++ -std=c++11 -fopenmp testKNL.cpp

	export OMP_NUM_THREADS=124
	export KMP_AFFINITY=explicit,granularity=thread,proclist=[2-63,66-127]

	./a.out $M $N $K $BETA $NNZ $ITER $MTX

	printf " "
	printf $BM
	printf " "
	printf $BN
done
