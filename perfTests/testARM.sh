#TEST 1

M=8
N=56
K=56
LDA=8
LDB_DENSE=56
LDB_SPARSE=0
LDC=8
BETA=0
BM=8
BN=4
BK=1
NNZ=294
ITER=262144

> arm/gemm_sparse.h
> arm/gemm_dense.h

python3.6 ../sparsemmgen.py $M $N $K $LDA $LDB_SPARSE $LDC $BETA --bm $BM --bn $BN --bk $BK --arch arm --output_funcname gemm_sparse --output_filename arm/gemm_sparse.h --mtx_filename mtx/56x56.mtx
python3.6 ../sparsemmgen.py $M $N $K $LDA $LDB_DENSE $LDC $BETA --bm $BM --bn $BN --bk $BK --arch arm --output_funcname gemm_sparse --output_filename arm/gemm_dense.h

srun g++ -std=c++11 -fopenmp testARM.cpp
srun a.out $M $N $K $BETA $NNZ $ITER mtx/56x56.mtx