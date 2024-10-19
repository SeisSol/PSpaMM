#!/usr/bin/env bash
# maybe do PYTHONPATH=$(pwd)/..:$PYTHONPATH

echo "SVE GEMM test. Right now, we do not test all multiples of 128 bit. Mostly powers of two, since gcc may not support others."

for BITLEN in 128 256 512 1024 2048
do
    echo ""
    echo ""
    echo "Testing $BITLEN bit SVE register GEMM"
    python unit_test.py arm_sve$BITLEN
    aarch64-linux-gnu-g++ -static -march=armv8.2-a+sve -msve-vector-bits=${BITLEN} build/arm_sve${BITLEN}_testsuite.cpp -o build/sve${BITLEN}-test
    qemu-aarch64-static -cpu max,sve${BITLEN}=on,sve-default-vector-length=-1 build/sve${BITLEN}-test
done

echo "All tests done. Bye!"
