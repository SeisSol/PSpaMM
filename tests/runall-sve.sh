#!/usr/bin/env bash
# maybe do PYTHONPATH=$(pwd)/..:$PYTHONPATH

echo "GEMM test. Start."

for ARCH in arm128 arm_sve128 arm_sve256 arm_sve512 arm_sve1024 arm_sve2048 hsw128 hsw256 knl128 knl256 knl512 rvv128 rvv256 rvv512 rvv1024 rvv2048 rvv4096 rvv8192
do
    echo ""
    echo ""
    echo "Testing $ARCH"
    ./runlocal.sh $ARCH
done

echo "All tests done. Bye!"
