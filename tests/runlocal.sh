#!/usr/bin/env bash
# maybe do PYTHONPATH=$(pwd)/..:$PYTHONPATH

echo "GEMM test for arch ${1}"
python unit_test.py ${1}

if [[ ${1:0:7} == "arm_sve" ]]; then
    BITLEN=${1:7:10}
    aarch64-linux-gnu-g++ -static -march=armv8.2-a+sve -msve-vector-bits=${BITLEN} build/${1}_testsuite.cpp -o build/${1}-test
    if [[ ${2} != "norun" ]]; then
        qemu-aarch64-static -cpu max,sve${BITLEN}=on,sve-default-vector-length=-1 build/${1}-test
    fi
elif [[ ${1:0:3} == "arm" ]]; then
    BITLEN=${1:3:6}
    aarch64-linux-gnu-g++ -static -march=armv8.2-a build/${1}_testsuite.cpp -o build/${1}-test
    if [[ ${2} != "norun" ]]; then
        qemu-aarch64-static -cpu max build/${1}-test
    fi
elif [[ ${1:0:3} == "rvv" ]]; then
    BITLEN=${1:3:6}
    riscv64-linux-gnu-g++ -static -march=rv64g_v_zvl${BITLEN}b_zicbop build/${1}_testsuite.cpp -o build/${1}-test
    if [[ ${2} != "norun" ]]; then
        qemu-riscv64-static -cpu max,g=on,v=on,vlen=${BITLEN} build/${1}-test
    fi
elif [[ ${1:0:3} == "hsw" ]]; then
    BITLEN=${1:3:6}
    g++ -static -mavx2 build/${1}_testsuite.cpp -o build/${1}-test
    if [[ ${2} != "norun" ]]; then
        qemu-x86_64-static -cpu Haswell build/${1}-test
    fi
elif [[ ${1:0:3} == "knl" ]]; then
    BITLEN=${1:3:6}
    g++ -static -mavx512f build/${1}_testsuite.cpp -o build/${1}-test
    if [[ ${2} != "norun" ]]; then
        qemu-x86_64-static -cpu Skylake-Server build/${1}-test
    fi
elif [[ ${1:0:3} == "lsx" ]]; then
    BITLEN=${1:3:6}
    # TODO: once established, remove the -14
    loongarch64-linux-gnu-g++-14 -static -mlasx build/${1}_testsuite.cpp -o build/${1}-test
    if [[ ${2} != "norun" ]]; then
        qemu-loongarch64-static -cpu max build/${1}-test
    fi
fi
