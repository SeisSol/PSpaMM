CXX=g++
CXXCLANG=clang++

CXXFLAGS=-std=c++17 -O3 -mcpu=native -msve-vector-bits=512 -fopenmp -Wall -g
CXXFLAGSNEON=-std=c++17 -O3 -mcpu=native -fopenmp -Wall -g

all: sve_testsuite neon_testsuite

sve_testsuite: sve_testsuite.cpp
    ${CXXCLANG} ${CXXFLAGS} $< -o sve_testsuite

neon_testsuite: testsuite.cpp
    ${CXX} ${CXXFLAGSNEON} $< -o neon_testsuite

clean:
	rm -f sve_testsuite
	rm -f neon_testsuite
