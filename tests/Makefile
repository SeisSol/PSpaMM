CXX=g++
CXXCLANG=clang++

CXXFLAGS=-std=c++17 -O3 -mcpu=native -fopenmp -Wall -g
CXXFLAGSNEON=-std=c++17 -O3 -mcpu=native -fopenmp -Wall -g

all: sve_testsuite neon_testsuite

sve_testsuite: build/sve_testsuite.cpp
	${CXX} ${CXXFLAGS} $< -o build/sve_testsuite

neon_testsuite: build/testsuite.cpp
	${CXX} ${CXXFLAGSNEON} $< -o build/neon_testsuite

clean:
	rm -f build/sve_testsuite
	rm -f build/neon_testsuite
