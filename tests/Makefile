CXX=g++

CXXFLAGS=-std=c++17 -Ofast -march=armv8.2-a+sve -msve-vector-bits=512 -fopenmp -Wall -g
CXXFLAGS2=-std=c++17 -Ofast -march=armv8.2-a -mcpu=a64fx -fopenmp -Wall -g

# if we use -march=armv8-a -> 2 of the 40 test cases in the unit tests for arm neon fail. Find out why 
CXXFLAGSNEON=-std=c++17 -Ofast -march=armv8.2-a -fopenmp -Wall -g# -Werror

#CXXFLAGS=-std=c++17 -O1 -march=native -msve-vector-bits=512 testsuite.cpp -o testsuite -falign-functions  -falign-jumps -falign-labels  -falign-loops -fcaller-saves -fcode-hoisting -fcrossjumping -fcse-follow-jumps  -fcse-skip-blocks  -fdelete-null-pointer-checks  -fdevirtualize  -fdevirtualize-speculatively -ffinite-loops -fgcse  -fgcse-lm  -fhoist-adjacent-loads -finline-functions -finline-small-functions -findirect-inlining  -fipa-bit-cp  -fipa-cp  -fipa-icf  -fipa-ra  -fipa-sra  -fipa-vrp -fisolate-erroneous-paths-dereference  -flra-remat  -foptimize-sibling-calls  -foptimize-strlen  -fpartial-inlining  -fpeephole2  -freorder-blocks-algorithm=stc  -freorder-blocks-and-partition  -freorder-functions  -frerun-cse-after-loop  -fschedule-insns
PAPIDIR=/home/ga27koz2/papi/src/install

all: benchmark_sve benchmark_neon

sve_testsuite: sve_testsuite.cpp
	${CXX} ${CXXFLAGS} $< -o sve_testsuite

neon_testsuite: testsuite.cpp
	${CXX} ${CXXFLAGSNEON} $< -o neon_testsuite

clean:
	rm -f sve_testsuite
	rm -f neon_testsuite
