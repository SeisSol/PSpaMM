# Guidelines on how to execute the NEON & SVE tests  
### DISCLAIMER: 
Some unit tests for SVE fail when including the gcc compiler flag "-mcpu=a64fx". 
We assume that this flag makes the compiler optimize the unit tests in a way that breaks them. 
Specifically, the values for `ldb`, `alpha`, and `beta` are sometimes set to 0 when calculating a reference solution which we compare to the solution of the PSpaMM kernel. 
To fix this, the generated testsuite for Arm NEON and SVE saves certain values as variables before passing them to specific functions
instead of passing them as constant values.
## Compiling with gcc
A Makefile is provided, however only NEON and SVE related unit tests can be compiled at the moment. 
Naturally, other compiler flags than the ones provided may be used.  
Compiling the SVE testsuite with gcc 11.0.0 seems to break some test cases. Within the provided test setup, the values of certain parameters are overwritten after
specific tests, namely ```sve_arm_only_test15_23_6.h``` and ```sve_arm_only_test16_23_6.h```. This leads to a wrong reference solution which is then compared to 
the one calculated by our generated kernel.  
Example output when using GDB:
```
Program received signal SIGSEGV, Segmentation fault.
#0  0x0000000000213dc4 in post<double> (M=7, M@entry=23, N=N@entry=29, K=K@entry=31, LDA=7, LDA@entry=23, LDB=0x3feeb851eb851eb8, LDB@entry=0xffffffffa5ec, LDC=7, LDC@entry=23, A=A@entry=0x2fa8c0, 
    B=B@entry=0x2fbf40, C=C@entry=0x300cc0, Cref=Cref@entry=0x2ff7c0, DELTA=DELTA@entry=9.9999999999999995e-08, BETA=<optimized out>, ALPHA=<optimized out>, BETA=<optimized out>, 
    ALPHA=<optimized out>) at sve_testsuite.cpp:179
#1  0x00000000002aecf4 in main () at sve_testsuite.cpp:619
```
Additional testing of gcc-based compilation is needed. Meanwhile, the SVE testsuite should be compiled with clang.
## Unit Tests
Unit tests for all 3 architectures (KNL, Arm NEON, Arm SVE) are provided. 
The testsuite that corresponds to a unit test needs to be executed on the respective processor/architecture. 
How to generate and execute a specific testsuite is shown below. 
If nothing breaks, the generated testsuite reports the number of successful test case executions.
### KNL
1. Generate the testsuite by calling ```python3 unit_tests_knl.py```
2. Adjust the Makefile as needed and compile the generated ```testsuite.cpp```
3. Run the compiled executable

### Arm NEON
1. Generate the testsuite by calling ```python3 unit_tests_arm.py```
2. Adjust the Makefile as needed and compile the generated ```testsuite.cpp``` by calling  
```make neon_testsuite```
3. Run the compiled executable with ```./neon_testsuite```

### Arm SVE 
1. Generate the testsuite by calling ```python3 unit_tests_arm_sve.py```
2. Adjust the Makefile as needed and compile the generated ```sve_testsuite.cpp``` by calling  
```make sve_testsuite```
3. Run the compiled executable with ```./sve_testsuite```

#### Notes Running SVE with QEMU user-static

Run `runall-sve.sh` which tests a bunch of configurations already.

For a bit length `BITLEN`, it executes the following commands:
```
# generate tests
python unit_tests_arm_sve.py $BITLEN

# compile: we use AVM V8.2 and SVE; the SVE vector length is set explicitly
aarch64-linux-gnu-g++ -static -march=armv8.2-a+sve -msve-vector-bits=${BITLEN} arm_sve${BITLEN}_testsuite.cpp -o sve${BITLEN}-test

# run using QEMU, this way we may run on x86-64 as well; enable all features and constrain to sve${BITLEN} SVE registers maximum length (cf. https://qemu-project.gitlab.io/qemu/system/arm/cpu-features.html); the sve-default-vector-length=-1 parameter is needed for 1024 and 2048 bit SVE to work correctly (otherwise, QEMU will assume 512 bit maximum)
qemu-aarch64-static -cpu max,sve${BITLEN}=on,sve-default-vector-length=-1 ./sve${BITLEN}-test
```


For debugging, for example for vector length 512 (cf. https://mariokartwii.com/showthread.php?tid=1998 ):
```
aarch64-linux-gnu-g++ -g -ggdb -static -march=armv9-a+sve -msve-vector-bits=512 sve_testsuite.cpp
qemu-aarch64-static -g 1234 -cpu max,sve512=on ./a.out
```
(we use 1234 as port here, and a.out as filename)

In a separate window, run `aarch64-linux-gnu-gdb --ex "target remote localhost:1234" --ex "file a.out"`.
The extra commands already connect you with QEMU and attach you to the compiled binary file, so method names etc. are printed correctly.
To run the program, just type `continue`. You may maybe want to set up breakpoints etc. before you do that.
