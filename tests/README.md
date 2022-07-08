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
