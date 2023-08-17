# Code Generator for Sparse Matrix Multiplication
Generates inline-Assembly for sparse Matrix Multiplication.

Currently Intel Xeon Phi 'Knights Landing' (AVX512), Haswell/Zen2 (AVX2), and ARM Cortex-A53 (ARMv8) are supported.

Usage: 

./pspamm M N K LDA LDB LDC ALPHA BETA --arch {arm,arm_sve,knl,hsw}

--mtx_filename MTX_FILE_PATH --output_funcname FUNCTION_NAME --output_filename OUTPUT_NAME
