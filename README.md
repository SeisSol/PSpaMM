# Code Generator for Sparse Matrix Multiplication
Generates inline-Assembly for sparse Matrix Multiplication.

Currently Intel Xeon Phi 'Knights Landing' (AVX512) and ARM Cortex-A53 (ARMv8) are supported.

Usage: 
./pspamm M N K LDA LDB LDC ALPHA BETA 
--arch arm/knl --mtx_filename FILE 
--output_funcname FUNCTION_NAME --output_filename FILE_NAME
