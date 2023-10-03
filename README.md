# Code Generator for Sparse Matrix Multiplication
Generates inline-Assembly for sparse Matrix Multiplication.

Currently Intel Xeon Phi 'Knights Landing' (AVX512), Haswell/Zen2 (AVX2), and ARM Cortex-A53 (ARMv8) are supported.

## Installation

PspaMM is a Python package. I.e. you may do

```
pip install .
```

to install it.

## Usage 

```
pspamm-generator M N K LDA LDB LDC ALPHA BETA --arch {arm,arm_sve{128,256,512,1024,2048},knl,hsw} \
    --mtx_filename MTX_FILE_PATH --output_funcname FUNCTION_NAME --output_filename OUTPUT_NAME

```
