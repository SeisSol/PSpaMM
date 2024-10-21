# PSpaMM
A Code Generator For Small Sparse (and Dense) Matrix Multiplications.

Currently supported:

* x86_64: AVX2, AVX512/AVX10.1
* ARM/AARCH64: NEON, SVE (128,256,512,1024,2048 bit)

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
