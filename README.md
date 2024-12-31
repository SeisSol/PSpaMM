# PSpaMM

A Code Generator For Small Sparse (and Dense) Matrix Multiplications.

Currently supported:

* x86_64: AVX2, AVX512/AVX10.1
* ARM/AARCH64: NEON, SVE (128,256,512,1024,2048 bit)
* RISC-V: V (128,256,512,1024,2048,4096,8192 bit)

## Installation

PspaMM is a Python package. I.e. you may do

```bash

pip install .

```

to install it.

## Usage

```bash

pspamm-generator M N K LDA LDB LDC ALPHA BETA \
    --arch {arm,arm_sve{128..2048},knl{128..512},hsw{128..256},rvv{128..8192}} \
    --amtx_filename MTX_FILE_PATH --bmtx_filename MTX_FILE_PATH \
    --output_funcname FUNCTION_NAME --output_filename OUTPUT_NAME

```
