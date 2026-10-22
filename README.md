# PSpaMM

A code generator for small matrix multiplications.

Currently supported:

* x86_64: AVX2, AVX512/AVX10.1
* ARM/AARCH64: NEON, SVE (128,256,512,1024,2048 bit)
* RISC-V: V (128,256,512,1024,2048,4096,8192 bit)
* LoongArch: LSX, LASX

## Installation

PSpaMM is a Python package. I.e. after cloning, may install it via pip.

Alternatively, you can install it directly by running

```bash

pip install git+https://github.com/SeisSol/PSpaMM.git

```

## Usage

```bash

pspamm-generator M N K LDA LDB LDC ALPHA BETA \
    --arch {arm,arm_sve{128..2048},knl{128..512},hsw{128..256},rvv{128..8192},lsx{128..256}} \
    --amtx_filename MTX_FILE_PATH --bmtx_filename MTX_FILE_PATH \
    --output_funcname FUNCTION_NAME --output_filename OUTPUT_NAME

```

### A sparse A with known values

With `--abake`, the values in the file of a sparse A are the values of A: the
kernel keeps them in a table of its own and does not read A, and any pattern
works, not only one of whole vector tiles. Such a kernel works through A in
vector tiles (`--aformat blocked`) or in sliced ELLPACK, gathering the rows of
B it needs by permutations (`--aformat ellpack`, on AVX-512 and AVX2);
`--aformat auto`, the default, takes the one estimated to be faster.

`pspamm-generator --capabilities` prints what the generator supports, as JSON.
