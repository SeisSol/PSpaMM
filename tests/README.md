# Test Running Guidelines

Run `runall-sve.sh` which tests all configurations using software emulation. Note that on a processor that does not support AVX512/AVX10, you might need to comment out some tests, since there is no known emulation method for them yet.

## Running a single test

Use `runlocal.sh` with the PSpaMM architecture of your choice (e.g. `knl512`). The script will also automatically execute the tests; unless you give it the `norun` flag as second argument.

## Debugging

For debugging, for example for SVE with vector length 512
(cf. <https://mariokartwii.com/showthread.php?tid=1998> ):

```bash
aarch64-linux-gnu-g++ -g -ggdb -static -march=armv9-a+sve -msve-vector-bits=512 sve_testsuite.cpp
qemu-aarch64-static -g 1234 -cpu max,sve512=on ./a.out
```

(we use 1234 as port here, and a.out as filename)

In a separate window, run
`aarch64-linux-gnu-gdb --ex "target remote localhost:1234" --ex "file a.out"`.
The extra commands already connect you with
QEMU and attach you to the compiled binary file,
so method names etc. are printed correctly.
To run the program, just type `continue`. You
may maybe want to set up breakpoints etc. before you do that.
