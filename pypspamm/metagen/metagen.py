"""Emit one kernel per architecture, plus a dispatcher that picks at run time.

A single architecture produces the kernel on its own. Two or more produce a
static kernel each and one exported function that resolves to the first kernel
the processor supports. The last architecture given is the fallback and is
called when nothing else matches, so architectures have to be listed from the
most to the least capable, and the last one has to be one the target is known
to have.

Kernels whose vector length is fixed by the instruction set are matched
exactly. RVV kernels set the vector length explicitly, so a kernel built for a
narrower register file also runs on a wider one and is matched with at least.
"""

from pypspamm.codegen.architectures import registry
from pypspamm.codegen.ccode import make_cfunc
from pypspamm.matmul import MatMul

TARGET_MACROS = """
#if defined(__GNUC__) || defined(__clang__)
#define PSPAMM_TARGET(...) __attribute__((target(__VA_ARGS__)))
#else
#define PSPAMM_TARGET(...)
#endif
#if defined(__GNUC__) || defined(__clang__)
#define PSPAMM_MAYBE_UNUSED __attribute__((unused))
#else
#define PSPAMM_MAYBE_UNUSED
#endif
#if defined(__clang__)
#define PSPAMM_TARGET_SVE PSPAMM_TARGET("+sve")
#define PSPAMM_TARGET_RVV PSPAMM_TARGET("+v")
#else
#define PSPAMM_TARGET_SVE PSPAMM_TARGET("arch=armv8-a+sve")
#define PSPAMM_TARGET_RVV PSPAMM_TARGET("arch=+v")
#endif
"""

SVE_DETECTION = """
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#include <arm_sve.h>
PSPAMM_TARGET_SVE
static unsigned long pspamm_sve_bytes(void) { return (unsigned long)svcntb(); }
PSPAMM_MAYBE_UNUSED static unsigned long pspamm_sve_width(void) {
  /* svcntb faults without SVE, so it is only reached past this check */
  if ((getauxval(AT_HWCAP) & HWCAP_SVE) == 0) { return 0; }
  return pspamm_sve_bytes();
}
#else
PSPAMM_MAYBE_UNUSED static unsigned long pspamm_sve_width(void) { return 0; }
#endif
"""

RVV_DETECTION = """
#if defined(__riscv) && defined(__linux__)
#include <sys/auxv.h>
#define PSPAMM_RISCV_HWCAP_V (1UL << ('V' - 'A'))
PSPAMM_MAYBE_UNUSED static unsigned long pspamm_rvv_width(void) {
  unsigned long vlenb;
  if ((getauxval(AT_HWCAP) & PSPAMM_RISCV_HWCAP_V) == 0) { return 0; }
  /* vlenb is an unprivileged read only CSR and traps without the V extension */
  __asm__ __volatile__("csrr %0, vlenb" : "=r"(vlenb));
  return vlenb;
}
#else
PSPAMM_MAYBE_UNUSED static unsigned long pspamm_rvv_width(void) { return 0; }
#endif
"""

LOONGARCH_DETECTION = """
#if defined(__loongarch__) && defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
PSPAMM_MAYBE_UNUSED static int pspamm_has_lsx(void) {
  return (getauxval(AT_HWCAP) & HWCAP_LOONGARCH_LSX) != 0;
}
PSPAMM_MAYBE_UNUSED static int pspamm_has_lasx(void) {
  return (getauxval(AT_HWCAP) & HWCAP_LOONGARCH_LASX) != 0;
}
#else
PSPAMM_MAYBE_UNUSED static int pspamm_has_lsx(void) { return 0; }
PSPAMM_MAYBE_UNUSED static int pspamm_has_lasx(void) { return 0; }
#endif
"""

DETECTION_BLOCKS = {
    "sve": SVE_DETECTION,
    "rvv": RVV_DETECTION,
    "lsx": LOONGARCH_DETECTION,
    "lasx": LOONGARCH_DETECTION,
}

# per detection kind: the attribute a kernel carries, and how its presence is
# tested. {bytes} is the vector register size the kernel was generated for.
KINDS = {
    "avx512": ('PSPAMM_TARGET("avx512f")', '__builtin_cpu_supports("avx512f")'),
    "avx512vl": ('PSPAMM_TARGET("avx512vl")', '__builtin_cpu_supports("avx512vl")'),
    "avx2": ('PSPAMM_TARGET("avx2")', '__builtin_cpu_supports("avx2")'),
    "neon": ("", "1"),
    "sve": ("PSPAMM_TARGET_SVE", "pspamm_sve_width() == {bytes}"),
    "rvv": ("PSPAMM_TARGET_RVV", "pspamm_rvv_width() >= {bytes}"),
    "lsx": ('PSPAMM_TARGET("lsx")', "pspamm_has_lsx()"),
    "lasx": ('PSPAMM_TARGET("lasx")', "pspamm_has_lasx()"),
}


def detection_kind(archspec, v_len):
    """The run time check this architecture at this width needs."""

    if archspec.detection == "avx512":
        # below 512 bit an AVX-512 kernel needs the VL extension
        return "avx512" if v_len == 4 else "avx512vl"
    if archspec.detection == "lsx":
        return "lsx" if v_len == 1 else "lasx"
    return archspec.detection


class MetaGenerator:
    def __init__(self, archs):
        self.archs = [name for entry in archs for name in registry.expand(entry)]

    def kernel(self, name, params, arch, static):
        """Generate one kernel; the architecture is selected here, not by the caller."""

        alg = MatMul(arch=arch, output_funcname=name, **params)
        text = make_cfunc(
            name,
            alg.generator.get_template(),
            alg.make(),
            alg.flop,
            alg.starting_regs,
            alg.generator.get_precision(),
        )
        if not static:
            return alg, text

        archspec, v_len = registry.parse(arch)
        attribute = KINDS[detection_kind(archspec, v_len)][0]
        if attribute:
            attribute += "\n"
        return alg, "static " + attribute + text.lstrip("\n")

    def dispatcher(self, name, ctype, entries):
        signature = (
            f"const {ctype}* A, const {ctype}* B, {ctype}* C, "
            f"{ctype} alpha, {ctype} beta, const {ctype}* prefetch"
        )
        pointer = f"pspamm_{name}_t"

        checks = "".join(
            f"  if ({cond}) {{ return {kernel}; }}\n" for kernel, cond in entries[:-1]
        )

        return f"""
typedef void (*{pointer})({signature});

static {pointer} pspamm_resolve_{name}(void) {{
{checks}  return {entries[-1][0]};
}}

void {name}({signature}) {{
#if defined(__cplusplus)
  static const {pointer} chosen = pspamm_resolve_{name}();
#else
  /* the resolver is idempotent, so a race here picks the same kernel twice */
  static {pointer} chosen = 0;
  if (chosen == 0) {{ chosen = pspamm_resolve_{name}(); }}
#endif
  chosen(A, B, C, alpha, beta, prefetch);
}}
"""

    def generate(self, params):
        name = params.pop("output_funcname", None)

        if len(self.archs) == 1:
            _, text = self.kernel(name, params, self.archs[0], static=False)
            return text

        assert name is not None, "a dispatched kernel needs an output function name"

        detection = ""
        entries = []
        kernels = ""
        ctype = None

        for arch in self.archs:
            archspec, v_len = registry.parse(arch)
            kind = detection_kind(archspec, v_len)
            assert (
                kind is not None
            ), f"{arch} has no run time check and cannot be dispatched"

            block = DETECTION_BLOCKS.get(kind, "")
            if block and block not in detection:
                detection += block

            kernel_name = f"{name}_{arch}"
            alg, kernel = self.kernel(kernel_name, dict(params), arch, static=True)
            ctype = alg.precision.ctype()
            kernels += kernel + "\n"
            entries.append((kernel_name, KINDS[kind][1].format(bytes=16 * v_len)))

        return (
            TARGET_MACROS
            + detection
            + "\n"
            + kernels
            + self.dispatcher(name, ctype, entries)
        )
