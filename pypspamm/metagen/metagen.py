"""Emit one kernel per architecture, plus a dispatcher that picks at run time.

A single architecture produces the kernel on its own, exactly as before. Two or
more produce a static kernel each and one exported function that resolves to
the first kernel the processor supports. The last architecture given is the
fallback and is called when nothing else matches, so architectures have to be
listed from the most to the least capable.
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
"""

SVE_DETECTION = """
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#include <arm_sve.h>
#if defined(__clang__)
__attribute__((target("+sve")))
#else
__attribute__((target("arch=armv8-a+sve")))
#endif
static unsigned long pspamm_sve_bytes(void) {
  return (unsigned long)svcntb();
}
static unsigned long pspamm_sve_width(void) {
  /* svcntb may only be called once SVE is known to be present */
  if ((getauxval(AT_HWCAP) & HWCAP_SVE) == 0) { return 0; }
  return pspamm_sve_bytes();
}
#else
static unsigned long pspamm_sve_width(void) { return 0; }
#endif
"""

# target attribute arguments per detection kind; None means no attribute
TARGET_ATTRIBUTE = {
    "avx512": '"avx512f"',
    "avx512vl": '"avx512vl"',
    "avx2": '"avx2"',
    "neon": None,
    "sve": None,  # emitted with an explicit compiler switch, see below
}

SVE_ATTRIBUTE = """#if defined(__clang__)
__attribute__((target("+sve")))
#else
__attribute__((target("arch=armv8-a+sve")))
#endif
"""


def detection_kind(archspec, v_len):
    """The run time check this architecture and width need."""

    if archspec.detection == "avx512":
        # a 128 or 256 bit AVX-512 kernel needs the VL extension
        return "avx512" if v_len == 4 else "avx512vl"
    return archspec.detection


def condition(kind, v_len):
    if kind in ("avx512", "avx512vl", "avx2"):
        return f"__builtin_cpu_supports({TARGET_ATTRIBUTE[kind]})"
    if kind == "sve":
        return f"pspamm_sve_width() == {16 * v_len}"
    if kind == "neon":
        return "1"
    return None


def preamble(kinds):
    text = TARGET_MACROS
    if "sve" in kinds:
        text += SVE_DETECTION
    return text


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
        kind = detection_kind(archspec, v_len)
        attribute = ""
        if kind == "sve":
            attribute = SVE_ATTRIBUTE
        elif TARGET_ATTRIBUTE.get(kind) is not None:
            attribute = f"PSPAMM_TARGET({TARGET_ATTRIBUTE[kind]})\n"

        return alg, "static " + attribute + text.lstrip("\n")

    def dispatcher(self, name, ctype, entries):
        signature = (
            f"const {ctype}* A, const {ctype}* B, {ctype}* C, "
            f"{ctype} alpha, {ctype} beta, const {ctype}* prefetch"
        )
        pointer = f"pspamm_{name}_t"

        checks = ""
        for kernel, cond in entries[:-1]:
            checks += f"  if ({cond}) {{ return {kernel}; }}\n"
        fallback = entries[-1][0]

        return f"""
typedef void (*{pointer})({signature});

static {pointer} pspamm_resolve_{name}(void) {{
{checks}  return {fallback};
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

        kinds = set()
        entries = []
        text = ""
        ctype = None

        for arch in self.archs:
            archspec, v_len = registry.parse(arch)
            kind = detection_kind(archspec, v_len)
            assert kind is not None, (
                f"{arch} cannot be detected at run time and can only be "
                f"generated on its own"
            )
            kinds.add(kind)

            kernel_name = f"{name}_{arch}"
            alg, kernel = self.kernel(kernel_name, dict(params), arch, static=True)
            ctype = alg.precision.ctype()
            text += kernel + "\n"
            entries.append((kernel_name, condition(kind, v_len)))

        return preamble(kinds) + "\n" + text + self.dispatcher(name, ctype, entries)
