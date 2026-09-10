"""What each code generator is called, and what it needs at run time.

The command line names an architecture together with a vector width, such as
``knl512`` or ``arm_sve256``. This module maps such a name onto the generator
module that implements it and onto the width in units of 128 bit, and it
carries the information the meta generator needs to pick between several
kernels at run time.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Architecture:
    prefix: str
    module: str
    widths: Tuple[int, ...]
    default_width: int
    family: str
    # how the presence of this architecture is established at run time; None
    # means it cannot be detected and the kernel can only be used on its own
    detection: Optional[str] = None


X86_WIDTHS = (128, 256)
AVX512_WIDTHS = (128, 256, 512)
SVE_WIDTHS = tuple(range(128, 2049, 128))
RVV_WIDTHS = (128, 256, 512, 1024, 2048, 4096, 8192)
LSX_WIDTHS = (128, 256)

# Longer prefixes come first, so that arm_sve is not read as arm.
ARCHITECTURES = (
    Architecture("arm_sve", "arm_sve", SVE_WIDTHS, 512, "aarch64", "sve"),
    Architecture("skx", "knl", AVX512_WIDTHS, 512, "x86", "avx512"),
    Architecture("knl", "knl", AVX512_WIDTHS, 512, "x86", "avx512"),
    Architecture("hsw", "hsw", X86_WIDTHS, 256, "x86", "avx2"),
    Architecture("rvv", "rvv", RVV_WIDTHS, 128, "riscv", None),
    Architecture("lasx", "lsx", LSX_WIDTHS, 256, "loongarch", None),
    Architecture("lsx", "lsx", LSX_WIDTHS, 128, "loongarch", None),
    Architecture("arm", "arm", (128,), 128, "aarch64", "neon"),
)

# Names that stand for a whole set of kernels plus a dispatcher.
PRESETS = {
    "aarch64": (
        "arm_sve2048",
        "arm_sve1024",
        "arm_sve512",
        "arm_sve256",
        "arm_sve128",
        "arm128",
    ),
    "x86_64": ("knl512", "knl256", "hsw256"),
}


def architecture_names():
    return tuple(a.prefix for a in ARCHITECTURES) + tuple(PRESETS)


def expand(name: str) -> Tuple[str, ...]:
    """Resolve a preset, or pass a single architecture name through."""
    if name in PRESETS:
        return PRESETS[name]
    return (name,)


def parse(name: str) -> Tuple[Architecture, int]:
    """Split an architecture name into its generator and its width.

    Returns the architecture and the width in units of 128 bit.
    """

    for arch in ARCHITECTURES:
        if not name.startswith(arch.prefix):
            continue
        suffix = name[len(arch.prefix) :]
        if suffix == "":
            width = arch.default_width
        else:
            if not suffix.isdigit():
                raise ValueError(f"{name}: {suffix!r} is not a vector width")
            width = int(suffix)
        if width not in arch.widths:
            raise ValueError(
                f"{name}: {arch.prefix} supports the widths "
                f"{', '.join(str(w) for w in arch.widths)}, not {width}"
            )
        return arch, width // 128

    raise ValueError(
        f"{name}: unknown architecture, expected one of "
        f"{', '.join(architecture_names())}"
    )
