from pspamm.matmul import MatMul
from pspamm.codegen.ccode import *

class MetaGenerator:
    def __init__(self):
        self.conditions = []
        self.archs = []
    
    def add_condition(self, condition, arch):
        self.conditions += [condition]
        self.archs += arch
    
    def generate_meta(self, funcname, params):
        condition_template = "        if ({condition}) {{ func = {funcname}_{arch}; }}\n"

        template = """
void {funcname}({params}) {{
    typedef void(*func_template)({params});
    static func_template func = NULL;

    if (func == NULL) {{
        {conditions}
    }}

    func({params});
}}
        """

        conditions = ""
        for (condition, arch) in zip(self.conditions, self.archs):
            conditions += condition_template.format(funcname=funcname, arch=arch, condition=condition)
        
        return template.format(funcname=funcname, params=params, conditions=conditions)

    def generate(self, alg: MatMul):
        block = alg.make()

        return make_cfunc(alg.output_funcname, alg.generator.get_template(), block, alg.flop, alg.starting_regs, alg.generator.get_precision())

        if len(self.archs) == 0:
            return ""
        
        if len(self.archs) == 1:
            block = alg.make()

            return make_cfunc(alg.output_funcname, alg.generator.get_template(), block, alg.flop, alg.starting_regs, alg.generator.get_precision())
            # only generate the kernel; nothing else
        else:
            text = ""
            # generate all kernels, and the meta
            for arch in self.archs:
                block = alg.make()

                funcname = f'{alg.output_funcname}_{arch}'

                func = make_cfunc(funcname, alg.generator.get_template(), block, alg.flop, alg.starting_regs, alg.generator.get_precision())

                text += f"static {func}\n\n"

            params = f"const {alg.precision.ctype()}* A, const {alg.precision.ctype()}* B, {alg.precision.ctype()}* C, {alg.precision.ctype()} alpha, {alg.precision.ctype()} beta, const {alg.precision.ctype()}* prefetch"    
            text += self.generate_meta(alg.output_funcname, params)
            text += "\n\n"

            return text
