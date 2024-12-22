
from typing import List
from pspamm.codegen.sugar import *

# TODO: We might eventually want to make this part of our syntax tree
# in order to do unrolls and other fancy stuff with it
class Loop(Block):

    _labels = []
    def __init__(self,
                 iteration_var: Register,
                 final_val: int,
                 body_contents: Block = None,
                 unroll: int = 1
                ) -> None:

        self.iteration_var = iteration_var
        self.final_val = final_val
        self.body_contents = body_contents
        self.unroll = unroll

        self.label = "loop_top_" + str(len(Loop._labels))
        Loop._labels.append(self.label)

        self.comment = "for {} <- {}:".format(self.iteration_var.ugly, 0) + \
                       "{}:{})".format(1, self.final_val)

    @property
    def contents(self):
        onestep = [*(self.body_contents.contents)]
        body = []
        rest = []
        for _ in range(self.unroll):
            body += onestep

        for _ in range(self.final_val % self.unroll):
            rest += onestep
        
        true_final_val = (self.final_val // self.unroll) * self.unroll

        allcode = []
        if true_final_val == self.unroll:
            allcode += body
        elif true_final_val > self.unroll:
            allcode += [mov(-true_final_val, self.iteration_var, vector=False),
                label(self.label)] + body + [add(self.unroll, self.iteration_var),
                jump(self.label, backwards=True)]
        allcode += rest

        return allcode

    def body(self, *args):
        self.body_contents = block("Loop body", *args)
        return self

def loop(iter_var, final_val, unroll=1):
    return Loop(iter_var, final_val, unroll=unroll)
