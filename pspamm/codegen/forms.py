
from typing import List
from pspamm.codegen.sugar import *

# TODO: We might eventually want to make this part of our syntax tree
# in order to do unrolls and other fancy stuff with it
class Loop(Block):

    _labels = []
    def __init__(self,
                 iteration_var: Register,
                 initial_val: int,
                 final_val: int,
                 increment: int = 1,
                 body_contents: Block = None,
                 unroll: int = 1
                ) -> None:

        self.iteration_var = iteration_var
        self.initial_val = initial_val
        self.final_val = final_val
        self.increment = increment
        self.body_contents = body_contents
        self.unroll = unroll
        assert self.unroll == 1 or self.initial_val == 0

        self.label = "loop_top_" + str(len(Loop._labels))
        Loop._labels.append(self.label)

        self.comment = "for {} <- {}:".format(self.iteration_var.ugly, self.initial_val) + \
                       "{}:{})".format(self.increment, self.final_val)

    @property
    def contents(self):
        onestep = [*(self.body_contents.contents),
                add(self.increment, self.iteration_var)]
        body = []
        rest = []
        for _ in range(self.unroll):
            body += onestep
        for _ in range(self.final_val % self.unroll):
            rest += onestep
        corrected_final_val = (self.final_val // self.unroll) * self.unroll
        return [mov(self.initial_val, self.iteration_var, vector=False),
                label(self.label)] + body + [cmp(corrected_final_val, self.iteration_var),
                jump(self.label, backwards=True)] + rest

    def body(self, *args):
        self.body_contents = block("Loop body", *args)
        return self

def loop(iter_var, initial_val, final_val, increment, unroll=1):
    return Loop(iter_var, initial_val, final_val, increment, unroll=unroll)
