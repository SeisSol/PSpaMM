from importlib import import_module

def init():
    global arch
    global generator
    global operands
    arch = "none"
    generator = None
    operands = None

def get_class( kls ):
    return import_module(kls)
